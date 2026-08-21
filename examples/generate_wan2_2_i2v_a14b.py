# End-to-end image-to-video inference script for Wan2.2 A14B on TPU.
#
# Like generate_wan2_2_t2v_a14b.py, A14B is a two-expert (high_noise_model /
# low_noise_model) Mixture-of-Experts model switched per sampling step by
# timestep vs. `--boundary` -- see that script's header comment for the full
# mechanism, unchanged here.
#
# Unlike Wan2.1's I2V-14B, A14B has no CLIP vision cross-attention branch at
# all (Wan2.2's WanDiT never had one -- see `vidax.models.wan.wan2_2.dit`'s
# module docstring). Image conditioning here instead concatenates a
# mask+VAE-latent `y` (built exactly like Wan2.1 I2V's, from Wan2.1's causal
# VAE) directly onto the noisy latent's channel axis *before* the DiT call --
# matches the reference `WanModel.forward`'s `x = cat([x, y], dim=channel)`,
# and is why `configs.I2V_A14B_CONFIG` sets `in_dim=36` (16 noise channels +
# 20 conditioning channels) instead of Wan2.1 I2V's separate-argument `y`.

import argparse
from functools import partial
import logging
import math
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from vidax.core.sharding import (
    build_tpu_mesh, shard_wan_params, get_replicated_sharding, get_batch_sharding,
    to_partition_specs, configure_jax_cache,
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_2.configs import I2V_A14B_CONFIG
from vidax.models.wan.wan2_2.dit import WanDiT, Wan22DiTBlock
from vidax.models.wan.wan2_1.vae import (
    WanVAEDecoder, WanVAEEncoder, Decoder3d, Encoder3d, _count_causal_convs,
    _count_causal_convs_encoder,
)
from vidax.models.wan.common.t5 import T5Encoder, Umt5Tokenizer
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Saves a sequence of frames as an MP4 video."""
    logging.info(f"Saving {frames.shape[0]} frames to {output_path}...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info("Video saved successfully.")


def cast_to_dtype(tree, dtype):
    """Casts every floating-point leaf of a pytree to `dtype`."""
    def cast_leaf(x):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def encode_prompts(prompts: list, t5_model: T5Encoder, t5_params, tokenizer: Umt5Tokenizer,
                    dtype) -> jnp.ndarray:
    """Tokenizes and T5-encodes one prompt per batch element."""
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)
    context = t5_model.apply(t5_params, ids, mask)
    seq_lens = mask.sum(axis=1)
    positions = jnp.arange(context.shape[1])[None, :]
    keep = positions < seq_lens[:, None]
    return jnp.where(keep[..., None], context, 0.0).astype(dtype)


def compute_latent_grid(image_h: int, image_w: int, max_area: int,
                         vae_stride: tuple, patch_size: tuple) -> tuple:
    """Reference's aspect-ratio-preserving resolution selection, identical
    to generate_wan2_1_i2v.py's copy (same VAE stride, same DiT patch size).
    """
    aspect_ratio = image_h / image_w
    lat_h = round(
        math.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(
        math.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    return lat_h * vae_stride[1], lat_w * vae_stride[2], lat_h, lat_w


def build_i2v_conditioning(image: np.ndarray, num_frames: int, pixel_h: int, pixel_w: int,
                            latent_t: int, vae_model: WanVAEEncoder, vae_params, dtype):
    """Builds `y` (mask + VAE-encoded conditioning frame): identical
    construction to generate_wan2_1_i2v.py's copy of this function (same
    Wan2.1 causal VAE) -- the only difference from that script is *how* the
    caller uses the result (concatenated onto the noisy latent here, instead
    of passed as a separate model argument), which happens in main() below.
    """
    img = jnp.asarray(image, dtype=jnp.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
    img = jax.image.resize(img, (pixel_h, pixel_w, 3), method="bicubic")

    video = jnp.zeros((num_frames, pixel_h, pixel_w, 3), dtype=jnp.float32)
    video = video.at[0].set(img)
    video = video[None].astype(dtype)

    x_full = video
    encode_chunk_jit = jax.jit(
        lambda params, x_chunk, cache_list: vae_model.apply(
            params, x_chunk, cache_list, method=vae_model.encode_chunk))

    encoder_cfg = Encoder3d(
        vae_model.dim, vae_model.z_dim * 2, vae_model.dim_mult, vae_model.num_res_blocks,
        vae_model.attn_scales, vae_model.temperal_downsample, vae_model.eps)
    cache_list = [None] * _count_causal_convs_encoder(encoder_cfg)
    t = x_full.shape[1]
    bounds = [(0, 1)] + [(1 + 4 * i, 1 + 4 * (i + 1)) for i in range((t - 1) // 4)]
    encoded_chunks = []
    for start, end in bounds:
        out_chunk, cache_list = encode_chunk_jit(vae_params, x_full[:, start:end], cache_list)
        encoded_chunks.append(out_chunk)
    out_full = jnp.concatenate(encoded_chunks, axis=1)
    latents = vae_model.apply(vae_params, out_full, method=vae_model.post_process)
    lat_h, lat_w = latents.shape[2], latents.shape[3]

    mask = jnp.zeros((1, latent_t, lat_h, lat_w, 4), dtype=dtype)
    mask = mask.at[:, 0].set(1.0)

    return jnp.concatenate([mask, latents.astype(dtype)], axis=-1)  # (1, latent_t, lat_h, lat_w, 20)


def main(args):
    """Main inference function."""
    configure_jax_cache()
    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size
    sp_size = args.sequence_parallel_size
    assert num_devices % (tp_size * sp_size) == 0, (
        f"num_devices ({num_devices}) must be divisible by "
        f"--tensor_parallel_size * --sequence_parallel_size ({tp_size} * {sp_size})")
    dp_size = num_devices // (tp_size * sp_size)
    mesh = build_tpu_mesh(
        data_parallel_size=dp_size, tensor_parallel_size=tp_size,
        sequence_parallel_size=sp_size)
    rng = jax.random.PRNGKey(args.seed)
    logging.info(
        f"Using {num_devices} devices: {dp_size}-way data // {tp_size}-way tensor // "
        f"{sp_size}-way sequence parallel.")

    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]
    sequence_parallel = sp_size > 1

    # --- Initialize models and scheduler ---
    dit_model = WanDiT(
        mesh=mesh, sequence_parallel=sequence_parallel, compute_dtype=dit_dtype,
        **I2V_A14B_CONFIG)
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder()
    t5_model = T5Encoder()
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=args.shift)
    boundary_val = args.boundary * scheduler.num_train_timesteps

    assert dit_model.num_heads % (tp_size * sp_size) == 0, (
        f"WanDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size * --sequence_parallel_size ({tp_size} * {sp_size}).")
    assert t5_model.num_heads % tp_size == 0, (
        f"T5Encoder.num_heads ({t5_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")

    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(args.t5_checkpoint_path), "google", "umt5-xxl")
    tokenizer = Umt5Tokenizer(tokenizer_path, seq_len=dit_model.text_len)

    # --- Load weights (both DiT experts, VAE, and T5 ship as separate checkpoints) ---
    logging.info(f"Loading high-noise DiT weights from {args.high_noise_dit_checkpoint_path}...")
    high_dit_params = load_torch_checkpoint_to_jax(
        args.high_noise_dit_checkpoint_path, model_type="wan2.2_dit")
    logging.info(f"Loading low-noise DiT weights from {args.low_noise_dit_checkpoint_path}...")
    low_dit_params = load_torch_checkpoint_to_jax(
        args.low_noise_dit_checkpoint_path, model_type="wan2.2_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(
        args.vae_checkpoint_path, model_type="wan2.1_vae")
    logging.info(f"Loading T5 weights from {args.t5_checkpoint_path}...")
    t5_params = load_torch_checkpoint_to_jax(
        args.t5_checkpoint_path, model_type="wan_t5")

    # Unlike every other script, the two DiT experts are *not* both put on
    # device here -- see generate_wan2_2_t2v_a14b.py's identical comment for
    # why (single-resident-expert swapping instead of both at once, since
    # this repo's 4-chip TP=4 target doesn't have headroom for both).
    # `--sequence_parallel_size` composes with this unchanged, same as that
    # script. `--offload_dit_weights` (below) generalizes this further: even
    # the one currently-needed expert doesn't have to be fully device-
    # resident at once -- see docs/weight_offloading.md.
    replicated = get_replicated_sharding(mesh)
    # `dit_dtype` (default float32), independent of `--dtype` -- see
    # generate_wan2_1_t2v.py's identical comment. Wan2.2's `WanDiT` never
    # got this repo's Wan2.1 precision fix ported over until now: at native
    # 720P's large token count, keeping DiT weights (and the residual
    # stream between blocks, see `Wan22DiTBlock`'s docstring) at bf16
    # produces visibly noisy/corrupted output -- see docs/models/wan2_2.md#status.
    high_dit_params = cast_to_dtype(high_dit_params, dit_dtype)
    low_dit_params = cast_to_dtype(low_dit_params, dit_dtype)
    vae_params = cast_to_dtype(vae_params, dtype)
    t5_params = cast_to_dtype(t5_params, dtype)

    dit_sharding_spec = shard_wan_params(high_dit_params, mesh)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)

    # `--offload_dit_weights`: generalizes A14B's existing whole-expert host/
    # device swap (above) to per-layer granularity, exactly like
    # generate_wan2_1_t2v.py/`_i2v.py`'s identical flag -- see
    # docs/weight_offloading.md. Composes with the two-expert MoE switch
    # naturally: since every layer's weights get a fresh `device_put` each
    # step regardless of chunk size anyway (see that doc's "Real cost"
    # section), there's no separate "expert changed" bookkeeping needed here
    # -- each step just offloads whichever expert's chunks the current
    # timestep calls for, chosen fresh every step from host memory. Only the
    # two experts' small `nonblock_params` (patch/time/text-embed + head)
    # stay permanently device-resident, mirroring how the non-offloaded path
    # above already keeps VAE/T5 resident throughout.
    if args.offload_dit_weights:
        # Unlike generate_wan2_1_t2v.py/`_i2v.py`, this *does* compose with
        # `--sequence_parallel_size > 1` -- see the shard_map-wrapped
        # `pre_apply`/`chunk_forward`/`post_apply` built below, needed
        # because A14B's per-*token* modulation (`e0`, unlike Wan2.1's
        # per-sample one -- see `vidax.models.wan.wan2_2.dit`'s module
        # docstring) makes activation memory, not just weight residency, a
        # real constraint at native 720P/81 frames: offloading alone (i.e.
        # `--sequence_parallel_size 1`) reduces peak HBM but still doesn't
        # fit there (measured ~56.6G required vs. ~61.7G without offloading,
        # both over this chip's ~30.75G budget) -- sequence_parallel is what
        # actually shrinks the per-token activations themselves.
        chunk_size = args.offload_chunk_size
        assert dit_model.num_layers % chunk_size == 0, (
            f"--offload_chunk_size ({chunk_size}) must divide WanDiT.num_layers "
            f"({dit_model.num_layers}) -- see docs/weight_offloading.md.")
        num_layers = dit_model.num_layers
        layer_sharding = dit_sharding_spec["params"]["blocks_0"]
        chunk_sharding = [layer_sharding] * chunk_size
        chunk_partition_specs = [to_partition_specs(layer_sharding)] * chunk_size
        nonblock_shardings = {
            k: v for k, v in dit_sharding_spec["params"].items() if not k.startswith("blocks_")}
        nonblock_partition_specs = to_partition_specs({"params": nonblock_shardings})

        def _split_expert(host_params):
            chunk_params_host = [
                [host_params["params"][f"blocks_{i}"] for i in range(c, c + chunk_size)]
                for c in range(0, num_layers, chunk_size)
            ]
            nonblock_params = {
                k: v for k, v in host_params["params"].items() if not k.startswith("blocks_")}
            device_nonblock_params = jax.device_put(
                {"params": nonblock_params}, {"params": nonblock_shardings})
            return chunk_params_host, device_nonblock_params

        high_chunk_params_host, high_nonblock_params = _split_expert(high_dit_params)
        low_chunk_params_host, low_nonblock_params = _split_expert(low_dit_params)
    logging.info("Weights loaded and cast (DiT experts stay on host until needed).")

    # --- Prepare inputs ---
    image = np.array(Image.open(args.image_path).convert("RGB"))
    pixel_h, pixel_w, lat_h, lat_w = compute_latent_grid(
        image.shape[0], image.shape[1], args.max_area, vae_stride=(4, 8, 8),
        patch_size=dit_model.patch_size)
    latent_t = 1 + (args.num_frames - 1) // 4
    logging.info(
        f"Input image {image.shape[1]}x{image.shape[0]} -> "
        f"output {pixel_w}x{pixel_h}, {args.num_frames} frames.")

    logging.info("Encoding conditioning image (VAE)...")
    # `y` gets concatenated directly onto the noisy latent before
    # `patch_embedding` touches it, so it needs to match `noise`'s
    # `dit_dtype`, not the general `--dtype` -- see the identical comment on
    # `noise`'s construction below.
    y = build_i2v_conditioning(
        image, args.num_frames, pixel_h, pixel_w, latent_t, vae_encoder, vae_params, dit_dtype)
    y = jax.device_put(jnp.broadcast_to(y, (dp_size,) + y.shape[1:]), get_batch_sharding(mesh, y.ndim))

    # 16 noise channels + 20 conditioning channels (mask + VAE latent) = the
    # DiT's in_dim=36 -- concatenated here, not passed as a separate
    # argument, unlike Wan2.1's I2V (see this script's header comment).
    noise_shape = (dp_size, latent_t, lat_h, lat_w, 16)
    latents_rng, rng = jax.random.split(rng)
    # `noise` is constructed in `dit_dtype`, not the general `--dtype` --
    # see generate_wan2_1_t2v.py's identical comment on its `latents`.
    noise = jax.random.normal(latents_rng, noise_shape, dtype=dit_dtype)
    noise = jax.device_put(noise, get_batch_sharding(mesh, noise.ndim))

    logging.info(f"Encoding prompt with T5: '{args.prompt}'")
    prompt_embeds = encode_prompts([args.prompt] * dp_size, t5_model, t5_params, tokenizer, dtype)
    prompt_embeds = jax.device_put(prompt_embeds, get_batch_sharding(mesh, prompt_embeds.ndim))
    negative_embeds = encode_prompts(
        [args.negative_prompt] * dp_size, t5_model, t5_params, tokenizer, dtype)
    negative_embeds = jax.device_put(negative_embeds, get_batch_sharding(mesh, negative_embeds.ndim))

    pt, ph, pw = dit_model.patch_size
    head_dim = dit_model.dim // dit_model.num_heads
    freqs = create_rope3d_freqs(
        t=latent_t // pt, h=lat_h // ph, w=lat_w // pw, head_dim=head_dim)
    freqs = jax.device_put(freqs, replicated)

    def _dit_apply(params, latents, t, freqs, context):
        return dit_model.apply(params, latents=latents, t=t, freqs=freqs, context=context)

    if sequence_parallel:
        dit_apply = shard_map(
            _dit_apply, mesh=mesh,
            in_specs=(to_partition_specs(dit_sharding_spec), P('dp', None, None, None, None),
                      P('dp'), (P(), P()), P('dp', None, None)),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

    if args.offload_dit_weights:
        # Per-layer offloading, composed with the two-expert MoE switch and
        # (unlike Wan2.1's identical flag) with `sequence_parallel` too --
        # see the comment above `_split_expert`'s construction for why the
        # latter matters here. `input_dtype`/`grid` are the same every call
        # (fixed resolution/dtype for the whole script run), so they're
        # computed once here as plain Python constants and closed over,
        # rather than threaded through `post_apply` as (`jax.jit`-static)
        # arguments -- simpler than making `shard_map` (below) aware of
        # non-array arguments, which it doesn't support directly.
        offload_input_dtype = dtype
        offload_grid = (2 * dp_size, latent_t // pt, lat_h // ph, lat_w // pw)

        def _pre_process_body(params, latents, t, freqs, context):
            x, ctx, e0, fr, _input_dtype, e, _grid = dit_model.apply(
                params, latents=latents, t=t, freqs=freqs, context=context,
                method=dit_model.pre_process)
            return x, ctx, e0, fr, e

        def _post_process_body(params, x, e):
            return dit_model.apply(
                params, x, e, offload_input_dtype, offload_grid, method=dit_model.post_process)

        def _chunk_forward_body(chunk_params, x, context, e0, freqs):
            for layer_params in chunk_params:
                x = Wan22DiTBlock(
                    dim=dit_model.dim, ffn_dim=dit_model.ffn_dim, num_heads=dit_model.num_heads,
                    qk_norm=dit_model.qk_norm, cross_attn_norm=dit_model.cross_attn_norm,
                    eps=dit_model.eps, compute_dtype=dit_dtype, mesh=mesh,
                    sequence_parallel=sequence_parallel,
                ).apply({"params": layer_params}, x, context, e0, freqs)
            return x

        if sequence_parallel:
            # `pre_process` (inside `_pre_process_body`) does the actual
            # sp-chunking of `x`/`e`/`e0`/`freqs` along the token axis
            # internally (see `WanDiT.pre_process`'s identical comment to
            # Wan2.1's) -- this just needs to run *inside* `shard_map` for
            # `jax.lax.axis_index(sp_axis_name)` to resolve, and needs
            # `out_specs` describing the now-sp-sharded token axis on every
            # output that carries one. `context`/`ctx` never get chunked
            # (already small, fully replicated), matching `dit_apply` above.
            sp_freqs_spec = P(None, 'sp', None, None)
            pre_apply = jax.jit(shard_map(
                _pre_process_body, mesh=mesh,
                in_specs=(nonblock_partition_specs, P('dp', None, None, None, None),
                          P('dp'), (P(), P()), P('dp', None, None)),
                out_specs=(P('dp', 'sp', None), P('dp', None, None), P('dp', 'sp', None, None),
                           (sp_freqs_spec, sp_freqs_spec), P('dp', 'sp', None)),
                check_rep=False,
            ))
            chunk_forward = jax.jit(shard_map(
                _chunk_forward_body, mesh=mesh,
                in_specs=(chunk_partition_specs, P('dp', 'sp', None), P('dp', None, None),
                          P('dp', 'sp', None, None), (sp_freqs_spec, sp_freqs_spec)),
                out_specs=P('dp', 'sp', None),
                check_rep=False,
            ), donate_argnums=(0,))
            post_apply = jax.jit(shard_map(
                _post_process_body, mesh=mesh,
                in_specs=(nonblock_partition_specs, P('dp', 'sp', None), P('dp', 'sp', None)),
                out_specs=P('dp', None, None, None, None),
                check_rep=False,
            ))
        else:
            pre_apply = jax.jit(_pre_process_body)
            chunk_forward = jax.jit(_chunk_forward_body, donate_argnums=(0,))
            post_apply = jax.jit(_post_process_body)

        def single_step_offloaded(current_noise, step_index, prompt_embeds, negative_embeds, y, freqs,
                                   chunk_params_host, nonblock_params, guide_scale):
            b_size = current_noise.shape[0]
            t_val = scheduler.timesteps[step_index]
            t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)
            latents_with_y = jnp.concatenate([current_noise, y], axis=-1)
            latents_2b = jnp.concatenate([latents_with_y, latents_with_y], axis=0)
            t_vec_2b = jnp.concatenate([t_vec, t_vec], axis=0)
            context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)

            x, ctx, e0, fr, e = pre_apply(nonblock_params, latents_2b, t_vec_2b, freqs, context_2b)
            for chunk_host in chunk_params_host:
                chunk_params = jax.device_put(chunk_host, chunk_sharding)
                x = chunk_forward(chunk_params, x, ctx, e0, fr)
            v_2b = post_apply(nonblock_params, x, e)

            v_cond, v_uncond = v_2b[:b_size], v_2b[b_size:]
            velocity = v_uncond + guide_scale * (v_cond - v_uncond)
            return scheduler.step(velocity, step_index, current_noise)

    # --- Euler sampling loop ---
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_noise, step_index, prompt_embeds, negative_embeds, y, freqs, params, guide_scale):
        b_size = current_noise.shape[0]
        t_val = scheduler.timesteps[step_index]
        t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)

        # Channel-concat the (step-invariant) conditioning `y` onto the
        # noisy latent before every DiT call -- matches the reference's
        # `x = cat([x, y], dim=channel)` in `WanModel.forward`.
        latents_with_y = jnp.concatenate([current_noise, y], axis=-1)

        latents_2b = jnp.concatenate([latents_with_y, latents_with_y], axis=0)
        t_vec_2b = jnp.concatenate([t_vec, t_vec], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        v_2b = dit_apply(params, latents_2b, t_vec_2b, freqs, context_2b)
        v_cond, v_uncond = v_2b[:b_size], v_2b[b_size:]
        velocity = v_uncond + guide_scale * (v_cond - v_uncond)
        return scheduler.step(velocity, step_index, current_noise)

    logging.info(
        f"Running sampling for {args.num_steps} steps "
        f"(shift={args.shift}, boundary={args.boundary}, guide_scale={args.guide_scale})...")
    device_params, active_expert = None, None
    for step_index in range(scheduler.num_steps):
        t_val = scheduler.timesteps[step_index]
        expert = "high_noise" if t_val >= boundary_val else "low_noise"
        expert_changed = expert != active_expert
        if expert_changed:
            logging.info(f"  step {step_index}: switched to {expert}_model (t={t_val:.1f})")
            active_expert = expert
        if args.offload_dit_weights:
            chunk_params_host = high_chunk_params_host if expert == "high_noise" else low_chunk_params_host
            nonblock_params = high_nonblock_params if expert == "high_noise" else low_nonblock_params
            noise = single_step_offloaded(
                noise, step_index, prompt_embeds, negative_embeds, y, freqs,
                chunk_params_host, nonblock_params, args.guide_scale)
        else:
            if expert_changed:
                host_params = high_dit_params if expert == "high_noise" else low_dit_params
                device_params = jax.device_put(host_params, dit_sharding_spec)
            noise = single_step(
                noise, step_index, prompt_embeds, negative_embeds, y, freqs, device_params, args.guide_scale)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    x_full = vae_decoder.apply(vae_params, noise.astype(dtype), method=vae_decoder.pre_process)
    decode_chunk_jit = jax.jit(
        lambda params, x_chunk, cache_list: vae_decoder.apply(
            params, x_chunk, cache_list, method=vae_decoder.decode_chunk))

    decoder_cfg = Decoder3d(
        vae_decoder.dim, vae_decoder.z_dim, vae_decoder.dim_mult, vae_decoder.num_res_blocks,
        vae_decoder.attn_scales, vae_decoder.temperal_upsample, vae_decoder.eps)
    cache_list = [None] * _count_causal_convs(decoder_cfg)
    decoded_chunks = []
    for i in range(x_full.shape[1]):
        out_chunk, cache_list = decode_chunk_jit(vae_params, x_full[:, i:i + 1], cache_list)
        decoded_chunks.append(out_chunk)
    decoded_frames = jnp.concatenate(decoded_chunks, axis=1)

    base, ext = os.path.splitext(args.output_path)
    for i in range(dp_size):
        video_frames = np.array(decoded_frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if dp_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end image-to-video generation with Wan2.2 A14B on TPU.")
    parser.add_argument("--high_noise_dit_checkpoint_path", type=str, required=True, help="Path to the high_noise_model DiT .safetensors checkpoint (or .safetensors.index.json manifest, sharded).")
    parser.add_argument("--low_noise_dit_checkpoint_path", type=str, required=True, help="Path to the low_noise_model DiT .safetensors checkpoint (or .safetensors.index.json manifest, sharded).")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the Wan2.1_VAE.pth checkpoint (A14B reuses Wan2.1's causal VAE, not Wan2.2's own).")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to the T5 (umt5-xxl encoder) .pth checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the umt5-xxl HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/google/umt5-xxl'.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the conditioning image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance.")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond).")
    parser.add_argument("--boundary", type=float, default=0.900, help="Fraction of num_train_timesteps (1000) above which the high_noise_model expert is used instead of low_noise_model. Reference default for I2V is 0.900 (0.875 for T2V).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard the (single device-resident) DiT expert's attention heads / FFN channels (weights) across. Must divide num_heads (40 per expert, 64 for T5) and num_devices. Composes independently with --sequence_parallel_size.")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Number of devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of --tensor_parallel_size's weight-sharding. See generate_wan2_1_t2v.py's identical flag for the full reasoning.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for VAE and T5 (and cast target for their loaded checkpoints). Note: TPU's XLA backend does not implement float16 matmuls -- float16 will fail at runtime on TPU.")
    parser.add_argument("--dit_dtype", type=str, default="float32", choices=list(DTYPES.keys()), help="Cast target for both DiT experts' *weights* specifically, independent of --dtype. Defaults to float32: keeping DiT weights (and the block-to-block residual stream) at bfloat16 produces visibly noisy/corrupted output once the video's total token count is large enough (e.g. native 720p), matching the identical Wan2.1 finding -- see docs/models/wan2_2.md#status. Pass --dit_dtype bfloat16 to opt back into the ~2x smaller DiT weight footprint at smaller/safer scales.")
    parser.add_argument("--offload_dit_weights", action="store_true", help="Keep each DiT expert's per-block weights host-resident and offload one --offload_chunk_size-block group's worth into HBM at a time during the sampling loop, instead of one whole expert staying HBM-resident at a time (the existing behavior without this flag). The same idea as generate_wan2_1_t2v.py's identical flag (DeepSpeed ZeRO-Offload / diffusers' enable_sequential_cpu_offload, applied per-layer), composed here with A14B's two-expert MoE switch -- see docs/weight_offloading.md. Unlike Wan2.1's identical flag, this composes with --sequence_parallel_size > 1 (needed to fit native 720P/81 frames here: A14B's per-token modulation makes activation memory, not just weight residency, a real constraint).")
    parser.add_argument("--offload_chunk_size", type=int, default=1, help="Number of consecutive DiT blocks grouped into one offloaded HBM buffer / one jax.jit compile when --offload_dit_weights is set (ignored otherwise). Must divide A14B's num_layers (40). See generate_wan2_1_t2v.py's identical flag.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of sampling steps. The reference's i2v default is 40 (vs 50 for t2v).")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-matching noise-schedule shift. Reference default for A14B I2V is 5.0 (12.0 for T2V).")
    parser.add_argument("--max_area", type=int, default=720 * 1280, help="Target output resolution's pixel area; actual (height, width) are derived from this and the input image's aspect ratio.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the output video.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With --tensor_parallel_size < num_devices (dp_size > 1), each replica's sample is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
