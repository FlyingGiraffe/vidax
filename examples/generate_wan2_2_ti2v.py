# End-to-end text-to-video / image-to-video inference script for Wan2.2
# TI2V-5B on TPU.
#
# TI2V-5B is a single checkpoint that supports both: pass --image_path for
# image-conditioned generation, omit it for plain text-to-video. Unlike
# Wan2.1's i2v (a separate 14B model, CLIP cross-attention, extra `y`
# channel concatenated onto the noisy latent), TI2V-5B's image conditioning
# works by substituting the known conditioning frame's latent back into `x`
# between sampling steps, driven by a per-token timestep of 0 for that
# frame's tokens -- no extra model input at all, see
# `vidax.models.wan.wan2_2.dit`'s module docstring for the architecture side
# of this and the reference's `WanTI2V.i2v` for the sampling-loop mechanics
# this mirrors (`masks_like`'s frame-0 mask, reapplied after every step).
#
# Unlike `generate_wan2_1_t2v.py`, the DiT here uses *sequence* parallelism
# (`WanDiT(sequence_parallel=True)`), not Megatron-style tensor parallelism:
# at TI2V-5B's only supported resolution (704x1280, 121 frames) the
# patch-token sequence is ~27k long, and Wan2.2's per-token modulation
# tensors scale with that directly -- sharding attention heads/FFN channels
# alone doesn't shrink them, so that alone doesn't fit a 4-chip v4 slice's
# HBM even after quartering weight memory. Sequence parallelism instead
# shards the token sequence itself between blocks, which is what actually
# cuts the memory that was overflowing; see
# `vidax.models.wan.wan2_2.dit`'s module docstring for the full mechanism.
# T5 (whose sequence length, 512, was never the bottleneck) keeps using the
# ordinary Megatron tensor-parallel path from `generate_wan2_1_t2v.py`,
# unchanged -- `--tensor_parallel_size` sets both, just with different
# meanings per model.

import argparse
from functools import partial
import logging
import math
import os

import imageio
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from PIL import Image
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from vidax.core.sharding import (
    build_tpu_mesh, shard_wan_params, get_replicated_sharding, get_batch_sharding,
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_2.dit import WanDiT
from vidax.models.wan.wan2_2.vae import (
    WanVAEDecoder, WanVAEEncoder, Decoder3d, Encoder3d,
    _count_causal_convs, _count_causal_convs_encoder, unpatchify, PATCH_SIZE,
)
from vidax.models.wan.common.t5 import T5Encoder, Umt5Tokenizer
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
# Numpy-side counterparts of DTYPES, for casting checkpoints *before*
# `jax.device_put` (see `cast_numpy_tree_to_dtype`'s docstring for why this
# matters here specifically).
NUMPY_DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": ml_dtypes.bfloat16}

# Wan's default negative prompt (Wan2.2-main/wan/configs/shared_config.py,
# `sample_neg_prompt` -- identical string to Wan2.1's), used for classifier-
# free guidance.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """Saves a sequence of frames as an MP4 video."""
    logging.info(f"Saving {frames.shape[0]} frames to {output_path}...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info("Video saved successfully.")


def cast_numpy_tree_to_dtype(tree, np_dtype):
    """Casts every floating-point leaf of a *numpy* (host-RAM) pytree to
    `np_dtype`, before it ever reaches a device.

    Must run before `jax.device_put`, not after: the DiT ships as raw
    float32 (5B params -- 20GB replicated per device under sequence
    parallelism, see this script's header comment for why weights are
    replicated rather than Megatron-sharded here), and casting *after*
    `device_put` needs the float32 copy and the new bf16 copy to coexist on
    that device for the duration of the cast -- 20GB + 10GB transiently,
    which is what was OOM-ing even though the steady-state bf16 weights
    alone fit comfortably. Casting on the host instead means only the
    already-small target-dtype array is ever placed on a device at all.
    """
    def cast_leaf(x):
        if x.dtype.kind == "f" and x.dtype != np_dtype:
            return x.astype(np_dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def resolve_batch_prompts(prompts: list, batch_size: int) -> list:
    """Broadcasts a single prompt to fill the batch, or requires one prompt per slot.

    This lets `batch_size` (== the data-parallel mesh size) double as either
    "N distinct prompts" (one video per prompt) or "N noise samples of one
    prompt" (each data-parallel replica gets independent noise).
    """
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) == batch_size:
        return prompts
    raise ValueError(
        f"Got {len(prompts)} prompts but the data-parallel batch size is "
        f"{batch_size} (num_devices // tensor_parallel_size). Pass exactly 1 "
        f"prompt (broadcast to all replicas) or exactly {batch_size}.")


def encode_prompts(prompts: list, t5_model: T5Encoder, t5_params, tokenizer: Umt5Tokenizer,
                    dtype) -> jnp.ndarray:
    """Tokenizes and T5-encodes one prompt per batch element.

    Matches the reference `T5EncoderModel.__call__`: the encoder runs over
    the full padded sequence with an attention mask, then everything past
    each sample's real (unpadded) length is zeroed out -- WanDiT's cross
    attention has no length masking of its own, so what's actually in those
    trailing positions matters, not just what attends to them.
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)

    context = t5_model.apply(t5_params, ids, mask)
    seq_lens = mask.sum(axis=1)
    positions = jnp.arange(context.shape[1])[None, :]
    keep = positions < seq_lens[:, None]
    return jnp.where(keep[..., None], context, 0.0).astype(dtype)


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> tuple:
    """Picks the (width, height) closest to `w:h`'s aspect ratio, both
    divisible by (dw, dh), with area <= expected_area -- ported directly
    from the reference's `wan/utils/utils.py:best_output_size` (used by
    `WanTI2V.i2v`, unlike `t2v`'s fixed `--size`/`SIZE_CONFIGS`, to derive
    i2v's output resolution from the *input image's* aspect ratio instead).
    Tries rounding width-first and height-first, keeping whichever stays
    closer to the true ratio.
    """
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def build_i2v_conditioning(
    image: np.ndarray, pixel_h: int, pixel_w: int, latent_t: int,
    vae_model: WanVAEEncoder, vae_params, dtype,
) -> jnp.ndarray:
    """Encodes the (already resized+cropped) conditioning frame into a
    (1, 1, lat_h, lat_w, z_dim) latent -- matches the reference's
    `z = self.vae.encode([img])` in `WanTI2V.i2v` (a single real frame, not
    a zero-padded full video the way Wan2.1's i2v conditioning is: TI2V-5B's
    conditioning is applied directly in latent space by the sampling loop,
    not by concatenating a channel onto the model's input, so there's no
    need to build a full-length video here at all).
    """
    img = jnp.asarray(image, dtype=jnp.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
    video = img[None, None].astype(dtype)  # (1, 1, pixel_h, pixel_w, 3)

    # jit-wrapped for the same reason as `WanVAEDecoder.decode_chunk` (see
    # its docstring): calling `encode_chunk`/`post_process` eagerly means
    # every individual op inside the (real-config: 160-channel-wide, 4-stage)
    # `Encoder3d` triggers its own separate XLA compile -- for a single
    # conditioning frame this is only *one* chunk (not ~20 like decode), but
    # that one eager call was still slow enough to look like a hang.
    pre_process_jit = jax.jit(lambda p, x: vae_model.apply(p, x, method=vae_model.pre_process))
    encode_chunk_jit = jax.jit(
        lambda p, x, c: vae_model.apply(p, x, c, method=vae_model.encode_chunk))
    post_process_jit = jax.jit(lambda p, x: vae_model.apply(p, x, method=vae_model.post_process))

    x_full = pre_process_jit(vae_params, video)
    encoder_cfg = Encoder3d(
        vae_model.dim, vae_model.z_dim * 2, vae_model.dim_mult,
        vae_model.num_res_blocks, vae_model.temperal_downsample, vae_model.eps)
    cache_list = [None] * _count_causal_convs_encoder(encoder_cfg)
    out_chunk, cache_list = encode_chunk_jit(vae_params, x_full, cache_list)
    return post_process_jit(vae_params, out_chunk)


def main(args):
    """Main inference function."""
    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size
    assert num_devices % tp_size == 0, (
        f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})")
    dp_size = num_devices // tp_size
    mesh = build_tpu_mesh(data_parallel_size=dp_size, tensor_parallel_size=tp_size)
    rng = jax.random.PRNGKey(args.seed)
    logging.info(f"Using {num_devices} devices: {dp_size}-way data // {tp_size}-way tensor parallel.")

    dtype = DTYPES[args.dtype]
    sequence_parallel = tp_size > 1
    has_image = args.image_path is not None
    num_steps = args.num_steps if args.num_steps is not None else 50

    # --- Initialize models and scheduler ---
    # `sequence_parallel=True` (see this file's header comment and
    # `vidax.models.wan.wan2_2.dit`'s module docstring) requires every
    # `WanDiT.apply(...)` call to run inside `shard_map(..., mesh=mesh)` --
    # done below via `dit_apply`, not by calling `dit_model.apply` directly.
    dit_model = WanDiT(mesh=mesh, sequence_parallel=sequence_parallel, sp_axis_name="tp")
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder() if has_image else None
    t5_model = T5Encoder()
    scheduler = RectifiedFlowScheduler(num_steps=num_steps, shift=args.shift)

    assert dit_model.num_heads % tp_size == 0, (
        f"WanDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}) -- DeepSpeed-Ulysses sequence "
        f"parallelism reshuffles across attention heads, same divisibility "
        f"requirement as Megatron head-sharding; e.g. tp in {{1,2,3,4,6,8,12,24}} "
        f"for the 5B model's 24 heads.")
    assert t5_model.num_heads % tp_size == 0, (
        f"T5Encoder.num_heads ({t5_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")

    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(args.t5_checkpoint_path), "google", "umt5-xxl")
    tokenizer = Umt5Tokenizer(tokenizer_path, seq_len=dit_model.text_len)

    # --- Load weights (DiT, VAE, and T5 ship as separate checkpoints) ---
    # Loaded as numpy (host-RAM) pytrees -- see `load_torch_checkpoint_to_jax`
    # /`convert_pt_tensor_to_jax`'s docstrings for why nothing here touches a
    # device yet.
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(
        args.dit_checkpoint_path, model_type="wan_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(
        args.vae_checkpoint_path, model_type="wan2.2_vae")
    logging.info(f"Loading T5 weights from {args.t5_checkpoint_path}...")
    t5_params = load_torch_checkpoint_to_jax(
        args.t5_checkpoint_path, model_type="wan_t5")

    # Cast to the target dtype *before* `device_put`, still on the host: the
    # DiT ships as raw float32, and every device needs a full replicated
    # copy of it under sequence parallelism (not a `1/tp_size` Megatron
    # shard) -- casting after `device_put` would need the float32 (20GB)
    # and bf16 (10GB) copies to coexist on-device for the cast, which is
    # what was actually OOM-ing (the steady-state bf16 weights alone fit
    # fine). See `cast_numpy_tree_to_dtype`'s docstring.
    np_dtype = NUMPY_DTYPES[args.dtype]
    dit_params = cast_numpy_tree_to_dtype(dit_params, np_dtype)
    vae_params = cast_numpy_tree_to_dtype(vae_params, np_dtype)
    t5_params = cast_numpy_tree_to_dtype(t5_params, np_dtype)

    # T5 is tensor-parallel sharded (attention heads / FFN channels split
    # across the 'tp' axis), same as `generate_wan2_1_t2v.py`. The DiT's
    # weights are instead left fully **replicated**: sequence parallelism
    # shards *activations* along the token axis, not weights, so every
    # device needs its own complete copy of every DiT weight (unlike
    # Megatron sharding, which is precisely what would otherwise let each
    # device hold only `1/tp_size` of them) -- see this file's header
    # comment for why activation, not weight, memory was the actual
    # bottleneck here. The VAE is comparatively small and stays replicated too.
    replicated = get_replicated_sharding(mesh)
    dit_params = jax.device_put(dit_params, replicated)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Prepare inputs ---
    prompts = resolve_batch_prompts(args.prompt, dp_size)
    batch_size = len(prompts)

    # Wan2.2's TI2V-5B causal VAE compresses time by 4x (with a "+1" for the
    # leading frame) and space by 16x: see `vae_stride = (4, 16, 16)` in the
    # reference's `wan_ti2v_5B` config, and
    # vidax.models.wan.wan2_2.vae.WanVAEDecoder's docstring (the 16x comes
    # from an 8x Encoder3d/Decoder3d plus a 2x pixel patchify/unpatchify
    # wrapped around it).
    pt, ph, pw = dit_model.patch_size
    latent_t = 1 + (args.num_frames - 1) // 4
    if has_image:
        # i2v derives output resolution from the *input image's* aspect
        # ratio + --max_area, ignoring --height/--width entirely (matching
        # `WanTI2V.i2v`, which doesn't take a `size` argument at all --
        # unlike `.t2v`, which uses a fixed `--size` preset).
        orig_image = np.array(Image.open(args.image_path).convert("RGB"))
        orig_h, orig_w = orig_image.shape[0], orig_image.shape[1]
        dw, dh = pw * 16, ph * 16
        pixel_w, pixel_h = best_output_size(orig_w, orig_h, dw, dh, args.max_area)

        # Unlike t2v's fixed 704x1280 (chosen so its patch token count
        # divides evenly by every --tensor_parallel_size this script
        # supports), an *arbitrary* input image's aspect ratio gives no such
        # guarantee -- sequence_parallel needs the exact token count to
        # divide evenly (no padding support, see `WanDiT`'s assertion).
        # Growing the width by `dw` up to `tp_size - 1` times is guaranteed
        # to hit a divisible value (`tp_size` consecutive integers cover
        # every residue mod `tp_size`), at the cost of a slightly-off
        # aspect ratio only when this was actually necessary.
        if sequence_parallel:
            t_p = latent_t // pt
            for extra in range(tp_size):
                candidate_w = pixel_w + extra * dw
                w_p = candidate_w // dw
                h_p = pixel_h // dh
                if (t_p * h_p * w_p) % tp_size == 0:
                    if extra:
                        logging.info(
                            f"Growing output width {pixel_w} -> {candidate_w} so the patch "
                            f"token count divides evenly by --tensor_parallel_size ({tp_size}).")
                    pixel_w = candidate_w
                    break

        scale = max(pixel_w / orig_w, pixel_h / orig_h)
        resized = Image.fromarray(orig_image).resize(
            (round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
        x1, y1 = (resized.width - pixel_w) // 2, (resized.height - pixel_h) // 2
        image = np.array(resized.crop((x1, y1, x1 + pixel_w, y1 + pixel_h)))
        logging.info(
            f"Input image {orig_w}x{orig_h} -> output {pixel_w}x{pixel_h}, "
            f"{args.num_frames} frames (--height/--width ignored in i2v mode).")
        latent_h, latent_w = pixel_h // 16, pixel_w // 16
    else:
        pixel_h, pixel_w = args.height, args.width
        latent_h = args.height // 16
        latent_w = args.width // 16
    latents_shape = (batch_size, latent_t, latent_h, latent_w, dit_model.in_dim)

    # Independent noise per batch slot -- when one prompt is broadcast across
    # multiple data-parallel replicas, this gives multiple distinct samples.
    latents_rng, rng = jax.random.split(rng)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=dtype)

    token_mask_flat = None
    frame_mask = None
    if has_image:
        logging.info("Encoding conditioning image...")
        z_cond = build_i2v_conditioning(
            image, pixel_h, pixel_w, latent_t, vae_encoder, vae_params, dtype)
        z_cond = jax.device_put(
            jnp.broadcast_to(z_cond, (batch_size,) + z_cond.shape[1:]),
            get_batch_sharding(mesh, z_cond.ndim))

        # Per-token timestep mask (0 at the conditioning frame's tokens, 1
        # elsewhere) and per-frame latent mask (same, at the un-patchified
        # latent's own T resolution) -- matches the reference's `masks_like`
        # frame-0 mask (`WanTI2V.i2v`), simplified since it's always
        # deterministic at inference (no stochastic `generator`-driven
        # masking, that's a training-only path).
        t_p, h_p, w_p = latent_t // pt, latent_h // ph, latent_w // pw
        token_mask = jnp.ones((t_p, h_p, w_p), dtype=jnp.float32)
        token_mask = token_mask.at[0].set(0.0)
        token_mask_flat = token_mask.reshape(-1)  # (seq_len,)

        frame_mask = jnp.ones((1, latent_t, 1, 1, 1), dtype=dtype)
        frame_mask = frame_mask.at[:, 0].set(0.0)
        frame_mask = jax.device_put(frame_mask, replicated)

        # Apply conditioning to the initial noise too (matches the
        # reference's `latent = (1. - mask2[0]) * z[0] + mask2[0] * latent`
        # immediately before its sampling loop, not just inside it).
        latents = (1.0 - frame_mask) * z_cond + frame_mask * latents

    latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))

    logging.info(f"Encoding {batch_size} prompt(s) with T5: {prompts}")
    prompt_embeds = encode_prompts(prompts, t5_model, t5_params, tokenizer, dtype)
    prompt_embeds = jax.device_put(prompt_embeds, get_batch_sharding(mesh, prompt_embeds.ndim))

    # Classifier-free guidance: the reference *always* runs this (default
    # guide_scale=5.0) -- skipping it isn't just a quality knob turned down,
    # it's the difference between the amplified, prompt-aligned
    # `uncond + guide_scale * (cond - uncond)` signal the model was tuned
    # around and the raw conditional-only prediction, which on its own
    # regresses hard toward a low-contrast, washed-out "average video".
    negative_prompts = resolve_batch_prompts([args.negative_prompt], dp_size)
    negative_embeds = encode_prompts(negative_prompts, t5_model, t5_params, tokenizer, dtype)
    negative_embeds = jax.device_put(negative_embeds, get_batch_sharding(mesh, negative_embeds.ndim))

    # RoPE frequencies for the DiT's patchified (T, H, W) grid.
    head_dim = dit_model.dim // dit_model.num_heads
    freqs = create_rope3d_freqs(
        t=latent_t // pt, h=latent_h // ph, w=latent_w // pw, head_dim=head_dim)
    freqs = jax.device_put(freqs, replicated)

    # `sequence_parallel=True` reshapes/reshuffles activations across the
    # 'tp' mesh axis inside `WanDiT.__call__` itself (chunking the token
    # sequence before the block loop, all-to-all around self-attention,
    # all-gathering back after the head) -- these are collectives, so the
    # call needs to run inside `shard_map`, not a plain `jax.jit` (which
    # only sees ordinary per-device-local ops). `in_specs` mirror the
    # shardings already applied above (`get_batch_sharding` shards the
    # leading batch axis on 'dp' and replicates the rest, `replicated` is
    # fully replicated) -- `shard_map` requires the actual input array
    # shardings to agree with what it's told here.
    def _dit_apply(params, latents, t, freqs, context):
        return dit_model.apply(params, latents=latents, t=t, freqs=freqs, context=context)

    # `t`'s spec differs between modes: t2v passes one scalar timestep per
    # sample (shape (B,), `P('dp')`); i2v passes a *per-token* timestep
    # (shape (B, seq_len), `P('dp', None)` -- replicated across 'tp' like
    # `x`/`freqs`, chunked the same way internally) so the conditioning
    # frame's tokens can be forced to t=0 -- see `token_mask_flat` above and
    # `single_step` below.
    t_spec = P('dp', None) if has_image else P('dp')
    if sequence_parallel:
        dit_apply = shard_map(
            _dit_apply, mesh=mesh,
            in_specs=(P(), P('dp', None, None, None, None), t_spec, (P(), P()), P('dp', None, None)),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

    # --- Euler sampling loop ---
    # `single_step` is jit-compiled once (t_val varies by *value*, not shape
    # or dtype, so this never recompiles) and called from a plain Python
    # loop, rather than jax.jit-ing the whole num_steps-iteration loop as one
    # fused program -- see `generate_wan2_1_t2v.py`'s identical comment here
    # for why. `donate_argnums=(0,)` additionally lets XLA overwrite the
    # previous step's latents buffer in place instead of allocating a fresh one.
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_latents, step_index, prompt_embeds, negative_embeds, freqs, params, guide_scale):
        b_size = current_latents.shape[0]
        # `timesteps` are on the ~[0, num_train_timesteps] scale the model
        # was trained on (see RectifiedFlowScheduler), not the raw [0, 1]
        # flow-matching sigma.
        t_val = scheduler.timesteps[step_index]
        if has_image:
            # Per-token: 0 at the conditioning frame's tokens (already
            # correct, no noise to remove there), `t_val` everywhere else --
            # matches the reference's `(mask2[0][0][:, ::2, ::2] *
            # timestep).flatten()`.
            t_vec = jnp.broadcast_to(token_mask_flat[None, :] * t_val, (b_size, token_mask_flat.shape[0]))
        else:
            # Uniform scalar-per-sample `t`; `WanDiT` broadcasts this to
            # every patch token internally (see its module docstring) --
            # this is the model's uniform-timestep degenerate case, not a
            # shortcut.
            t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)

        # Classifier-free guidance: two forward passes (conditional and
        # unconditional/negative-prompt), amplifying their difference. This
        # is not optional in the reference pipeline -- see the comment above
        # `negative_prompts` in main().
        v_cond = dit_apply(params, current_latents, t_vec, freqs, prompt_embeds)
        v_uncond = dit_apply(params, current_latents, t_vec, freqs, negative_embeds)
        velocity = v_uncond + guide_scale * (v_cond - v_uncond)
        current_latents = scheduler.step(velocity, step_index, current_latents)

        if has_image:
            # Re-clamp the conditioning frame's latent back to the known
            # encoding after every step -- the model's own prediction there
            # is never used, matching the reference's re-application of
            # `(1. - mask2[0]) * z[0] + mask2[0] * latent` after each
            # `sample_scheduler.step`.
            current_latents = (1.0 - frame_mask) * z_cond + frame_mask * current_latents
        return current_latents

    logging.info(
        f"Running sampling for {num_steps} steps "
        f"(shift={args.shift}, guide_scale={args.guide_scale})...")
    for step_index in range(scheduler.num_steps):
        latents = single_step(
            latents, step_index, prompt_embeds, negative_embeds, freqs, dit_params, args.guide_scale)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    # Unlike `generate_wan2_1_t2v.py`'s VAE decode call (a single, simple
    # `vae_decoder.apply(vae_params, latents)`), this uses `WanVAEDecoder`'s
    # `decode_chunk` method explicitly, called from a plain Python loop with
    # `jax.jit` wrapped around *each per-frame call*, not the whole loop:
    # - Wrapping the whole loop (or not jit-ing at all, i.e. calling
    #   `vae_decoder.apply(vae_params, latents)` directly) unrolls or eagerly
    #   dispatches every op inside all ~31 latent frames' worth of a very
    #   deep, 1024-channel-wide decoder -- at TI2V-5B's full resolution,
    #   both blow up (unrolling needs every chunk's activations to coexist
    #   in one HLO program's memory; eager dispatch means *every individual
    #   op*, not just each per-frame step, triggers its own slow XLA
    #   compile, which is what made a first attempt at this appear to hang).
    # - Jit-ing just `decode_chunk` compiles the whole per-frame computation
    #   as one fused program, once per distinct cache-state shape (in
    #   practice: the first frame, the second, then one more reused for
    #   every remaining frame) -- see `WanVAEDecoder.decode_chunk`'s
    #   docstring for the full reasoning.
    pre_process_jit = jax.jit(lambda p, z: vae_decoder.apply(p, z, method=vae_decoder.pre_process))
    x_full = pre_process_jit(vae_params, latents.astype(dtype))
    decode_chunk_jit = jax.jit(
        lambda params, x_chunk, cache_list, first_chunk: vae_decoder.apply(
            params, x_chunk, cache_list, first_chunk, method=vae_decoder.decode_chunk),
        static_argnums=(3,))

    decoder_cfg = Decoder3d(
        vae_decoder.dim, vae_decoder.z_dim, vae_decoder.dim_mult,
        vae_decoder.num_res_blocks, vae_decoder.temperal_upsample, vae_decoder.eps)
    cache_list = [None] * _count_causal_convs(decoder_cfg)
    decoded_chunks = []
    for i in range(x_full.shape[1]):
        out_chunk, cache_list = decode_chunk_jit(vae_params, x_full[:, i:i + 1], cache_list, i == 0)
        # Move each chunk to the host immediately, rather than keeping all
        # ~31 of them device-resident until a single big `jnp.concatenate`
        # at the end -- with the 5B DiT/T5/VAE params (plus this run's i2v-
        # specific extras: the conditioning image's encoded latent, masks)
        # already occupying most of a device's HBM, that final concatenate
        # needing its own large contiguous allocation was enough to OOM on
        # its own, even after sampling and every individual decode chunk
        # had already succeeded.
        decoded_chunks.append(np.asarray(out_chunk))
    decoded_frames = unpatchify(np.concatenate(decoded_chunks, axis=1), PATCH_SIZE)

    # One output video per batch element.
    base, ext = os.path.splitext(args.output_path)
    for i in range(batch_size):
        video_frames = np.array(decoded_frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end text-to-video / image-to-video generation with Wan2.2 TI2V-5B on TPU.")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the DiT checkpoint: a single .safetensors file, or (the 5B/14B models ship sharded) the accompanying diffusion_pytorch_model.safetensors.index.json manifest.")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the Wan2.2_VAE.pth checkpoint.")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to the T5 (umt5-xxl encoder) .pth checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the umt5-xxl HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/google/umt5-xxl'.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a conditioning image, for image-to-video generation. Omit for plain text-to-video. When given, output resolution is derived from the image's aspect ratio + --max_area instead of --height/--width (which are then ignored) -- matching the reference's WanTI2V.i2v, which has no fixed --size the way .t2v does.")
    parser.add_argument("--max_area", type=int, default=704 * 1280, help="i2v only: target output resolution's pixel area; actual (height, width) are derived from this and the input image's aspect ratio (see best_output_size).")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance. Defaults to the reference's `sample_neg_prompt`.")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond). The reference's default is 5.0; skipping CFG (there is no flag to do so here, matching the reference always running it) produces washed-out, low-contrast output.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to parallelize each model across -- the DiT via *sequence* parallelism (shards the ~27k-token patch sequence itself between blocks; see this script's header comment) and T5 via ordinary Megatron tensor parallelism (shards attention heads / FFN channels). Must divide num_heads (24 for the 5B DiT, 64 for the T5 encoder) and num_devices, and the DiT's patch token count must be evenly divisible by it too (true by construction at the default 704x1280x121 resolution on 1/2/4/5/8-way splits). Increase this if you hit HBM OOM.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the DiT, VAE, and T5 (and cast target for their loaded checkpoints). The reference uses bfloat16 for the DiT/T5 and float32 for the VAE; vidax uses one unified dtype for simplicity. Note: TPU's XLA backend does not implement float16 matmuls (a hardware/compiler limitation, not a vidax one) -- float16 will fail at runtime on TPU.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of sampling steps for the scheduler. Defaults to the reference's own per-mode default: 50 for text-to-video, 40 for image-to-video (WanTI2V.generate / .i2v).")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-matching noise-schedule shift (see RectifiedFlowScheduler). The reference's default for TI2V-5B is 5.0.")
    parser.add_argument("--height", type=int, default=704, help="Output video height. Must be a multiple of 16 (the VAE's spatial stride). Ignored if --image_path is given (see --image_path).")
    parser.add_argument("--width", type=int, default=1280, help="Output video width. Must be a multiple of 16 (the VAE's spatial stride). Ignored if --image_path is given (see --image_path).")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames in the output video. The reference's default for TI2V-5B is 121 (vs. 81 for Wan2.1).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the saved video. The reference's `sample_fps` for TI2V-5B is 24 (vs. 16 for Wan2.1).")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With multiple prompts, each video is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
