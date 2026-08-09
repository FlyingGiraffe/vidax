# End-to-end text-to-video inference script for Wan2.2 A14B on TPU.
#
# A14B is a Mixture-of-Experts model: two separately-checkpointed 14B DiTs
# (high_noise_model / low_noise_model, same `vidax.models.wan.wan2_2.dit.
# WanDiT` architecture and config -- see `configs.T2V_A14B_CONFIG`), switched
# per sampling step by comparing the step's timestep against `--boundary *
# num_train_timesteps` (`high_noise_model` above the boundary, i.e. the
# noisier early steps; `low_noise_model` below it) -- matches the reference's
# `_prepare_model_for_timestep` in Wan2.2-main/wan/text2video.py. This is a
# plain Python-level choice of which params pytree to feed the same jitted
# `single_step` on a given iteration, not a traced/data-dependent branch:
# `RectifiedFlowScheduler.timesteps` is precomputed on the host in the same
# units as `boundary` (both `sigma * num_train_timesteps`), so which expert
# a given step needs is known before the loop even starts.
#
# Unlike Wan2.2's TI2V-5B, A14B uses Wan2.1's causal VAE (`Wan2.1_VAE.pth`,
# vae_stride=(4,8,8)) and Wan2.1-style Megatron tensor parallelism
# (`--tensor_parallel_size`) by default, not sequence parallelism -- same
# architecture and scale as `generate_wan2_1_t2v.py`'s 14B path (which was
# verified to fit fine under plain TP). `--sequence_parallel_size` composes
# independently (see `docs/hardware_and_sharding.md`'s "Combining both"
# section) and is worth trying together with `--tensor_parallel_size` at
# resolutions where even one device-resident 14B expert alone doesn't fit.

import argparse
from functools import partial
import logging
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from vidax.core.sharding import (
    build_tpu_mesh, shard_wan_params, get_replicated_sharding, get_batch_sharding,
    to_partition_specs, configure_jax_cache,
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_2.configs import T2V_A14B_CONFIG
from vidax.models.wan.wan2_2.dit import WanDiT
from vidax.models.wan.wan2_1.vae import WanVAEDecoder, Decoder3d, _count_causal_convs
from vidax.models.wan.common.t5 import T5Encoder, Umt5Tokenizer
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}

# Wan2.2's default negative prompt (Wan2.2-main/wan/configs/shared_config.py,
# `sample_neg_prompt` -- identical string to Wan2.1's), used for classifier-
# free guidance.
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
    """Casts every floating-point leaf of a pytree to `dtype` (int/bool, and
    leaves already in `dtype`, are left untouched -- avoids a transient
    duplicate allocation for a no-op cast).
    """
    def cast_leaf(x):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def resolve_batch_prompts(prompts: list, batch_size: int) -> list:
    """Broadcasts a single prompt to fill the batch, or requires one prompt per slot."""
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
    """Tokenizes and T5-encodes one prompt per batch element (see
    generate_wan2_1_t2v.py's identical copy of this function for why the
    zero-fill past each sample's real length matters).
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)

    context = t5_model.apply(t5_params, ids, mask)
    seq_lens = mask.sum(axis=1)
    positions = jnp.arange(context.shape[1])[None, :]
    keep = positions < seq_lens[:, None]
    return jnp.where(keep[..., None], context, 0.0).astype(dtype)


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
    sequence_parallel = sp_size > 1

    # --- Initialize models and scheduler ---
    dit_model = WanDiT(mesh=mesh, sequence_parallel=sequence_parallel, **T2V_A14B_CONFIG)
    vae_model = WanVAEDecoder()
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

    # See generate_wan2_1_t2v.py's identical comment for why casting happens
    # before device_put. DiT weights are always Megatron-sharded (see that
    # same script's comment -- weight-sharding and token-sharding are
    # independent mesh axes now, so `shard_wan_params` covers every
    # `--sequence_parallel_size` value on its own).
    #
    # Unlike every other script, the two DiT experts are *not* both put on
    # device here: at TP=4 (this repo's target 4-chip v4 slice), a single
    # 14B expert's sharded weights (~7.5GB/device) leave enough HBM headroom
    # for its own forward pass, but two residents at once (~15GB/device)
    # don't -- the forward pass itself needs on the order of 17GB/device of
    # working memory (FFN/attention intermediates across 40 layers), which
    # doesn't fit in the ~15.75GB left over. Since `scheduler.timesteps` is
    # monotonically decreasing, the high/low expert boundary is crossed
    # *once*, so only one expert is ever kept device-resident at a time
    # (swapped in the sampling loop below) -- this matches the memory
    # footprint already verified to work for a single 14B model under TP=4
    # (see generate_wan2_1_t2v.py's 14B path), just paying one host<->device
    # transfer at the single boundary crossing instead of the (infeasible,
    # on 4 chips) cost of both experts resident together.
    # `--sequence_parallel_size` composes with this unchanged: it just makes
    # the "single resident expert" itself both weight- and token-sharded
    # instead of weight-sharded alone (see `docs/hardware_and_sharding.md`'s
    # "Combining both" section) -- worth trying if even one expert alone
    # doesn't fit at the resolution you want.
    replicated = get_replicated_sharding(mesh)
    high_dit_params = cast_to_dtype(high_dit_params, dtype)
    low_dit_params = cast_to_dtype(low_dit_params, dtype)
    vae_params = cast_to_dtype(vae_params, dtype)
    t5_params = cast_to_dtype(t5_params, dtype)

    dit_sharding_spec = shard_wan_params(high_dit_params, mesh)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)
    logging.info("Weights loaded and cast (DiT experts stay on host until needed).")

    # --- Prepare inputs ---
    prompts = resolve_batch_prompts(args.prompt, dp_size)
    batch_size = len(prompts)

    # Same causal VAE as Wan2.1 (4x temporal / 8x spatial compression).
    latent_t = 1 + (args.num_frames - 1) // 4
    latent_h = args.height // 8
    latent_w = args.width // 8
    latents_shape = (batch_size, latent_t, latent_h, latent_w, dit_model.in_dim)

    latents_rng, rng = jax.random.split(rng)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=dtype)
    latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))

    logging.info(f"Encoding {batch_size} prompt(s) with T5: {prompts}")
    prompt_embeds = encode_prompts(prompts, t5_model, t5_params, tokenizer, dtype)
    prompt_embeds = jax.device_put(prompt_embeds, get_batch_sharding(mesh, prompt_embeds.ndim))

    negative_prompts = resolve_batch_prompts([args.negative_prompt], dp_size)
    negative_embeds = encode_prompts(negative_prompts, t5_model, t5_params, tokenizer, dtype)
    negative_embeds = jax.device_put(negative_embeds, get_batch_sharding(mesh, negative_embeds.ndim))

    pt, ph, pw = dit_model.patch_size
    head_dim = dit_model.dim // dit_model.num_heads
    freqs = create_rope3d_freqs(
        t=latent_t // pt, h=latent_h // ph, w=latent_w // pw, head_dim=head_dim)
    freqs = jax.device_put(freqs, replicated)

    # See generate_wan2_1_t2v.py's identical comment on why `sequence_parallel`
    # needs `shard_map`, not just `jax.jit`.
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

    # --- Euler sampling loop (see generate_wan2_1_t2v.py for why this isn't
    # one big jax.jit) ---
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_latents, step_index, prompt_embeds, negative_embeds, freqs, params, guide_scale):
        b_size = current_latents.shape[0]
        t_val = scheduler.timesteps[step_index]
        t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)

        latents_2b = jnp.concatenate([current_latents, current_latents], axis=0)
        t_vec_2b = jnp.concatenate([t_vec, t_vec], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        v_2b = dit_apply(params, latents_2b, t_vec_2b, freqs, context_2b)
        v_cond, v_uncond = v_2b[:b_size], v_2b[b_size:]
        velocity = v_uncond + guide_scale * (v_cond - v_uncond)
        return scheduler.step(velocity, step_index, current_latents)

    logging.info(
        f"Running sampling for {args.num_steps} steps "
        f"(shift={args.shift}, boundary={args.boundary}, guide_scale={args.guide_scale})...")
    device_params, active_expert = None, None
    for step_index in range(scheduler.num_steps):
        t_val = scheduler.timesteps[step_index]
        expert = "high_noise" if t_val >= boundary_val else "low_noise"
        if expert != active_expert:
            # Only one expert is ever device-resident at a time -- see the
            # comment above `dit_sharding_spec` in main() for why. Dropping
            # the reference lets JAX free that expert's device buffers
            # before the new one is placed.
            device_params = None
            host_params = high_dit_params if expert == "high_noise" else low_dit_params
            device_params = jax.device_put(host_params, dit_sharding_spec)
            active_expert = expert
            logging.info(f"  step {step_index}: switched to {expert}_model (t={t_val:.1f})")
        latents = single_step(
            latents, step_index, prompt_embeds, negative_embeds, freqs, device_params, args.guide_scale)

    # --- Decode latents to video frames (see generate_wan2_1_t2v.py for why
    # this uses a per-chunk jit loop rather than one big jax.jit call) ---
    logging.info("Decoding final latents into video frames...")
    x_full = vae_model.apply(vae_params, latents.astype(dtype), method=vae_model.pre_process)
    decode_chunk_jit = jax.jit(
        lambda params, x_chunk, cache_list: vae_model.apply(
            params, x_chunk, cache_list, method=vae_model.decode_chunk))

    decoder_cfg = Decoder3d(
        vae_model.dim, vae_model.z_dim, vae_model.dim_mult, vae_model.num_res_blocks,
        vae_model.attn_scales, vae_model.temperal_upsample, vae_model.eps)
    cache_list = [None] * _count_causal_convs(decoder_cfg)
    decoded_chunks = []
    for i in range(x_full.shape[1]):
        out_chunk, cache_list = decode_chunk_jit(vae_params, x_full[:, i:i + 1], cache_list)
        decoded_chunks.append(out_chunk)
    decoded_frames = jnp.concatenate(decoded_chunks, axis=1)

    base, ext = os.path.splitext(args.output_path)
    for i in range(batch_size):
        video_frames = np.array(decoded_frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end text-to-video generation with Wan2.2 A14B on TPU.")
    parser.add_argument("--high_noise_dit_checkpoint_path", type=str, required=True, help="Path to the high_noise_model DiT .safetensors checkpoint (or .safetensors.index.json manifest, sharded).")
    parser.add_argument("--low_noise_dit_checkpoint_path", type=str, required=True, help="Path to the low_noise_model DiT .safetensors checkpoint (or .safetensors.index.json manifest, sharded).")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the Wan2.1_VAE.pth checkpoint (A14B reuses Wan2.1's causal VAE, not Wan2.2's own).")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to the T5 (umt5-xxl encoder) .pth checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the umt5-xxl HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/google/umt5-xxl'.")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance.")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond).")
    parser.add_argument("--boundary", type=float, default=0.875, help="Fraction of num_train_timesteps (1000) above which the high_noise_model expert is used instead of low_noise_model. Reference default for T2V is 0.875 (0.900 for I2V).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard the (single device-resident) DiT expert's attention heads / FFN channels (weights) across. Must divide num_heads (40 per expert, 64 for T5) and num_devices. Composes independently with --sequence_parallel_size.")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Number of devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of --tensor_parallel_size's weight-sharding. See generate_wan2_1_t2v.py's identical flag for the full reasoning; worth trying together with --tensor_parallel_size if even one A14B expert alone doesn't fit HBM at the resolution you want (see this script's header comment). Also requires the DiT's patch token count to be evenly divisible by this value.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for both DiT experts, VAE, and T5. Note: TPU's XLA backend does not implement float16 matmuls -- float16 will fail at runtime on TPU.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps for the scheduler.")
    parser.add_argument("--shift", type=float, default=12.0, help="Flow-matching noise-schedule shift. Reference default for A14B T2V is 12.0 (5.0 for I2V).")
    parser.add_argument("--height", type=int, default=720, help="Output video height.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the output video.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With multiple prompts, each video is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
