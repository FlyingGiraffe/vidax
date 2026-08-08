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
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_2.configs import I2V_A14B_CONFIG
from vidax.models.wan.wan2_2.dit import WanDiT
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
    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size
    assert num_devices % tp_size == 0, (
        f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})")
    dp_size = num_devices // tp_size
    mesh = build_tpu_mesh(data_parallel_size=dp_size, tensor_parallel_size=tp_size)
    rng = jax.random.PRNGKey(args.seed)
    logging.info(f"Using {num_devices} devices: {dp_size}-way data // {tp_size}-way tensor parallel.")

    dtype = DTYPES[args.dtype]
    sequence_parallel = args.sequence_parallel

    # --- Initialize models and scheduler ---
    dit_model = WanDiT(
        mesh=mesh, sequence_parallel=sequence_parallel, sp_axis_name="tp", **I2V_A14B_CONFIG)
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder()
    t5_model = T5Encoder()
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=args.shift)
    boundary_val = args.boundary * scheduler.num_train_timesteps

    assert dit_model.num_heads % tp_size == 0, (
        f"WanDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}); e.g. tp in {{1,2,4,5,8,10,20,40}}.")
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
    replicated = get_replicated_sharding(mesh)
    high_dit_params = cast_to_dtype(high_dit_params, dtype)
    low_dit_params = cast_to_dtype(low_dit_params, dtype)
    vae_params = cast_to_dtype(vae_params, dtype)
    t5_params = cast_to_dtype(t5_params, dtype)

    dit_sharding_spec = replicated if sequence_parallel else shard_wan_params(high_dit_params, mesh)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)
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
    y = build_i2v_conditioning(
        image, args.num_frames, pixel_h, pixel_w, latent_t, vae_encoder, vae_params, dtype)
    y = jax.device_put(jnp.broadcast_to(y, (dp_size,) + y.shape[1:]), get_batch_sharding(mesh, y.ndim))

    # 16 noise channels + 20 conditioning channels (mask + VAE latent) = the
    # DiT's in_dim=36 -- concatenated here, not passed as a separate
    # argument, unlike Wan2.1's I2V (see this script's header comment).
    noise_shape = (dp_size, latent_t, lat_h, lat_w, 16)
    latents_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(latents_rng, noise_shape, dtype=dtype)
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
            in_specs=(P(), P('dp', None, None, None, None), P('dp'), (P(), P()), P('dp', None, None)),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

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
        if expert != active_expert:
            device_params = None
            host_params = high_dit_params if expert == "high_noise" else low_dit_params
            device_params = jax.device_put(host_params, dit_sharding_spec)
            active_expert = expert
            logging.info(f"  step {step_index}: switched to {expert}_model (t={t_val:.1f})")
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
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to shard each model's attention heads / FFN channels across. Must divide num_heads (40 for each DiT expert, 64 for T5) and num_devices.")
    parser.add_argument("--sequence_parallel", action="store_true", help="Shard the DiT's token sequence itself across --tensor_parallel_size devices (DeepSpeed-Ulysses) instead of Megatron-style tensor parallelism. See generate_wan2_1_t2v.py's identical flag for the full reasoning.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for both DiT experts, VAE, and T5. Note: TPU's XLA backend does not implement float16 matmuls -- float16 will fail at runtime on TPU.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of sampling steps. The reference's i2v default is 40 (vs 50 for t2v).")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-matching noise-schedule shift. Reference default for A14B I2V is 5.0 (12.0 for T2V).")
    parser.add_argument("--max_area", type=int, default=720 * 1280, help="Target output resolution's pixel area; actual (height, width) are derived from this and the input image's aspect ratio.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the output video.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With --tensor_parallel_size < num_devices (dp_size > 1), each replica's sample is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
