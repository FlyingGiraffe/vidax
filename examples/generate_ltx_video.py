# End-to-end text-to-video / image-to-video inference script for LTX-Video
# 0.9.8 on TPU.
#
# Supports Megatron-style 1D tensor parallelism (--tensor_parallel_size,
# see vidax.core.sharding) for the DiT and T5 encoder -- needed for the 13B
# variants, whose bf16 weights alone (~26GB) don't fit replicated on a
# single TPU v4 chip's ~32GB HBM alongside the T5 encoder and VAE. No
# --sequence_parallel_size yet (LTXDiT has no sequence-parallel wiring --
# see docs/models/ltx_video.md's Status section); TP-only composes with a
# plain jax.jit + GSPMD auto-partitioning here, no shard_map needed (unlike
# Wan's scripts, which need shard_map once sequence_parallel is involved).
# Single-scale generation only
# (the reference's multi-scale two-pass pipeline and its separate
# LatentUpsampler model are not implemented), plain constant-scale
# classifier-free guidance (no STG, no cfg_star_rescale, no per-step
# guidance_scale schedule -- the reference's own configs use a schedule
# that varies guidance_scale/stg_scale per step; this script uses one
# constant value for the whole run instead).
#
# T2V and I2V share this one script/pipeline, exactly as the reference
# does (`Transformer3DModel.forward`'s signature never distinguishes them --
# only whether a conditioning image was VAE-encoded and lerp'd into the
# initial latent differs): pass --image_path for I2V, omit it for T2V.

import argparse
from functools import partial
import logging
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vidax.core.sharding import build_tpu_mesh, configure_jax_cache, get_replicated_sharding, shard_wan_params
from vidax.models.ltx_video.configs import (
    load_ltx_checkpoint_metadata, load_ltx_vae_per_channel_stats, vae_scale_factors,
)
from vidax.models.ltx_video.dit import LTXDiT
from vidax.models.ltx_video.patchifier import get_latent_coords, latent_to_pixel_coords, patchify, unpatchify
from vidax.models.ltx_video.t5 import PixArtT5Tokenizer, T5Encoder
from vidax.models.ltx_video.vae import LTXVAE
from vidax.schedulers.ltx_rectified_flow import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}

# The reference's own default (`ltx_video/inference.py`'s `InferenceConfig`).
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
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


def load_conditioning_image(image_path: str, height: int, width: int) -> np.ndarray:
    """Loads and resizes a conditioning image to (1, 1, H, W, 3) in [-1, 1],
    matching the reference's own image-loading convention.
    """
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    return arr[None, None]  # (1, 1, H, W, 3)


def encode_prompts(prompts, t5_model: T5Encoder, t5_params, tokenizer: PixArtT5Tokenizer, dtype):
    """Tokenizes and T5-encodes one prompt per batch element. Unlike Wan's
    UMT5 (whose DiT cross-attention has no length masking of its own, so
    padding must be explicitly zeroed), LTX's transformer takes the
    attention mask directly as `encoder_attention_mask` (see
    `vidax.models.ltx_video.dit.LTXDiT.pre_process`) -- so the raw T5
    output and mask are both returned as-is here.
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)
    context = t5_model.apply(t5_params, ids, mask).astype(dtype)
    return context, mask.astype(jnp.float32)


def main(args):
    configure_jax_cache()
    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size
    assert num_devices % tp_size == 0, (
        f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})")
    dp_size = num_devices // tp_size
    mesh = build_tpu_mesh(data_parallel_size=dp_size, tensor_parallel_size=tp_size, sequence_parallel_size=1)
    rng = jax.random.PRNGKey(args.seed)

    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]

    # --- Build models from the checkpoint's own embedded architecture config ---
    metadata = load_ltx_checkpoint_metadata(args.checkpoint_path)
    transformer_cfg, vae_cfg = metadata["transformer"], metadata["vae"]
    causal_fix = transformer_cfg.get("causal_temporal_positioning", False)

    dit_model = LTXDiT(
        num_attention_heads=transformer_cfg["num_attention_heads"],
        attention_head_dim=transformer_cfg["attention_head_dim"],
        in_channels=transformer_cfg["in_channels"],
        out_channels=transformer_cfg["out_channels"],
        num_layers=transformer_cfg["num_layers"],
        cross_attention_dim=transformer_cfg["cross_attention_dim"],
        caption_channels=transformer_cfg["caption_channels"],
        positional_embedding_theta=transformer_cfg["positional_embedding_theta"],
        positional_embedding_max_pos=tuple(transformer_cfg["positional_embedding_max_pos"]),
        timestep_scale_multiplier=transformer_cfg.get("timestep_scale_multiplier", 1000),
        compute_dtype=dit_dtype,
    )
    encoder_blocks = tuple((name, dict(params)) for name, params in vae_cfg["encoder_blocks"])
    decoder_blocks = tuple((name, dict(params)) for name, params in vae_cfg["decoder_blocks"])
    vae_model = LTXVAE(
        latent_channels=vae_cfg["latent_channels"], encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks, patch_size=vae_cfg["patch_size"],
        base_channels=vae_cfg.get("base_channels", 128))
    temporal_scale, spatial_scale = vae_scale_factors(vae_cfg)

    t5_model = T5Encoder()
    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(os.path.dirname(args.t5_checkpoint_path)), "tokenizer")
    tokenizer = PixArtT5Tokenizer(tokenizer_path, seq_len=args.text_max_tokens)

    assert dit_model.num_attention_heads % tp_size == 0, (
        f"LTXDiT.num_attention_heads ({dit_model.num_attention_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    assert t5_model.num_heads % tp_size == 0, (
        f"T5Encoder.num_heads ({t5_model.num_heads}) must be divisible by --tensor_parallel_size ({tp_size}).")

    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, sampler=args.sampler, shift=args.shift)

    # --- Load weights (DiT+VAE share one checkpoint file; T5 is separate) ---
    logging.info(f"Loading DiT+VAE weights from {args.checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.checkpoint_path, model_type="ltx_video_dit")
    vae_params = load_torch_checkpoint_to_jax(args.checkpoint_path, model_type="ltx_video_vae")
    logging.info(f"Loading T5 weights from {args.t5_checkpoint_path}...")
    t5_params = load_torch_checkpoint_to_jax(args.t5_checkpoint_path, model_type="ltx_video_t5")
    vae_mean, vae_std = load_ltx_vae_per_channel_stats(args.checkpoint_path)
    vae_mean = jnp.asarray(vae_mean, dtype=jnp.float32)
    vae_std = jnp.asarray(vae_std, dtype=jnp.float32)

    # DiT and T5 are Megatron-sharded across the 'tp' axis (attention heads
    # / FFN channels split per chip -- see vidax.core.sharding); the VAE is
    # comparatively small and stays fully replicated regardless, same as
    # every other model's script.
    replicated = get_replicated_sharding(mesh)
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    t5_params = cast_to_dtype(t5_params, dtype)
    dit_params = jax.device_put(dit_params, shard_wan_params(dit_params, mesh))
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Prepare inputs ---
    batch_size = len(args.prompt)
    latent_f = 1 + (args.num_frames - 1) // temporal_scale
    latent_h = args.height // spatial_scale
    latent_w = args.width // spatial_scale

    latents_rng, rng = jax.random.split(rng)
    latents = jax.random.normal(
        latents_rng, (batch_size, latent_f, latent_h, latent_w, dit_model.in_channels), dtype=jnp.float32
    ).astype(dit_dtype)

    logging.info(f"Encoding {batch_size} prompt(s) with T5: {args.prompt}")
    prompt_embeds, prompt_mask = encode_prompts(args.prompt, t5_model, t5_params, tokenizer, dtype)
    negative_embeds, negative_mask = encode_prompts(
        [args.negative_prompt] * batch_size, t5_model, t5_params, tokenizer, dtype)

    latent_coords = get_latent_coords(latent_f, latent_h, latent_w, batch_size)
    pixel_coords = latent_to_pixel_coords(latent_coords, temporal_scale, spatial_scale, causal_fix=causal_fix)

    # --- I2V conditioning: frame-0 lerp + per-token mask (see file docstring) ---
    image_cond_mask = None
    if args.image_path is not None:
        logging.info(f"I2V: encoding conditioning image {args.image_path}")
        image = load_conditioning_image(args.image_path, args.height, args.width)
        image = np.broadcast_to(image, (batch_size,) + image.shape[1:])
        img_noise_rng, rng = jax.random.split(rng)
        img_noise = jax.random.normal(
            img_noise_rng, (batch_size, 1, latent_h, latent_w, vae_cfg["latent_channels"]), dtype=jnp.float32)
        cond_latent = vae_model.apply(vae_params, jnp.asarray(image, dtype=dtype), img_noise, method=vae_model.encode)
        cond_latent_norm = (cond_latent.astype(jnp.float32) - vae_mean) / vae_std

        strength = args.conditioning_strength
        first_frame = latents[:, :1].astype(jnp.float32) * (1 - strength) + cond_latent_norm * strength
        latents = latents.at[:, :1].set(first_frame.astype(dit_dtype))

        cond_mask_grid = jnp.zeros((batch_size, latent_f, latent_h, latent_w, 1), dtype=jnp.float32)
        cond_mask_grid = cond_mask_grid.at[:, :1].set(strength)
        image_cond_mask = patchify(cond_mask_grid)[..., 0]  # (B, N)

    tokens = patchify(latents)

    dit_apply = jax.jit(lambda params, tokens, coords, timestep, context, mask: dit_model.apply(
        params, tokens, coords, timestep, context, mask))

    # `step_index` (not the timestep value itself) is the jitted argument --
    # `scheduler.sigmas[step_index]` is a traced dynamic-slice, so this
    # compiles once and is reused for every step (mirrors
    # `examples/generate_wan2_1_t2v.py`'s identical reasoning).
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_tokens, step_index, prompt_embeds, prompt_mask, negative_embeds, negative_mask,
                     pixel_coords, image_cond_mask, params, guidance_scale):
        b = current_tokens.shape[0]
        t_val = scheduler.sigmas[step_index]
        if image_cond_mask is None:
            timestep = jnp.full((b,), t_val, dtype=jnp.float32)
        else:
            # I2V: clamp each token's effective timestep so conditioning
            # tokens (mask close to 1) sit near-frozen at their VAE-encoded
            # value instead of being denoised from scratch -- matches the
            # reference's `torch.min(current_timestep, 1.0 - conditioning_mask)`.
            timestep = jnp.minimum(jnp.full_like(image_cond_mask, t_val), 1.0 - image_cond_mask)

        tokens_2b = jnp.concatenate([current_tokens, current_tokens], axis=0)
        timestep_2b = jnp.concatenate([timestep, timestep], axis=0)
        coords_2b = jnp.concatenate([pixel_coords, pixel_coords], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        mask_2b = jnp.concatenate([prompt_mask, negative_mask], axis=0)
        v_2b = dit_apply(params, tokens_2b, coords_2b, timestep_2b, context_2b, mask_2b)
        v_cond, v_uncond = v_2b[:b], v_2b[b:]
        velocity = v_uncond + guidance_scale * (v_cond - v_uncond)

        next_tokens = scheduler.step(velocity, timestep, current_tokens)
        if image_cond_mask is not None:
            # Only step tokens whose eligibility timestep has been reached;
            # conditioning tokens keep their (lerp'd, not further-denoised)
            # value until then -- the reference's `denoising_step`'s
            # `tokens_to_denoise_mask`-gated `torch.where`.
            t_eps = 1e-6
            eligible = (t_val - t_eps) < (1.0 - image_cond_mask)
            next_tokens = jnp.where(eligible[..., None], next_tokens, current_tokens)
        return next_tokens

    logging.info(f"Running {scheduler.num_steps} sampling steps (sampler={args.sampler}, guidance_scale={args.guidance_scale})...")
    for step_index in range(scheduler.num_steps):
        tokens = single_step(
            tokens, step_index, prompt_embeds, prompt_mask, negative_embeds, negative_mask,
            pixel_coords, image_cond_mask, dit_params, args.guidance_scale)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    latents = unpatchify(tokens, latent_f, latent_h, latent_w)
    latents_unnorm = latents.astype(jnp.float32) * vae_std + vae_mean

    # The decoder is noise-conditioned (`timestep_conditioning=True` for
    # every released checkpoint) -- re-noise by `decode_noise_scale` before
    # decoding with `decode_timestep`, matching the reference exactly (see
    # `vidax.models.ltx_video.vae.Decoder`'s docstring for why this isn't
    # optional).
    decode_noise_rng, rng = jax.random.split(rng)
    decode_noise = jax.random.normal(decode_noise_rng, latents_unnorm.shape, dtype=jnp.float32)
    decode_noise_scale = args.decode_noise_scale if args.decode_noise_scale is not None else args.decode_timestep
    latents_unnorm = latents_unnorm * (1 - decode_noise_scale) + decode_noise * decode_noise_scale
    decode_timestep = jnp.full((batch_size,), args.decode_timestep, dtype=jnp.float32)

    decode_fn = jax.jit(lambda params, z, t: vae_model.apply(params, z, t, method=vae_model.decode))
    frames = decode_fn(vae_params, latents_unnorm.astype(dtype), decode_timestep)

    base, ext = os.path.splitext(args.output_path)
    for i in range(batch_size):
        video_frames = np.array(frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end T2V/I2V video generation with LTX-Video 0.9.8 on TPU.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the flat .safetensors checkpoint bundling both the DiT and VAE (e.g. ltxv-2b-0.9.8-distilled.safetensors).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard the DiT's/T5's attention heads and FFN channels across. Must divide num_devices and each model's own head count. 1 (replicated) works for the 2B checkpoint; the 13B checkpoints need more (their bf16 weights alone don't fit replicated on a single chip).")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to PixArt-XL-2-1024-MS's text_encoder/model.safetensors.index.json.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/../tokenizer'.")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to the whole batch) or exactly `batch_size` prompts.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance.")
    parser.add_argument("--image_path", type=str, default=None, help="Conditioning image for I2V. Omit for T2V.")
    parser.add_argument("--conditioning_strength", type=float, default=1.0, help="I2V only: how strongly the conditioning image is enforced (1.0 = the first latent frame IS the encoded image, no noise mixed in).")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale: velocity = uncond + guidance_scale * (cond - uncond). Set to 1.0 for distilled checkpoints (which don't need CFG per their own reference config).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the VAE, T5, and DiT activations/latents.")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Cast target for the DiT's weights specifically -- every released checkpoint ships natively as bfloat16.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of sampling steps.")
    parser.add_argument("--sampler", type=str, default="LinearQuadratic", choices=["Uniform", "LinearQuadratic", "Constant"], help="Sigma schedule shape -- every released checkpoint's own config uses 'LinearQuadratic'.")
    parser.add_argument("--shift", type=float, default=None, help="Required only for --sampler Constant.")
    parser.add_argument("--text_max_tokens", type=int, default=256, help="T5 prompt padding/truncation length (the reference pipeline's own default).")
    parser.add_argument("--decode_timestep", type=float, default=0.05, help="VAE decoder noise-conditioning timestep (the reference config's own default for every released checkpoint).")
    parser.add_argument("--decode_noise_scale", type=float, default=None, help="How much fresh noise to mix into the latent before decoding. Defaults to --decode_timestep, matching the reference.")
    parser.add_argument("--height", type=int, default=512, help="Output video height.")
    parser.add_argument("--width", type=int, default=768, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=97, help="Number of frames in the output video.")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s).")

    args = parser.parse_args()
    main(args)
