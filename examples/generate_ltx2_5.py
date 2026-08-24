# End-to-end text-to-video / image-to-video inference script for LTX-2.5
# (video-only) on TPU.
#
# T2V and I2V share this one script: pass --image_path for I2V, omit it for
# T2V. I2V conditioning (`VideoConditionByLatentIndex` + the ancestral-Euler
# loop's `post_process_latent` masking, both read directly from the
# reference -- see `ltx_core/conditioning/types/latent_cond.py` and
# `ltx_pipelines/utils/{helpers,samplers}.py`) works differently from
# LTX-Video's own I2V: instead of clamping a per-token *timestep*, LTX-2.5
# threads an explicit per-token `denoise_mask` (`1` = fully denoise, `0` =
# frozen at its clean/conditioning value) through three places every step:
# (1) `timesteps = denoise_mask * sigma` fed to the DiT (so a frozen
# token's own AdaLN sees timestep 0, i.e. "already clean"), (2) the x0
# estimate is blended back toward the clean value
# (`denoised*mask + clean*(1-mask)`), and (3) the sampler's stepped output
# gets the same blend applied again. The initial noisy latent is built the
# same way: `lerp(clean_latent, noise, denoise_mask)`.
#
# Scope (see docs/models/ltx2_5.md's Status section for the full list):
# - **Video-only.** No audio generation -- see vidax.models.ltx2_5.dit's
#   module docstring for why (skips an entire second modality's weights).
# - **Single-stage.** No LatentUpsampler / half-res-then-2x-refine second
#   pass -- generates directly at --height/--width/--num_frames.
# - Supports Megatron-style 1D tensor parallelism (--tensor_parallel_size,
#   see vidax.core.sharding) for the DiT, the embeddings connector, and the
#   Gemma-4 text encoder -- required even at tp=1 replication limits: the
#   22B DiT's bf16 weights alone (~44GB) and the 12B Gemma encoder's
#   (~24GB) don't fit replicated on a single TPU v4 chip's ~32GB HBM.
#   LTX-2.5's own submodule names (to_q/to_k/to_v/to_out_0/ff_proj/ff_out
#   for the DiT/connector; q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
#   down_proj for Gemma) already match vidax.core.sharding's existing
#   whitelist (from LTX-Video/Reason1/Cosmos3) -- no sharding.py changes
#   were needed for this port.
# - `--sampler distilled` (default, 8 ancestral-Euler steps, eta=1.0, no
#   CFG) or `--sampler dev` (30 plain-Euler steps, eta=0.0, real CFG at
#   guidance_scale=3.0, a token-count-dependent shifted sigma schedule --
#   see vidax.schedulers.ltx2_5_ancestral_euler). Plain constant-scale CFG
#   only, no STG/audio-guidance-term/per-sigma-bucket guidance schedule
#   (the reference's own dev recipe uses those; pure inference-loop
#   refinements on a working base model, same scope decision as
#   LTX-Video's own port).

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
from vidax.models.ltx2_5.configs import (
    VAE_CONFIG, connector_kwargs_from_transformer_config, dit_kwargs_from_transformer_config,
    gemma4_text_model_kwargs, load_gemma4_config, load_ltx2_5_metadata, vae_scale_factors,
)
from vidax.models.ltx2_5.connector import Embeddings1DConnector
from vidax.models.ltx2_5.dit import LTXDiT
from vidax.models.ltx2_5.gemma4 import Gemma4Tokenizer, Gemma4TextModel, extract_video_features
from vidax.models.ltx2_5.patchifier import get_latent_coord_bounds, latent_to_pixel_coord_bounds, patchify, unpatchify
from vidax.models.ltx2_5.vae import LTXVAE
from vidax.schedulers.ltx2_5_ancestral_euler import DEV_CFG_GUIDANCE_SCALE, AncestralEulerScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax
from vidax.translator.mappings.ltx2_5 import load_gemma4_video_aggregate_embed

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}

# `ltx_pipelines.utils.constants.DEFAULT_NEGATIVE_PROMPT` -- read directly
# from the reference during this port.
DEFAULT_NEGATIVE_PROMPT = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    logging.info(f"Saving {frames.shape[0]} frames to {output_path}...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info("Video saved successfully.")


def load_conditioning_image(image_path: str, height: int, width: int) -> np.ndarray:
    """Loads and resizes a conditioning image to (1, 1, H, W, 3) in [-1, 1]."""
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return arr[None, None]  # (1, 1, H, W, 3)


def cast_to_dtype(tree, dtype):
    def cast_leaf(x):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def encode_prompts(prompts, gemma_model, gemma_params, video_kernel, video_bias,
                    connector_model, connector_params, tokenizer, dtype):
    """Gemma-4 forward -> `extract_video_features` -> embeddings connector,
    the real pipeline's `EmbeddingsProcessor.process_hidden_states`
    (see vidax.models.ltx2_5.gemma4/connector module docstrings). Runs once
    per prompt, reused across every denoising step (unchanged for a fixed
    prompt) -- same reasoning as T5 encoding in
    examples/generate_ltx_video.py.
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask, dtype=jnp.float32)
    hidden_states = gemma_model.apply(gemma_params, ids, mask)
    video_feats = extract_video_features(hidden_states, mask, video_kernel, video_bias).astype(dtype)

    additive_mask = ((1.0 - mask) * jnp.finfo(jnp.float32).min)[:, None, None, :]
    context, out_mask = connector_model.apply(connector_params, video_feats, additive_mask)
    # `out_mask` (the connector's own (B, 1, 1, L) *additive* mask, all-zero
    # once its learnable registers have substituted every padded position
    # -- see vidax.models.ltx2_5.connector's module docstring) is discarded
    # here, not threaded into the DiT: `LTXDiT.pre_process`'s
    # `encoder_attention_mask` expects a plain (B, L) *binary* mask and
    # builds its own additive bias from it -- feeding the connector's
    # already-additive, already-4D mask through that same `[:, None,
    # None, :]` broadcast corrupts it to rank 6 (a real bug this exact
    # wiring hit during a real end-to-end run: a downstream attention
    # einsum failed with "wrong number of indices"). The reference itself
    # passes `context_mask=None` at this point in the pipeline
    # (`modality_from_latent_state`) for exactly this reason -- masking is
    # already handled upstream by the connector, not needed again here.
    return context.astype(dtype)


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

    # --- Build models from each checkpoint's own embedded architecture config ---
    dit_metadata = load_ltx2_5_metadata(args.dit_checkpoint_path)["transformer"]
    causal_fix = dit_metadata.get("causal_temporal_positioning", False)

    dit_model = LTXDiT(**dit_kwargs_from_transformer_config(dit_metadata), compute_dtype=dit_dtype)
    connector_model = Embeddings1DConnector(
        **connector_kwargs_from_transformer_config(dit_metadata), compute_dtype=dtype)

    vae_kwargs = dict(
        latent_channels=VAE_CONFIG["latent_channels"], encoder_blocks=VAE_CONFIG["encoder_blocks"],
        decoder_blocks=VAE_CONFIG["decoder_blocks"], patch_size=VAE_CONFIG["patch_size"],
        base_channels=VAE_CONFIG["base_channels"], causal_decoder=VAE_CONFIG["causal_decoder"],
        timestep_conditioning=VAE_CONFIG["timestep_conditioning"])
    vae_model = LTXVAE(**vae_kwargs)
    temporal_scale, spatial_scale = vae_scale_factors(VAE_CONFIG)

    gemma_config = load_gemma4_config(args.text_encoder_checkpoint_path)
    gemma_model = Gemma4TextModel(**gemma4_text_model_kwargs(gemma_config), compute_dtype=dtype)
    tokenizer = Gemma4Tokenizer(args.text_encoder_checkpoint_path, seq_len=args.text_max_tokens)

    assert dit_model.num_attention_heads % tp_size == 0, (
        f"LTXDiT.num_attention_heads ({dit_model.num_attention_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    assert gemma_model.num_attention_heads % tp_size == 0, (
        f"Gemma4TextModel.num_attention_heads ({gemma_model.num_attention_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")

    # --- Load weights ---
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="ltx2_5_dit")
    connector_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="ltx2_5_connector")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(args.vae_checkpoint_path, model_type="ltx2_5_vae")
    logging.info(f"Loading Gemma-4 weights from {args.text_encoder_checkpoint_path}...")
    gemma_params = load_torch_checkpoint_to_jax(args.text_encoder_checkpoint_path, model_type="gemma4_text")
    import safetensors.numpy
    gemma_sd = safetensors.numpy.load_file(args.text_encoder_checkpoint_path)
    video_kernel, video_bias = load_gemma4_video_aggregate_embed(gemma_sd)
    video_kernel = jnp.asarray(video_kernel, dtype=dtype)
    video_bias = jnp.asarray(video_bias, dtype=dtype)
    del gemma_sd

    replicated = get_replicated_sharding(mesh)
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    connector_params = cast_to_dtype(connector_params, dtype)
    gemma_params = cast_to_dtype(gemma_params, dtype)
    dit_params = jax.device_put(dit_params, shard_wan_params(dit_params, mesh))
    connector_params = jax.device_put(connector_params, shard_wan_params(connector_params, mesh))
    gemma_params = jax.device_put(gemma_params, shard_wan_params(gemma_params, mesh))
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Prepare inputs ---
    batch_size = len(args.prompt)
    latent_f = 1 + (args.num_frames - 1) // temporal_scale
    latent_h = args.height // spatial_scale
    latent_w = args.width // spatial_scale

    latent_shape = (batch_size, latent_f, latent_h, latent_w, dit_model.in_channels)
    num_tokens = latent_f * latent_h * latent_w

    scheduler = AncestralEulerScheduler(
        sampler=args.sampler, sigmas=(jnp.asarray(args.sigmas, dtype=jnp.float32) if args.sigmas else None),
        eta=args.eta, num_tokens=num_tokens)
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None
        else (1.0 if args.sampler == "distilled" else DEV_CFG_GUIDANCE_SCALE))
    use_cfg = guidance_scale != 1.0

    logging.info(f"Encoding {batch_size} prompt(s) with Gemma-4: {args.prompt}")
    context = encode_prompts(
        args.prompt, gemma_model, gemma_params, video_kernel, video_bias,
        connector_model, connector_params, tokenizer, dtype)
    if use_cfg:
        logging.info(f"CFG enabled (guidance_scale={guidance_scale}): encoding negative prompt.")
        negative_context = encode_prompts(
            [args.negative_prompt] * batch_size, gemma_model, gemma_params, video_kernel, video_bias,
            connector_model, connector_params, tokenizer, dtype)

    latent_coord_bounds = get_latent_coord_bounds(latent_f, latent_h, latent_w, batch_size)
    pixel_coord_bounds = latent_to_pixel_coord_bounds(
        latent_coord_bounds, temporal_scale, spatial_scale, causal_fix=causal_fix)

    # --- I2V conditioning: VAE-encode the image into the first latent frame,
    # `clean_latent`/`denoise_mask` grids (all-ones mask = plain T2V, see
    # file docstring) ---
    denoise_mask_grid = jnp.ones((batch_size, latent_f, latent_h, latent_w, 1), dtype=jnp.float32)
    clean_latent = jnp.zeros(latent_shape, dtype=jnp.float32)
    if args.image_path is not None:
        logging.info(f"I2V: encoding conditioning image {args.image_path}")
        image = load_conditioning_image(args.image_path, args.height, args.width)
        image = np.broadcast_to(image, (batch_size,) + image.shape[1:])
        cond_latent = vae_model.apply(vae_params, jnp.asarray(image, dtype=dtype), method=vae_model.encode)
        clean_latent = clean_latent.at[:, :1].set(cond_latent.astype(jnp.float32))
        strength = args.conditioning_strength
        denoise_mask_grid = denoise_mask_grid.at[:, :1].set(1.0 - strength)

    denoise_mask = patchify(denoise_mask_grid)  # (B, N, 1)
    clean_tokens = patchify(clean_latent)  # (B, N, C)

    noise_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, clean_tokens.shape, dtype=jnp.float32)
    init_tokens = denoise_mask * noise + (1.0 - denoise_mask) * clean_tokens
    tokens = init_tokens.astype(dit_dtype)

    dit_apply = jax.jit(lambda params, tokens, coords, timestep, sigma, context: dit_model.apply(
        params, tokens, coords, timestep, sigma, context))

    @partial(jax.jit, donate_argnums=(0,), static_argnames=("use_cfg",))
    def single_step(current_tokens, step_index, context, negative_context, pixel_coord_bounds,
                     denoise_mask, clean_tokens, params, noise_rng, guidance_scale, use_cfg):
        b = current_tokens.shape[0]
        sigma_val = scheduler.sigmas[step_index]
        timestep = denoise_mask[..., 0] * sigma_val  # (B, N) per-token, see file docstring.

        if use_cfg:
            tokens_2b = jnp.concatenate([current_tokens, current_tokens], axis=0)
            timestep_2b = jnp.concatenate([timestep, timestep], axis=0)
            sigma_2b = jnp.full((2 * b,), sigma_val, dtype=jnp.float32)
            coords_2b = jnp.concatenate([pixel_coord_bounds, pixel_coord_bounds], axis=0)
            context_2b = jnp.concatenate([context, negative_context], axis=0)
            v_2b = dit_apply(params, tokens_2b, coords_2b, timestep_2b, sigma_2b, context_2b)
            v_cond, v_uncond = v_2b[:b], v_2b[b:]
            velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            sigma = jnp.full((b,), sigma_val, dtype=jnp.float32)
            velocity = dit_apply(params, current_tokens, pixel_coord_bounds, timestep, sigma, context)

        denoised = current_tokens.astype(jnp.float32) - velocity.astype(jnp.float32) * sigma_val
        denoised = denoise_mask * denoised + (1.0 - denoise_mask) * clean_tokens

        noise = jax.random.normal(noise_rng, current_tokens.shape, dtype=jnp.float32)
        next_tokens = scheduler.step(denoised, current_tokens, step_index, noise)
        next_tokens = denoise_mask * next_tokens + (1.0 - denoise_mask) * clean_tokens
        return next_tokens.astype(current_tokens.dtype)

    logging.info(
        f"Running {scheduler.num_steps} Euler sampling steps (eta={scheduler.eta}, "
        f"guidance_scale={guidance_scale})...")
    for step_index in range(scheduler.num_steps):
        step_rng, rng = jax.random.split(rng)
        tokens = single_step(
            tokens, step_index, context, (negative_context if use_cfg else context), pixel_coord_bounds,
            denoise_mask, clean_tokens, dit_params, step_rng, guidance_scale, use_cfg)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    latents = unpatchify(tokens, latent_f, latent_h, latent_w)
    decode_fn = jax.jit(lambda params, z: vae_model.apply(params, z, method=vae_model.decode))
    frames = decode_fn(vae_params, latents.astype(dtype))

    base, ext = os.path.splitext(args.output_path)
    for i in range(batch_size):
        video_frames = np.array(frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-video generation with LTX-2.5 (video-only) on TPU.")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors checkpoint (bundles the DiT and the video embeddings connector).")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to ltx-2.5-video-vae-conv-bf16.safetensors.")
    parser.add_argument("--text_encoder_checkpoint_path", type=str, required=True, help="Path to gemma4-12b-with-proj-ltx-2.5-bf16.safetensors (bundles the Gemma-4 text tower, its embedded tokenizer, and the video_aggregate_embed feature projection).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of devices to Megatron-shard the DiT/connector/Gemma-4's attention heads and FFN channels across. Required at any value >1 device -- neither the 22B DiT nor the 12B Gemma encoder's bf16 weights fit replicated on a single TPU v4 chip.")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to the whole batch) or exactly `batch_size` prompts.")
    parser.add_argument("--image_path", type=str, default=None, help="Conditioning image for I2V. Omit for T2V.")
    parser.add_argument("--conditioning_strength", type=float, default=1.0, help="I2V only: how strongly the conditioning image is enforced (1.0 = the first latent frame is frozen at the encoded image, never denoised).")
    parser.add_argument("--text_max_tokens", type=int, default=256, help="Tokenizer padding length.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the VAE, Gemma-4, connector, and DiT activations/latents.")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Cast target for the DiT's weights specifically -- every released checkpoint ships natively as bfloat16.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--sampler", type=str, default="distilled", choices=["distilled", "dev"], help="Named sigma schedule + real recipe: 'distilled' (8 ancestral-Euler steps, eta=1.0, no CFG) or 'dev' (30 plain-Euler steps, eta=0.0, guidance_scale=3.0 CFG) -- see vidax.schedulers.ltx2_5_ancestral_euler.")
    parser.add_argument("--sigmas", type=float, default=None, nargs="+", help="Explicit sigma schedule override (space-separated, descending, ending in 0.0), instead of --sampler's built-in schedule.")
    parser.add_argument("--eta", type=float, default=None, help="Euler eta (1.0 = fully ancestral/SDE, 0.0 = plain deterministic). Defaults to --sampler's own real value (1.0 distilled, 0.0 dev).")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Classifier-free guidance scale: velocity = uncond + guidance_scale * (cond - uncond). Defaults to --sampler's own real value (1.0/no-CFG for distilled, 3.0 for dev).")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance (dev checkpoint / guidance_scale != 1.0 only).")
    parser.add_argument("--height", type=int, default=704, help="Output height in pixels.")
    parser.add_argument("--width", type=int, default=1216, help="Output width in pixels.")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of output frames (1 + 8*k).")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate.")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path.")
    main(parser.parse_args())
