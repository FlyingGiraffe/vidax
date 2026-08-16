# End-to-end image-to-video inference script for Wan2.1 on TPU.
#
# I2V only ships as a 14B model (no 1.3B variant), and additionally needs a
# CLIP vision encoder checkpoint for image conditioning. See
# docs/models/wan2_1.md for where to get both. Otherwise this mirrors
# generate_wan2_1_t2v.py closely -- see its module docstring for the
# tensor-parallel/flash-attention/loop-unrolling rationale, which all
# applies unchanged here.

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
from jax.sharding import PartitionSpec as P, NamedSharding

from vidax.core.sharding import (
    build_tpu_mesh, shard_wan_params, get_replicated_sharding, get_batch_sharding,
    to_partition_specs, configure_jax_cache,
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_1.configs import I2V_14B_CONFIG
from vidax.models.wan.wan2_1.dit import WanDiT
from vidax.models.wan.wan2_1.vae import (
    WanVAEDecoder, WanVAEEncoder, Decoder3d, Encoder3d, _count_causal_convs,
    _count_causal_convs_encoder,
)
from vidax.models.wan.common.t5 import T5Encoder, Umt5Tokenizer
from vidax.models.wan.wan2_1.clip_vision import ClipVisionTransformer, preprocess_image_for_clip
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}

# Wan2.1's default I2V negative prompt: the shared t2v one with a "camera
# shake" exclusion prepended (Wan2.1-main/wan/configs/wan_i2v_14B.py).
DEFAULT_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
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


def encode_prompts(prompts: list, t5_model: T5Encoder, t5_params, tokenizer: Umt5Tokenizer,
                    dtype) -> jnp.ndarray:
    """Tokenizes and T5-encodes one prompt per batch element (see
    generate_wan2_1.py's copy of this function for why the zero-fill matters).
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)
    context = t5_model.apply(t5_params, ids, mask)
    seq_lens = mask.sum(axis=1)
    positions = jnp.arange(context.shape[1])[None, :]
    keep = positions < seq_lens[:, None]
    return jnp.where(keep[..., None], context, 0.0).astype(dtype)


def compute_latent_grid(image_h: int, image_w: int, max_area: int,
                         vae_stride: tuple, patch_size: tuple) -> tuple:
    """Reference's aspect-ratio-preserving resolution selection
    (`WanI2V.generate`): picks the largest (lat_h, lat_w) grid, aligned to
    both the VAE's spatial stride and the DiT's patch size, whose pixel-area
    is close to `max_area` while preserving the input image's aspect ratio.

    Returns (pixel_h, pixel_w, lat_h, lat_w).
    """
    aspect_ratio = image_h / image_w
    lat_h = round(
        math.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(
        math.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    return lat_h * vae_stride[1], lat_w * vae_stride[2], lat_h, lat_w


def build_i2v_conditioning(image: np.ndarray, num_frames: int, pixel_h: int, pixel_w: int,
                            latent_t: int, vae_model: WanVAEEncoder, vae_params, dtype):
    """Builds `y` (mask + VAE-encoded conditioning frame): matches the
    reference's `msk`/`y` construction in `WanI2V.generate`.

    The reference's mask (`msk`) ends up, after its reshape/transpose dance,
    equal to 1 for every channel at the first latent frame and 0 everywhere
    else -- constructed directly here rather than reproducing that dance.
    """
    img = jnp.asarray(image, dtype=jnp.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
    img = jax.image.resize(img, (pixel_h, pixel_w, 3), method="bicubic")

    video = jnp.zeros((num_frames, pixel_h, pixel_w, 3), dtype=jnp.float32)
    video = video.at[0].set(img)
    video = video[None].astype(dtype)

    # Encodes the *full* (num_frames-long, mostly-zero) video, not just the
    # one real frame -- the causal chunked encoder's output position/shape
    # depends on how many frames precede it, so this is what actually
    # produces correctly-shaped latents. `encode_chunk` (jit-wrapped here)
    # takes raw RGB pixels directly (Wan2.1's VAE has no pixel-unshuffle
    # pre-process step, unlike Wan2.2's).
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
    if args.shift is None:
        # Matches the reference's own auto-selection (Wan2.1-main/generate.py):
        # the 480P checkpoint is trained/tuned around shift=3.0, the 720P
        # checkpoint (and everything else) around shift=5.0.
        args.shift = 3.0 if args.max_area <= 832 * 480 else 5.0
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
    # See `generate_wan2_1_t2v.py`'s identical comment on `sequence_parallel`
    # for what this does, and how it composes with `--tensor_parallel_size`.
    dit_model = WanDiT(
        mesh=mesh, sequence_parallel=sequence_parallel, compute_dtype=dit_dtype,
        **I2V_14B_CONFIG)
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder()
    t5_model = T5Encoder()
    clip_model = ClipVisionTransformer()
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=args.shift)

    assert dit_model.num_heads % tp_size == 0, (
        f"WanDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}); e.g. tp in {{1,2,4,5,8,10,20,40}} for the 14B model.")
    assert t5_model.num_heads % tp_size == 0, (
        f"T5Encoder.num_heads ({t5_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")

    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(args.t5_checkpoint_path), "google", "umt5-xxl")
    tokenizer = Umt5Tokenizer(tokenizer_path, seq_len=dit_model.text_len)

    # --- Load weights (DiT, VAE, T5, and CLIP ship as separate checkpoints) ---
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="wan2.1_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(args.vae_checkpoint_path, model_type="wan2.1_vae")
    logging.info(f"Loading T5 weights from {args.t5_checkpoint_path}...")
    t5_params = load_torch_checkpoint_to_jax(args.t5_checkpoint_path, model_type="wan_t5")
    logging.info(f"Loading CLIP weights from {args.clip_checkpoint_path}...")
    clip_params = load_torch_checkpoint_to_jax(args.clip_checkpoint_path, model_type="wan2.1_clip")

    # Cast on the host (numpy) before device_put -- see
    # `vidax.translator.converter.convert_pt_tensor_to_jax`'s docstring.
    #
    # `dit_params` are cast to `--dit_dtype` (default float32), independent
    # of `--dtype` -- see the identical comment in `generate_wan2_1_t2v.py`.
    # Wan2.1's released DiT checkpoints ship as raw float32 on disk;
    # rounding them down to bf16 at load time (as this script used to do
    # unconditionally) produces severely degraded output once the video's
    # total token count is large enough, e.g. native 720p at 81 frames --
    # see docs/models/wan2_1.md#status.
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    vae_params = cast_to_dtype(vae_params, dtype)
    t5_params = cast_to_dtype(t5_params, dtype)
    clip_params = cast_to_dtype(clip_params, dtype)

    # DiT weights are always Megatron-sharded regardless of
    # `sequence_parallel` (see `generate_wan2_1_t2v.py`'s identical
    # comment -- weight-sharding and token-sharding are independent axes now).
    replicated = get_replicated_sharding(mesh)
    dit_shardings = shard_wan_params(dit_params, mesh)
    dit_params = jax.device_put(dit_params, dit_shardings)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)
    clip_params = jax.device_put(clip_params, replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Prepare inputs ---
    image = np.array(Image.open(args.image_path).convert("RGB"))
    pixel_h, pixel_w, lat_h, lat_w = compute_latent_grid(
        image.shape[0], image.shape[1], args.max_area, vae_stride=(4, 8, 8),
        patch_size=dit_model.patch_size)
    latent_t = 1 + (args.num_frames - 1) // 4
    logging.info(
        f"Input image {image.shape[1]}x{image.shape[0]} -> "
        f"output {pixel_w}x{pixel_h}, {args.num_frames} frames.")

    logging.info("Encoding conditioning image (VAE + CLIP)...")
    # `y` gets concatenated directly onto `latents` before any Dense layer
    # touches it (`WanDiT.__call__`'s `jnp.concatenate([latents, y], ...)`),
    # so it needs to match `latents`'s `dit_dtype`, not the general
    # `--dtype` -- see the identical comment on `latents`'s construction
    # below for why.
    y = build_i2v_conditioning(
        image, args.num_frames, pixel_h, pixel_w, latent_t, vae_encoder, vae_params, dit_dtype)
    y = jax.device_put(jnp.broadcast_to(y, (dp_size,) + y.shape[1:]), get_batch_sharding(mesh, y.ndim))

    clip_input = preprocess_image_for_clip(image).astype(dtype)
    clip_fea = clip_model.apply(clip_params, clip_input)  # (1, 257, image_dim)
    clip_fea = jax.device_put(
        jnp.broadcast_to(clip_fea, (dp_size,) + clip_fea.shape[1:]), get_batch_sharding(mesh, clip_fea.ndim))

    latents_shape = (dp_size, latent_t, lat_h, lat_w, dit_model.in_dim)
    latents_rng, rng = jax.random.split(rng)
    # `latents` are constructed in `dit_dtype`, not the general `--dtype` --
    # see the identical comment in `generate_wan2_1_t2v.py`.
    latents = jax.random.normal(latents_rng, latents_shape, dtype=dit_dtype)
    latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))

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

    # See `generate_wan2_1_t2v.py`'s identical comment on why
    # `sequence_parallel` needs `shard_map`, not just `jax.jit`. `y`/
    # `clip_fea` are batch-sharded on 'dp' (replicated on 'tp'), matching
    # the `get_batch_sharding` calls already applied to them above.
    def _dit_apply(params, latents, t, freqs, context, y, clip_fea):
        return dit_model.apply(
            params, latents=latents, t=t, freqs=freqs, context=context, y=y, clip_fea=clip_fea)

    if sequence_parallel:
        dit_apply = shard_map(
            _dit_apply, mesh=mesh,
            in_specs=(to_partition_specs(dit_shardings), P('dp', None, None, None, None),
                      P('dp'), (P(), P()), P('dp', None, None),
                      P('dp', None, None, None, None), P('dp', None, None)),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

    # --- Euler sampling loop (see generate_wan2_1.py for why this isn't one big jax.jit) ---
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_latents, step_index, prompt_embeds, negative_embeds, y, clip_fea,
                     freqs, params, guide_scale):
        b_size = current_latents.shape[0]
        t_val = scheduler.timesteps[step_index]
        t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)

        # CFG batched into one `2*B`-batch `dit_apply` call instead of two
        # separate dispatches -- see `generate_wan2_1_t2v.py`'s identical
        # comment. `y`/`clip_fea` (the image conditioning) don't vary with
        # the text prompt, so they're just duplicated alongside everything
        # else, not meaningfully recomputed.
        latents_2b = jnp.concatenate([current_latents, current_latents], axis=0)
        t_vec_2b = jnp.concatenate([t_vec, t_vec], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        y_2b = jnp.concatenate([y, y], axis=0)
        clip_fea_2b = jnp.concatenate([clip_fea, clip_fea], axis=0)
        v_2b = dit_apply(params, latents_2b, t_vec_2b, freqs, context_2b, y_2b, clip_fea_2b)
        v_cond, v_uncond = v_2b[:b_size], v_2b[b_size:]
        velocity = v_uncond + guide_scale * (v_cond - v_uncond)
        return scheduler.step(velocity, step_index, current_latents)

    logging.info(
        f"Running sampling for {args.num_steps} steps "
        f"(shift={args.shift}, guide_scale={args.guide_scale})...")
    for step_index in range(scheduler.num_steps):
        latents = single_step(
            latents, step_index, prompt_embeds, negative_embeds, y, clip_fea, freqs,
            dit_params, args.guide_scale)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    # Uses `WanVAEDecoder.decode_chunk` (jit-wrapped per frame) rather than a
    # single `vae_decoder.apply(vae_params, latents)` call -- see
    # `generate_wan2_1_t2v.py`'s identical decode section for why.
    x_full = vae_decoder.apply(vae_params, latents.astype(dtype), method=vae_decoder.pre_process)
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
    parser = argparse.ArgumentParser(description="End-to-end image-to-video generation with Wan 2.1 on TPU.")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the I2V-14B DiT .safetensors checkpoint.")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the VAE .pth checkpoint.")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to the T5 (umt5-xxl encoder) .pth checkpoint.")
    parser.add_argument("--clip_checkpoint_path", type=str, required=True, help="Path to the CLIP (models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth) checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the umt5-xxl HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/google/umt5-xxl'.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the conditioning image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance. Defaults to the reference's i2v `sample_neg_prompt`.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard each model's attention heads / FFN channels (weights) across. Must divide num_devices and num_heads (40 for the 14B DiT, 64 for the T5 encoder). Composes independently with --sequence_parallel_size.")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Number of devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of --tensor_parallel_size's weight-sharding. See generate_wan2_1_t2v.py's identical flag for the full reasoning. Also requires the DiT's patch token count to be evenly divisible by this value.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the VAE, T5, and CLIP (and cast target for their loaded checkpoints; also used for the DiT's activations/latents). Note: TPU's XLA backend does not implement float16 matmuls -- float16 will fail at runtime on TPU.")
    parser.add_argument("--dit_dtype", type=str, default="float32", choices=list(DTYPES.keys()), help="Cast target for the DiT's *weights* specifically, independent of --dtype. Defaults to float32 because Wan2.1's released DiT checkpoints ship as raw float32 on disk, and rounding them down to bfloat16 produces severely degraded (flat, hazy) output once the video's total token count is large enough (e.g. native 720p at 81 frames) -- see docs/models/wan2_1.md#status. Pass --dit_dtype bfloat16 to opt back into the ~2x smaller DiT weight footprint at smaller/safer scales (verified fine at this repo's existing 480p benchmarks).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of sampling steps. The reference's i2v default is 40 (vs 50 for t2v).")
    parser.add_argument("--shift", type=float, default=None, help="Flow-matching noise-schedule shift. Defaults to the reference's own auto-selection (Wan2.1-main/generate.py): 3.0 if --max_area is at the 480p checkpoint's scale (<= 832*480), 5.0 otherwise (720p checkpoint's scale). Pass explicitly to override.")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond).")
    parser.add_argument("--max_area", type=int, default=720 * 1280, help="Target output resolution's pixel area; actual (height, width) are derived from this and the input image's aspect ratio (see compute_latent_grid). Use 720*1280 with the 720P checkpoint, 480*832 with the 480P checkpoint -- the two ship as separate weights trained at different resolution ranges (Wan2.1-I2V-14B-480P/720P), unlike T2V's single 14B checkpoint used at any resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the output video.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With --tensor_parallel_size < num_devices (i.e. dp_size > 1), each data-parallel replica's independent sample is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
