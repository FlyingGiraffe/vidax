# End-to-end text-to-video / image-to-video inference script for
# HunyuanVideo-1.5 on TPU.
#
# T2V and I2V share this one script (same DiT class/checkpoint-shape
# family for a given --resolution/--task -- see
# vidax.models.hunyuan_video.hunyuan_video1_5.dit's module docstring):
# pass --image_path for I2V, omit it for T2V. --resolution/--task select
# which of the 4 core checkpoint variants (480p_t2v/480p_i2v/720p_t2v/
# 720p_i2v) to load and the matching default flow-match --shift.
#
# Scope for this first landing (see docs/models/hunyuan_video1_5.md):
# - Supports Megatron-style 1D tensor parallelism (--tensor_parallel_size)
#   for the DiT's double/single-stream blocks -- plain GSPMD
#   auto-partitioning (`vidax.core.sharding.shard_wan_params`), *not*
#   `shard_map`, except for the attention call itself: Pallas/Mosaic
#   kernels are opaque custom calls GSPMD can't auto-partition, so
#   `vidax.models.hunyuan_video.common.dit_layers.masked_self_attention`
#   wraps just that one call in `shard_map` when `mesh` is given (see its
#   docstring). No --sequence_parallel_size or weight offloading yet.
# - VAE decode is spatially tiled (--vae_tile_latent_size) so the
#   reference's own 121-frame default decodes without OOM -- see
#   spatial_tiled_vae_decode below and
#   docs/lessons/hunyuan_video1_5_debugging.md. No temporal tiling (the
#   reference VAE doesn't support it either).
# - No distilled/sparse-attention/SR checkpoint variants.
# - Real classifier-free guidance (uncond + guidance_scale * (cond -
#   uncond)), matching the reference (these checkpoints have
#   guidance_embed=False, i.e. no embedded/distilled-guidance path).

import argparse
import functools
import logging
import math
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vidax.core.sharding import build_tpu_mesh, configure_jax_cache, get_replicated_sharding, shard_wan_params
from vidax.models.cosmos2_5.reason1 import Qwen2TextModel
from vidax.models.hunyuan_video.hunyuan_video1_5.byt5 import byt5_encoder
from vidax.models.hunyuan_video.hunyuan_video1_5.configs import (
    dit_kwargs_from_transformer_config,
    default_shift_for,
    load_hunyuan_video1_5_transformer_config,
    load_hunyuan_video1_5_vae_config,
    vae_kwargs_from_vae_config,
)
from vidax.models.hunyuan_video.hunyuan_video1_5.dit import HunyuanVideo15DiT
from vidax.models.hunyuan_video.hunyuan_video1_5.qwen_text import (
    HunyuanVideoMLLMTokenizer,
    extract_hunyuan_mllm_embeddings,
)
from vidax.models.hunyuan_video.hunyuan_video1_5.siglip import SiglipVisionEncoder, siglip_kwargs_from_config
from vidax.models.hunyuan_video.hunyuan_video1_5.vae import HunyuanVideo15VAE, blend_h, blend_v
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}

BYT5_MAX_LENGTH = 256  # `byt5_max_length` default, hyvideo_pipeline.py:109
MLLM_MAX_LENGTH = 1000  # `self.text_len`, hunyuan_video_pipeline.py:1592


def cast_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """frames: (T, H, W, 3) uint8."""
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def load_conditioning_image(image_path: str, height: int, width: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    return arr  # (H, W, 3)


def compute_i2v_resolution(image_h: int, image_w: int, max_area: int, ffactor_spatial: int) -> "tuple[int, int]":
    """Aspect-ratio-preserving resolution selection for I2V, matching
    `generate_wan2_1_i2v.py`'s `compute_latent_grid`: picks the largest
    (height, width), aligned to the VAE's spatial stride, whose pixel area
    is close to `max_area` while preserving the *conditioning image's* own
    aspect ratio -- resizing a portrait image to a fixed landscape
    `--height`/`--width` (or vice versa) otherwise silently squishes it.
    HunyuanVideo-1.5 has `patch_size=[1,1,1]` (no additional DiT-side
    patchify beyond the VAE's own compression, see `dit.py`'s module
    docstring), so alignment is to `ffactor_spatial` alone -- no extra
    patch-size constraint the way Wan's own version also aligns to.
    """
    aspect_ratio = image_h / image_w
    lat_h = round(math.sqrt(max_area * aspect_ratio) // ffactor_spatial)
    lat_w = round(math.sqrt(max_area / aspect_ratio) // ffactor_spatial)
    return lat_h * ffactor_spatial, lat_w * ffactor_spatial


class ByT5PromptTokenizer:
    """Byte-level tokenizer for the glyph/color prompt path, mirroring
    `vidax.models.ltx_video.t5.PixArtT5Tokenizer`'s pattern -- fixed-length,
    zero-padded (ids, attention_mask), using byT5-small's real (Glyph-SDXL-
    v2-expanded) tokenizer.
    """

    def __init__(self, tokenizer_path: str, seq_len: int = BYT5_MAX_LENGTH):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.seq_len = seq_len

    def __call__(self, texts):
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True, max_length=self.seq_len)
        return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)


def main(args):
    configure_jax_cache()
    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]
    task = "i2v" if args.image_path is not None else "t2v"
    variant = f"{args.resolution}_{task}"
    logger.info("Loading HunyuanVideo-1.5 %s", variant)

    # Megatron-style 1D tensor parallelism for the DiT (the real memory
    # bottleneck: 8.3B params); everything else (VAE, Qwen, byT5, SigLIP --
    # none of which have TP-sharded weights, see docs/hardware_and_sharding.md)
    # is simply replicated across the same mesh, which comfortably fits
    # alongside the TP-sharded DiT (e.g. at tp=4: DiT ~4.15GB/chip bf16 +
    # Qwen ~14GB + VAE ~2.5GB + byT5 ~0.5GB replicated on every chip
    # ~21GB/chip, well under a TPU v4 chip's ~33GB).
    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size or num_devices
    assert num_devices % tp_size == 0, f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})."
    mesh = build_tpu_mesh(data_parallel_size=1, tensor_parallel_size=tp_size, sequence_parallel_size=1)
    replicated = get_replicated_sharding(mesh)

    # --- DiT ---
    transformer_dir = os.path.join(args.checkpoint_dir, "transformer", variant)
    dit_config = load_hunyuan_video1_5_transformer_config(transformer_dir)
    dit_kwargs = dit_kwargs_from_transformer_config(dit_config)
    assert dit_kwargs["heads_num"] % tp_size == 0, (
        f"HunyuanVideo15DiT's heads_num ({dit_kwargs['heads_num']}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    dit_model = HunyuanVideo15DiT(**dit_kwargs, mesh=mesh)
    dit_params = load_torch_checkpoint_to_jax(
        os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"),
        model_type="hunyuan_video1_5_dit")
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    dit_params = jax.device_put(dit_params, shard_wan_params(dit_params, mesh))

    # --- VAE ---
    vae_dir = os.path.join(args.checkpoint_dir, "vae")
    vae_config = load_hunyuan_video1_5_vae_config(vae_dir)
    vae_kwargs = vae_kwargs_from_vae_config(vae_config)
    vae_model = HunyuanVideo15VAE(**vae_kwargs)
    vae_params = load_torch_checkpoint_to_jax(
        os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), model_type="hunyuan_video1_5_vae")
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    scaling_factor = vae_config["scaling_factor"]
    shift_factor = vae_config.get("shift_factor")
    # Reference default: `tile_latent_min_size = sample_size // ffactor_spatial`
    # (`AutoencoderKLConv3D.__init__`), `tile_overlap_factor = 0.25`.
    tile_latent_min_size = args.vae_tile_latent_size or (vae_config["sample_size"] // vae_kwargs["ffactor_spatial"])
    tile_overlap_factor = 0.25

    # --- Qwen2.5-VL MLLM text tower ---
    qwen_model = Qwen2TextModel()
    qwen_params = load_torch_checkpoint_to_jax(
        os.path.join(args.checkpoint_dir, "text_encoder", "llm", "model.safetensors.index.json"),
        model_type="reason1_text_encoder")
    qwen_params = jax.device_put(cast_to_dtype(qwen_params, dtype), replicated)
    mllm_tokenizer = HunyuanVideoMLLMTokenizer(
        os.path.join(args.checkpoint_dir, "text_encoder", "llm"),
        max_length=MLLM_MAX_LENGTH, data_type="image" if task == "i2v" else "video")

    # --- byT5 glyph tower ---
    import torch
    byt5_sd = torch.load(
        os.path.join(args.checkpoint_dir, "text_encoder", "Glyph-SDXL-v2", "checkpoints", "byt5_model.pt"),
        map_location="cpu", weights_only=True)
    byt5_vocab_size = byt5_sd["embed_tokens.weight"].shape[0]
    byt5_model = byt5_encoder(byt5_vocab_size)
    byt5_params = load_torch_checkpoint_to_jax(
        os.path.join(args.checkpoint_dir, "text_encoder", "Glyph-SDXL-v2", "checkpoints", "byt5_model.pt"),
        model_type="hunyuan_video1_5_byt5")
    byt5_params = jax.device_put(cast_to_dtype(byt5_params, dtype), replicated)
    byt5_tokenizer = ByT5PromptTokenizer(os.path.join(args.checkpoint_dir, "text_encoder", "byt5-small"))

    # --- SigLIP (I2V only) ---
    siglip_model = siglip_params = siglip_processor = None
    if task == "i2v":
        if args.siglip_checkpoint_dir is None:
            raise ValueError("--siglip_checkpoint_dir is required for I2V.")
        import json
        with open(os.path.join(args.siglip_checkpoint_dir, "image_encoder", "config.json")) as f:
            siglip_config = json.load(f)
        siglip_model = SiglipVisionEncoder(**siglip_kwargs_from_config(siglip_config))
        siglip_params = load_torch_checkpoint_to_jax(
            os.path.join(args.siglip_checkpoint_dir, "image_encoder", "model.safetensors"),
            model_type="hunyuan_video1_5_siglip")
        siglip_params = jax.device_put(cast_to_dtype(siglip_params, dtype), replicated)
        from transformers import SiglipImageProcessor
        siglip_processor = SiglipImageProcessor.from_pretrained(
            os.path.join(args.siglip_checkpoint_dir, "feature_extractor"))

    # --- Prompt encoding ---
    def encode(prompt):
        ids, mask, crop_start = mllm_tokenizer([prompt])
        ids = jax.device_put(jnp.array(ids), replicated)
        mask = jax.device_put(jnp.array(mask), replicated)
        text_states, text_mask = extract_hunyuan_mllm_embeddings(
            qwen_params, ids, mask, crop_start, model=qwen_model)
        byt5_ids, byt5_mask = byt5_tokenizer([prompt])
        byt5_ids = jax.device_put(jnp.array(byt5_ids), replicated)
        byt5_mask_dev = jax.device_put(jnp.array(byt5_mask), replicated)
        byt5_raw = byt5_model.apply(byt5_params, byt5_ids, byt5_mask_dev)
        # Move everything to the DiT's device -- the DiT consumes all four
        # of these together, and mixed-device inputs to one `apply` call
        # would otherwise force an implicit (and easy to miss) transfer.
        return (jax.device_put(text_states.astype(dit_dtype), replicated),
                jax.device_put(text_mask, replicated),
                jax.device_put(byt5_raw.astype(dit_dtype), replicated),
                jax.device_put(jnp.array(byt5_mask), replicated))

    text_states, text_mask, byt5_states, byt5_mask = encode(args.prompt)
    neg_text_states, neg_text_mask, neg_byt5_states, neg_byt5_mask = encode(args.negative_prompt)

    # --- Resolution (T2V: --height/--width or the --resolution default;
    # I2V without an explicit override: derived from the conditioning
    # image's own aspect ratio, matching generate_wan2_1_i2v.py's
    # --max_area convention -- see compute_i2v_resolution's docstring for
    # why this matters (a fixed landscape default otherwise silently
    # squishes a portrait input image). ---
    ffs, fft = vae_kwargs["ffactor_spatial"], vae_kwargs["ffactor_temporal"]
    default_h, default_w = {"480p": (480, 832), "720p": (720, 1280)}[args.resolution]
    height, width = args.height, args.width
    if task == "i2v" and args.height is None and args.width is None:
        with Image.open(args.image_path) as _im:
            image_w0, image_h0 = _im.size
        max_area = args.max_area or (default_h * default_w)
        height, width = compute_i2v_resolution(image_h0, image_w0, max_area, ffs)
        logger.info("I2V: derived resolution %dx%d from %s's own aspect ratio (max_area=%d)",
                    height, width, args.image_path, max_area)
    else:
        height = height if height is not None else default_h
        width = width if width is not None else default_w
    # Write the resolved values back so anything reading args.height/
    # args.width after this point (e.g. benchmarks/common.py's result
    # metadata) sees the real resolution, not I2V's original None sentinel.
    args.height, args.width = height, width

    # --- Vision conditioning (I2V) ---
    latent_channels = vae_kwargs["latent_channels"]
    lt = (args.num_frames - 1) // fft + 1
    lh, lw = height // ffs, width // ffs

    vision_states = jax.device_put(jnp.zeros((1, 1, dit_kwargs["vision_states_dim"]), dtype=dit_dtype), replicated)
    cond_latents = jax.device_put(jnp.zeros((1, lt, lh, lw, latent_channels), dtype=dit_dtype), replicated)
    cond_mask = jax.device_put(jnp.zeros((1, lt, lh, lw, 1), dtype=dit_dtype), replicated)
    mask_type = "t2v"
    if task == "i2v":
        mask_type = "i2v"
        image = load_conditioning_image(args.image_path, height, width)
        image_uint8 = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
        pixel_values = siglip_processor.preprocess(images=[image_uint8], return_tensors="np")["pixel_values"]
        pixel_values = jax.device_put(jnp.array(np.transpose(pixel_values, (0, 2, 3, 1))), replicated)
        vision_states = siglip_model.apply(siglip_params, pixel_values.astype(dtype))
        vision_states = jax.device_put(vision_states.astype(dit_dtype), replicated)

        img_5d = jax.device_put(jnp.array(image)[None, None], replicated)  # (1, 1, H, W, 3)
        mean, _ = vae_model.apply(vae_params, img_5d, method=vae_model.encode)
        first_latent = mean if shift_factor is None else (mean - shift_factor)
        first_latent = first_latent * scaling_factor
        cond_latents = jnp.concatenate(
            [first_latent, jnp.zeros((1, lt - 1, lh, lw, latent_channels), dtype=first_latent.dtype)], axis=1
        ).astype(dit_dtype)
        cond_mask = jnp.concatenate(
            [jnp.ones((1, 1, lh, lw, 1)), jnp.zeros((1, lt - 1, lh, lw, 1))], axis=1
        ).astype(dit_dtype)
        cond_latents = jax.device_put(cond_latents, replicated)
        cond_mask = jax.device_put(cond_mask, replicated)

    # --- Sampling loop ---
    key = jax.random.PRNGKey(args.seed)
    latents = jax.random.normal(key, (1, lt, lh, lw, latent_channels), dtype=jnp.float32).astype(dit_dtype)
    latents = jax.device_put(latents, replicated)

    shift = args.shift if args.shift is not None else default_shift_for(args.resolution, task)
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=shift)

    # One jitted step (both CFG branches + the Euler update) per sampling
    # step, and one jitted VAE decode -- matches every other model's
    # example script (`@jax.jit`/`jax.jit(...)` inside `main`, see
    # `benchmarks/common.py`'s module docstring for why this placement
    # matters for the benchmark harness's compile-vs-generation split).
    # Always computes both cond/uncond branches (no runtime
    # `guidance_scale != 1.0` branch) -- `guidance_scale=1.0` still gives
    # the exact right answer (`uncond + 1*(cond-uncond) == cond`), and
    # avoiding the branch keeps this one `jax.jit`-traced program instead
    # of two.
    @functools.partial(jax.jit, static_argnames=("mask_type",))
    def sampling_step(
        dit_params, latents, timestep, dsigma,
        text_states, text_mask, byt5_states, byt5_mask,
        neg_text_states, neg_text_mask, neg_byt5_states, neg_byt5_mask,
        vision_states, cond_latents, cond_mask, guidance_scale, mask_type,
    ):
        lat_cf = jnp.moveaxis(latents, -1, 1)
        cond_cf = jnp.moveaxis(cond_latents, -1, 1)
        mask_cf = jnp.moveaxis(cond_mask, -1, 1)
        hidden_states = jnp.concatenate([lat_cf, cond_cf, mask_cf], axis=1)

        v_cond = dit_model.apply(
            dit_params, hidden_states, timestep, text_states, text_mask,
            vision_states=vision_states, byt5_text_states=byt5_states, byt5_text_mask=byt5_mask,
            mask_type=mask_type)
        v_uncond = dit_model.apply(
            dit_params, hidden_states, timestep, neg_text_states, neg_text_mask,
            vision_states=vision_states, byt5_text_states=neg_byt5_states, byt5_text_mask=neg_byt5_mask,
            mask_type=mask_type)
        v_cond = jnp.moveaxis(v_cond, 1, -1)
        v_uncond = jnp.moveaxis(v_uncond, 1, -1)
        v = (v_uncond + guidance_scale * (v_cond - v_uncond)).astype(jnp.float32)

        new_latents = latents.astype(jnp.float32) - v * dsigma
        return new_latents.astype(latents.dtype)

    # Staged (per-decoder-level), not one fused `jax.jit`, decode: at real
    # frame counts a single fused decode OOMs even though the VAE's own
    # weights are only ~2.5GB bf16 on an otherwise-dedicated chip -- XLA
    # doesn't free one level's temporaries before the next level's ops run
    # inside one program (same root cause `docs/lessons/ltx2_5_debugging.md`
    # already documented for LTX-2.5's DiT). Each level as its own
    # `jax.jit` call gives every level's temporaries a real chance to be
    # freed between levels -- see `Decoder.stage_level`'s docstring.
    @jax.jit
    def vae_decode_stage_in_and_mid(vae_params, z):
        return vae_model.apply(vae_params, z, method=vae_model.decode_stage_in_and_mid)

    @functools.partial(jax.jit, static_argnames=("i_level", "i_block"))
    def vae_decode_stage_level_block(vae_params, h, i_level, i_block):
        return vae_model.apply(vae_params, h, i_level, i_block, method=vae_model.decode_stage_level_block)

    @functools.partial(jax.jit, static_argnames=("i_level",))
    def vae_decode_stage_level_upsample(vae_params, h, i_level):
        return vae_model.apply(vae_params, h, i_level, method=vae_model.decode_stage_level_upsample)

    @jax.jit
    def vae_decode_stage_out(vae_params, h):
        return vae_model.apply(vae_params, h, method=vae_model.decode_stage_out)  # (1, T, H, W, 3), raw (not yet clipped)

    def vae_decode_tile(vae_params, latent_tile):
        """Fully-staged (per-decoder-block) decode of one latent tile --
        returns raw floating pixel values (not yet clipped/uint8'd, since
        `spatial_tiled_vae_decode` blends adjacent tiles' *raw* output,
        matching the reference's own `spatial_tiled_decode`, which blends
        before the pipeline-level uint8 conversion -- blending after
        quantization would introduce banding at tile seams).
        """
        h = vae_decode_stage_in_and_mid(vae_params, latent_tile)
        for i_level in range(vae_model.num_decoder_levels):
            for i_block in range(vae_model.num_blocks_per_level):
                h = vae_decode_stage_level_block(vae_params, h, i_level, i_block)
            h = vae_decode_stage_level_upsample(vae_params, h, i_level)
        return vae_decode_stage_out(vae_params, h)[0]  # (T, H, W, 3)

    def spatial_tiled_vae_decode(vae_params, latents_for_decode):
        """Port of `AutoencoderKLConv3D.spatial_tiled_decode`: tile the
        *latent* H/W, decode each tile (small enough to comfortably fit
        one chip even at the real 121-frame default -- confirmed even
        finer per-ResnetBlock staging of the *whole* volume still OOM'd at
        480x832x121, since staging only avoids accumulating multiple
        stages' temporaries, it can't shrink one stage's own tensor once a
        single tile is already the full frame), then linearly cross-fade
        overlapping edges (`blend_h`/`blend_v`) and crop to avoid a
        doubled-up seam. Falls back to a single "tile" (the whole volume)
        when the latent is already small enough -- same as
        `HunyuanVideo15VAE.decode`'s single-full-volume-tile default, at
        which point this is equivalent to (but slower than, from the extra
        per-block staging) calling `vae_decode_tile` once directly.
        """
        b, t, h_lat, w_lat, c = latents_for_decode.shape
        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
        tile_pixel_size = tile_latent_min_size * vae_kwargs["ffactor_spatial"]
        blend_extent = int(tile_pixel_size * tile_overlap_factor)
        row_limit = tile_pixel_size - blend_extent

        rows = []
        for i in range(0, h_lat, overlap_size):
            row = []
            for j in range(0, w_lat, overlap_size):
                tile = latents_for_decode[:, :, i:i + tile_latent_min_size, j:j + tile_latent_min_size, :]
                row.append(vae_decode_tile(vae_params, tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :row_limit, :row_limit, :])
            result_rows.append(jnp.concatenate(result_row, axis=-2))  # W axis
        dec = jnp.concatenate(result_rows, axis=-3)  # H axis
        return jnp.clip((dec + 1) * 127.5, 0, 255).astype(jnp.uint8)

    guidance_scale = jnp.asarray(args.guidance_scale, dtype=jnp.float32)
    for step in range(args.num_steps):
        t = jax.device_put(jnp.reshape(scheduler.timesteps[step], (1,)), replicated)
        dsigma = jax.device_put((scheduler.sigmas[step] - scheduler.sigmas[step + 1]).astype(jnp.float32), replicated)

        latents = sampling_step(
            dit_params, latents, t, dsigma,
            text_states, text_mask, byt5_states, byt5_mask,
            neg_text_states, neg_text_mask, neg_byt5_states, neg_byt5_mask,
            vision_states, cond_latents, cond_mask, guidance_scale, mask_type)
        logger.info("step %d/%d done", step + 1, args.num_steps)

    # --- VAE decode ---
    latents_for_decode = latents.astype(dtype) / scaling_factor
    if shift_factor is not None:
        latents_for_decode = latents_for_decode + shift_factor
    latents_for_decode = jax.device_put(latents_for_decode, replicated)
    pixels = np.array(spatial_tiled_vae_decode(vae_params, latents_for_decode))

    save_video(pixels, args.output_path, fps=args.fps)
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Root dir containing transformer/, vae/, text_encoder/{llm,byt5-small,Glyph-SDXL-v2}/ (i.e. tencent/HunyuanVideo-1.5's downloaded layout).")
    parser.add_argument("--siglip_checkpoint_dir", type=str, default=None,
                         help="Path to black-forest-labs/FLUX.1-Redux-dev's downloaded dir (image_encoder/ + feature_extractor/ subfolders). Required for I2V.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                         help="Number of devices to Megatron-shard the DiT's attention heads/FFN channels across. Must divide num_devices and heads_num (16). Defaults to every local device.")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default=None, help="Conditioning image for I2V. Omit for T2V.")
    parser.add_argument("--height", type=int, default=None,
                         help="Output height. Defaults to --resolution's own default (480p: 480, 720p: 720) for T2V; for I2V, derived from the conditioning image's own aspect ratio unless both --height and --width are given explicitly.")
    parser.add_argument("--width", type=int, default=None,
                         help="Output width. See --height.")
    parser.add_argument("--max_area", type=int, default=None,
                         help="I2V only, when --height/--width aren't both given: target output pixel area used together with the conditioning image's aspect ratio (see compute_i2v_resolution). Defaults to --resolution's own default area (480p: 480*832, 720p: 720*1280).")
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=None, help="Defaults to the per-(resolution,task) value from PIPELINE_CONFIGS.")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--vae_tile_latent_size", type=int, default=None,
                         help="Latent-space spatial tile size for `spatial_tiled_vae_decode` (pixel tile size = this * ffactor_spatial). Defaults to the reference's own `sample_size // ffactor_spatial` (16); shrink this (e.g. 8) if VAE decode OOMs -- more likely at --tensor_parallel_size > 1, where the other (replicated) components' resident weights leave less headroom per chip than the single-device-per-component layout this default was sized for.")
    args = parser.parse_args()
    main(args)
