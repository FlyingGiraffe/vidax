# End-to-end image-to-video inference script for HunyuanVideo-I2V's
# `token_replace` mode (the released `hunyuan-video-i2v-720p` checkpoint's
# default/only shipped config -- `latent_concat`, the reference's other
# I2V mode, is not ported, see docs/models/hunyuan_video.md).
#
# A **separate script** from `generate_hunyuan_video.py`, not a shared
# `--image_path` branch (unlike `generate_hunyuan_video1_5.py`'s T2V/I2V
# unification) -- HunyuanVideo 1.0's T2V and I2V ship genuinely different
# checkpoints (`hunyuan-video-t2v-720p` vs `hunyuan-video-i2v-720p`) *and*
# entirely different text encoders (plain Llama vs the full multimodal
# LLaVA), much more divergent than 1.5's T2V/I2V (one checkpoint/DiT class,
# differing only in whether conditioning tensors are zero) -- matches
# `generate_wan2_1_t2v.py`/`generate_wan2_1_i2v.py`'s own precedent of
# separate scripts when the checkpoints/encoders genuinely differ.
#
# Architecture (see `hunyuan_video.llava_text`/`llava_vision`'s module
# docstrings for the full derivation):
# - No channel-concat: the reference image's own clean VAE-encoded latent
#   literally replaces the first *latent* frame before every sampling
#   step; the DiT's AdaLN modulation uses a second "as-if-t=0" vector for
#   exactly that first frame's tokens (`i2v_condition_type="token_replace"`
#   on `HunyuanVideoDiT`).
# - Text conditioning is the *full* multimodal LLaVA model: the reference
#   image is projected through a CLIP ViT-L/14-336 vision tower + a 2-layer
#   projector, spliced into the Llama decoder's input embedding sequence
#   at the `<image>` placeholder's fixed positions, and the resulting
#   hidden states are split back into "image"/"text" regions and
#   re-concatenated -- see `extract_hunyuan_llava_embeddings`.
# - Two separate image preprocessing pipelines for the same reference
#   photo: one for the VAE/`token_replace` latent (resized to the video's
#   own target resolution bucket), one for the CLIP vision tower (resized
#   to CLIP's fixed 336x336) -- see `preprocess_image_for_vae`/
#   `preprocess_image_for_llava`.
# - The unconditional (negative) branch's semantic image is a *black*
#   image of the same size as the real reference image (matches the
#   reference's own `black_image(...)` call), not a zeroed-out feature.
#
# Real defaults, confirmed against the checkpoint's own recommended launch
# script (`scripts/run_sample_image2video_dynamic.sh`) rather than
# `config.py`'s bare argparse defaults (which don't match, e.g.
# `--embedded-cfg-scale` defaults to `None` in argparse but the shipped
# script always passes `6.0`): `--flow-shift 17.0` (**not** T2V's 7.0),
# `--embedded_guidance_scale 6.0`, `--num_steps 50`, `--num_frames 129`,
# `--i2v_resolution 720p`. The real checkpoint's own `guidance_in.*` keys
# confirm `guidance_embed=True` (`"HYVideo-T/2-cfgdistill"` preset)
# regardless of the example script's own `--model HYVideo-T/2` flag (a
# likely documentation inconsistency in the upstream repo -- the actual
# checkpoint's param-tree only exact-matches the cfgdistill preset, 856/856
# leaves, confirmed directly).

import argparse
import functools
import gc
import logging
import os
import time

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vidax.core.sharding import build_tpu_mesh, configure_jax_cache, get_replicated_sharding, shard_wan_params
from vidax.models.hunyuan_video.common.dit_layers import MMDoubleStreamBlock, MMSingleStreamBlock
from vidax.models.hunyuan_video.hunyuan_video.clip_text import ClipTextModel, extract_clip_pooled
from vidax.models.hunyuan_video.hunyuan_video.configs import (
    DIT_CONFIGS,
    dit_kwargs_from_config,
    load_hunyuan_video_vae_config,
    vae_kwargs_from_vae_config,
)
from vidax.models.hunyuan_video.hunyuan_video.dit import HunyuanVideoDiT
from vidax.models.hunyuan_video.hunyuan_video.llama_text import LlamaTextModel
from vidax.models.hunyuan_video.hunyuan_video.llava_text import (
    LlavaPromptTokenizer,
    compute_i2v_closest_size,
    extract_hunyuan_llava_embeddings,
    preprocess_image_for_llava,
    preprocess_image_for_vae,
)
from vidax.models.hunyuan_video.hunyuan_video.llava_vision import ClipVisionModel, LlavaMultiModalProjector
from vidax.models.hunyuan_video.hunyuan_video.vae import HunyuanVideoVAE, blend_h, blend_v
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
CLIP_MAX_LENGTH = 77  # config.py's `--text-len-2` default.


class ClipPromptTokenizer:
    """`--text-encoder-2`/`--tokenizer-2` both default to `"clipL"`
    *unconditionally* -- I2V still uses the plain pooled CLIP-L text
    encoder for `vector_in`'s input, exactly like T2V (confirmed against
    `config.py`: `--text-encoder-2`/`--tokenizer-2` aren't gated on
    `--i2v-mode` at all) -- reused verbatim from
    `generate_hunyuan_video.py`."""

    def __init__(self, tokenizer_path: str):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=CLIP_MAX_LENGTH)

    def __call__(self, texts):
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True, max_length=CLIP_MAX_LENGTH)
        return encoded["input_ids"].astype(np.int32)


def cast_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """frames: (T, H, W, 3) uint8."""
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def black_image(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), (0, 0, 0))


def main(args):
    configure_jax_cache()
    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]

    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size or num_devices
    assert num_devices % tp_size == 0, f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})."
    mesh = build_tpu_mesh(data_parallel_size=1, tensor_parallel_size=tp_size, sequence_parallel_size=1)
    replicated = get_replicated_sharding(mesh)

    # --- DiT (token_replace: same architecture/shapes as T2V, real,
    # different trained weights -- confirmed via an exact 856/856 param
    # match against the real checkpoint) ---
    dit_config = DIT_CONFIGS[args.model]
    dit_kwargs = dit_kwargs_from_config(dit_config)
    assert dit_kwargs["heads_num"] % tp_size == 0, (
        f"HunyuanVideoDiT's heads_num ({dit_kwargs['heads_num']}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    dit_model = HunyuanVideoDiT(**dit_kwargs, i2v_condition_type="token_replace", mesh=mesh)
    dit_params = load_torch_checkpoint_to_jax(
        os.path.join(args.checkpoint_dir, "hunyuan-video-i2v-720p", "transformers", "mp_rank_00_model_states.pt"),
        model_type="hunyuan_video_dit")
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    dit_shardings = shard_wan_params(dit_params, mesh)

    # See generate_hunyuan_video.py's identical comment for the full
    # reasoning -- unchanged here (`token_replace` doesn't change block
    # shapes/depths).
    if args.offload_dit_weights:
        n_double = dit_model.mm_double_blocks_depth
        n_single = dit_model.mm_single_blocks_depth
        double_chunk_size = args.offload_chunk_size_double or n_double
        single_chunk_size = args.offload_chunk_size_single or n_single
        assert n_double % double_chunk_size == 0, (
            f"--offload_chunk_size_double ({double_chunk_size}) must divide "
            f"HunyuanVideoDiT.mm_double_blocks_depth ({n_double}) -- see docs/weight_offloading.md.")
        assert n_single % single_chunk_size == 0, (
            f"--offload_chunk_size_single ({single_chunk_size}) must divide "
            f"HunyuanVideoDiT.mm_single_blocks_depth ({n_single}) -- see docs/weight_offloading.md.")

        double_chunks_host = [
            [dit_params["params"][f"double_blocks_{i}"] for i in range(c, c + double_chunk_size)]
            for c in range(0, n_double, double_chunk_size)
        ]
        single_chunks_host = [
            [dit_params["params"][f"single_blocks_{i}"] for i in range(c, c + single_chunk_size)]
            for c in range(0, n_single, single_chunk_size)
        ]
        double_chunk_sharding = [dit_shardings["params"]["double_blocks_0"]] * double_chunk_size
        single_chunk_sharding = [dit_shardings["params"]["single_blocks_0"]] * single_chunk_size

        nonblock_params = {
            k: v for k, v in dit_params["params"].items()
            if not (k.startswith("double_blocks_") or k.startswith("single_blocks_"))
        }
        nonblock_shardings = {
            k: v for k, v in dit_shardings["params"].items()
            if not (k.startswith("double_blocks_") or k.startswith("single_blocks_"))
        }
        dit_params = jax.device_put({"params": nonblock_params}, {"params": nonblock_shardings})
    else:
        dit_params = jax.device_put(dit_params, dit_shardings)

    # --- VAE (byte-identical checkpoint to T2V's -- see this module's
    # docstring; --vae_checkpoint_dir defaults to --checkpoint_dir's own
    # hunyuan-video-i2v-720p/vae, which is the same file) ---
    vae_dir = os.path.join(args.vae_checkpoint_dir or args.checkpoint_dir, "hunyuan-video-i2v-720p", "vae")
    vae_config = load_hunyuan_video_vae_config(vae_dir)
    vae_kwargs = vae_kwargs_from_vae_config(vae_config)
    vae_model = HunyuanVideoVAE(**vae_kwargs)
    vae_params = load_torch_checkpoint_to_jax(
        os.path.join(vae_dir, "pytorch_model.pt"), model_type="hunyuan_video_vae")
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    scaling_factor = vae_config["scaling_factor"]
    ffs = 2 ** int(np.log2(vae_kwargs["spatial_compression_ratio"]))
    fft = vae_kwargs["time_compression_ratio"]
    tile_latent_min_size = args.vae_tile_latent_size or (vae_config.get("sample_size", 256) // ffs)
    tile_overlap_factor = 0.25

    @jax.jit
    def vae_encode(vae_params, pixel_values):
        mean, _logvar = vae_model.apply(vae_params, pixel_values, method=vae_model.encode)
        return mean * scaling_factor

    # --- Reference image: resolution bucket + the two preprocessing paths ---
    image = Image.open(args.image_path).convert("RGB")
    closest_w, closest_h = compute_i2v_closest_size(image, args.i2v_resolution)
    logger.info("Reference image %s -> target resolution %dx%d", args.image_path, closest_w, closest_h)

    pixel_values_vae = preprocess_image_for_vae(image, (closest_w, closest_h))
    pixel_values_vae = jax.device_put(cast_to_dtype(pixel_values_vae, dtype), replicated)
    img_latents = vae_encode(vae_params, pixel_values_vae)  # (1, 1, lh, lw, latent_channels)
    img_latents = jax.device_put(img_latents.astype(dit_dtype), replicated)

    pixel_values_clip = preprocess_image_for_llava(image)
    pixel_values_clip = jax.device_put(cast_to_dtype(pixel_values_clip, dtype), replicated)
    black_pixel_values_clip = preprocess_image_for_llava(black_image(*image.size))
    black_pixel_values_clip = jax.device_put(cast_to_dtype(black_pixel_values_clip, dtype), replicated)

    # --- LLaVA text/vision encoder (full multimodal checkpoint) ---
    llama_model = LlamaTextModel()
    clip_model = ClipVisionModel()
    projector = LlavaMultiModalProjector()
    llava_index = os.path.join(args.llava_checkpoint_dir, "model.safetensors.index.json")
    llama_params = load_torch_checkpoint_to_jax(llava_index, model_type="hunyuan_video_llava_llama_text")
    llama_params = jax.device_put(cast_to_dtype(llama_params, dtype), replicated)
    clip_params = load_torch_checkpoint_to_jax(llava_index, model_type="hunyuan_video_clip_vision")
    clip_params = jax.device_put(cast_to_dtype(clip_params, dtype), replicated)
    projector_params = load_torch_checkpoint_to_jax(llava_index, model_type="hunyuan_video_llava_projector")
    projector_params = jax.device_put(cast_to_dtype(projector_params, dtype), replicated)
    tokenizer = LlavaPromptTokenizer(args.llava_checkpoint_dir)

    # --- CLIP-L pooled text encoder (--text-encoder-2, same tower/checkpoint
    # as T2V -- see this module's docstring on why this isn't zeroed) ---
    clip_text_model = ClipTextModel()
    clip_text_params = load_torch_checkpoint_to_jax(
        os.path.join(args.clip_checkpoint_dir, "model.safetensors"), model_type="hunyuan_video_clip_text")
    clip_text_params = jax.device_put(cast_to_dtype(clip_text_params, dtype), replicated)
    clip_text_tokenizer = ClipPromptTokenizer(args.clip_checkpoint_dir)

    def encode(prompt, pixel_values):
        expanded_ids, image_start, image_end, raw_ids, raw_mask = tokenizer([prompt])
        text_states, text_mask = extract_hunyuan_llava_embeddings(
            llama_params, clip_params, projector_params,
            expanded_ids, raw_ids, raw_mask, image_start, image_end, pixel_values,
            llama_model, clip_model, projector,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            image_embed_interleave=args.image_embed_interleave)

        clip_ids = clip_text_tokenizer([prompt])
        clip_ids_d = jax.device_put(jnp.array(clip_ids), replicated)
        text_states_2 = extract_clip_pooled(clip_text_params, clip_ids_d, model=clip_text_model)

        return (jax.device_put(text_states.astype(dit_dtype), replicated),
                jax.device_put(text_mask, replicated),
                jax.device_put(text_states_2.astype(dit_dtype), replicated))

    text_states, text_mask, text_states_2 = encode(args.prompt, pixel_values_clip)
    negative_prompt = args.negative_prompt if args.negative_prompt else (
        args.negative_prompt_default if args.guidance_scale != 1.0 else "")
    neg_text_states, neg_text_mask, neg_text_states_2 = encode(negative_prompt, black_pixel_values_clip)

    # See generate_hunyuan_video.py's identical comment: the LLaVA tower
    # (+ CLIP-L text encoder) are ~16-17GB/chip replicated and unneeded
    # past this point -- free them before the DiT sampling loop allocates
    # its own activations.
    del (encode, llama_model, llama_params, clip_model, clip_params, projector, projector_params, tokenizer,
         clip_text_model, clip_text_params, clip_text_tokenizer)
    gc.collect()

    # --- Resolution / frame count (derived from the reference image, not
    # explicit --height/--width -- matches the reference's own i2v-resolution
    # bucketing) ---
    height, width = closest_h, closest_w
    assert (args.num_frames - 1) % 4 == 0, "--num_frames must be `1 + 4k`."

    latent_channels = vae_kwargs["latent_channels"]
    lt = (args.num_frames - 1) // fft + 1
    lh, lw = height // ffs, width // ffs
    assert img_latents.shape[2:4] == (lh, lw), (
        f"img_latents' own spatial shape {img_latents.shape[2:4]} doesn't match the derived latent "
        f"grid ({lh}, {lw}) -- the VAE's spatial_compression_ratio and the closest_size bucketing "
        f"must agree; this should never happen.")

    # --- Sampling loop ---
    key = jax.random.PRNGKey(args.seed)
    latents = jax.random.normal(key, (1, lt, lh, lw, latent_channels), dtype=jnp.float32).astype(dit_dtype)
    # The first latent frame is always the reference image's own clean
    # latent, never noise -- see this module's docstring on `token_replace`.
    latents = latents.at[:, :1].set(img_latents)
    latents = jax.device_put(latents, replicated)

    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=args.shift)

    guidance_embed = dit_kwargs["guidance_embed"]
    embedded_guidance = None
    if guidance_embed:
        embedded_guidance = jax.device_put(
            jnp.asarray([args.embedded_guidance_scale * 1000.0], dtype=jnp.float32).astype(dit_dtype), replicated)

    @jax.jit
    def sampling_step(
        dit_params, latents, timestep, dsigma,
        text_states, text_mask, text_states_2, neg_text_states, neg_text_mask, neg_text_states_2,
        img_latents, embedded_guidance, guidance_scale,
    ):
        latents = latents.at[:, :1].set(img_latents)
        lat_cf = jnp.moveaxis(latents, -1, 1)

        v_cond = dit_model.apply(
            dit_params, lat_cf, timestep, text_states, text_mask, text_states_2,
            guidance=embedded_guidance)
        v_uncond = dit_model.apply(
            dit_params, lat_cf, timestep, neg_text_states, neg_text_mask, neg_text_states_2,
            guidance=embedded_guidance)
        v_cond = jnp.moveaxis(v_cond, 1, -1)
        v_uncond = jnp.moveaxis(v_uncond, 1, -1)
        v = (v_uncond + guidance_scale * (v_cond - v_uncond)).astype(jnp.float32)

        # Only the non-first frames are ever denoised -- frame 0 is
        # re-substituted with the clean reference latent every step (see
        # this module's docstring).
        new_rest = latents[:, 1:].astype(jnp.float32) - v[:, 1:] * dsigma
        new_latents = jnp.concatenate([latents[:, :1], new_rest.astype(latents.dtype)], axis=1)
        return new_latents

    if args.offload_dit_weights:
        # See generate_hunyuan_video.py's identical comment for the full
        # reasoning behind this split -- unchanged here except for the
        # `text_states_2`-free DiT signature (I2V's `vector_in` input is a
        # constant zero vector, see below) and the per-step first-frame
        # substitution.
        def _pre_process_body(params, hidden_states, timestep, text_states, text_mask, text_states_2, guidance):
            img, txt, vec, freqs, key_valid, _img_len, _tt, _th, _tw, tr_vec, tr_n = dit_model.apply(
                params, hidden_states, timestep, text_states, text_mask, text_states_2,
                guidance=guidance, method=dit_model.pre_process)
            return img, txt, vec, freqs, key_valid, tr_vec

        pre_apply = jax.jit(_pre_process_body)

        def _chunk_forward_double_body(chunk_params, img, txt, vec, freqs, key_valid, token_replace_vec, first_frame_token_num):
            for layer_params in chunk_params:
                img, txt = MMDoubleStreamBlock(
                    hidden_size=dit_model.hidden_size, heads_num=dit_model.heads_num,
                    mlp_width_ratio=dit_model.mlp_width_ratio, mlp_act_type=dit_model.mlp_act_type,
                    qk_norm=dit_model.qk_norm, qkv_bias=dit_model.qkv_bias, mesh=mesh,
                ).apply({"params": layer_params}, img, txt, vec, freqs, key_valid, token_replace_vec, first_frame_token_num)
            return img, txt

        # `first_frame_token_num` must be static (used for slice bounds).
        chunk_forward_double = jax.jit(_chunk_forward_double_body, static_argnums=(7,), donate_argnums=(0,))

        def _chunk_forward_single_body(chunk_params, x, vec, txt_len, freqs, key_valid, token_replace_vec, first_frame_token_num):
            for layer_params in chunk_params:
                x = MMSingleStreamBlock(
                    hidden_size=dit_model.hidden_size, heads_num=dit_model.heads_num,
                    mlp_width_ratio=dit_model.mlp_width_ratio, mlp_act_type=dit_model.mlp_act_type,
                    qk_norm=dit_model.qk_norm, mesh=mesh,
                ).apply({"params": layer_params}, x, vec, txt_len, freqs, key_valid, token_replace_vec, first_frame_token_num)
            return x

        chunk_forward_single = jax.jit(_chunk_forward_single_body, static_argnums=(3, 7), donate_argnums=(0,))

        post_apply = jax.jit(
            lambda params, x, vec, img_len, tt, th, tw: dit_model.apply(
                params, x, vec, img_len, tt, th, tw, method=dit_model.post_process),
            static_argnums=(3, 4, 5, 6))

        def sampling_step_offloaded(
            nonblock_params, latents, timestep, dsigma,
            text_states, text_mask, text_states_2, neg_text_states, neg_text_mask, neg_text_states_2,
            img_latents, embedded_guidance, guidance_scale,
        ):
            latents = latents.at[:, :1].set(img_latents)
            lat_cf = jnp.moveaxis(latents, -1, 1)
            lat_cf_2b = jnp.concatenate([lat_cf, lat_cf], axis=0)
            t_2b = jnp.concatenate([timestep, timestep], axis=0)
            text_states_2b = jnp.concatenate([text_states, neg_text_states], axis=0)
            text_mask_2b = jnp.concatenate([text_mask, neg_text_mask], axis=0)
            text_states_2_2b = jnp.concatenate([text_states_2, neg_text_states_2], axis=0)
            guidance_2b = (
                jnp.concatenate([embedded_guidance, embedded_guidance], axis=0)
                if embedded_guidance is not None else None)

            pt, ph, pw = dit_model.patch_size
            _, _, ot, oh, ow = lat_cf_2b.shape
            tt, th, tw = ot // pt, oh // ph, ow // pw
            first_frame_token_num = th * tw

            img, txt, vec, freqs, key_valid, token_replace_vec = pre_apply(
                nonblock_params, lat_cf_2b, t_2b, text_states_2b, text_mask_2b, text_states_2_2b, guidance_2b)
            img_len = img.shape[1]
            for chunk_host in double_chunks_host:
                chunk_params = jax.device_put(chunk_host, double_chunk_sharding)
                img, txt = chunk_forward_double(chunk_params, img, txt, vec, freqs, key_valid, token_replace_vec, first_frame_token_num)

            x, txt_len = HunyuanVideoDiT.mid_process(img, txt)

            for chunk_host in single_chunks_host:
                chunk_params = jax.device_put(chunk_host, single_chunk_sharding)
                x = chunk_forward_single(chunk_params, x, vec, txt_len, freqs, key_valid, token_replace_vec, first_frame_token_num)

            v_2b = post_apply(nonblock_params, x, vec, img_len, tt, th, tw)
            v_cond, v_uncond = jnp.split(v_2b, 2, axis=0)
            v_cond = jnp.moveaxis(v_cond, 1, -1)
            v_uncond = jnp.moveaxis(v_uncond, 1, -1)
            v = (v_uncond + guidance_scale * (v_cond - v_uncond)).astype(jnp.float32)

            new_rest = latents[:, 1:].astype(jnp.float32) - v[:, 1:] * dsigma
            new_latents = jnp.concatenate([latents[:, :1], new_rest.astype(latents.dtype)], axis=1)
            return new_latents

    # --- Staged (per-decoder-level) VAE decode -- see generate_hunyuan_video.py's
    # identical section for the full reasoning. ---
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
        return vae_model.apply(vae_params, h, method=vae_model.decode_stage_out)

    def vae_decode_tile(vae_params, latent_tile):
        h = vae_decode_stage_in_and_mid(vae_params, latent_tile)
        for i_level in range(vae_model.num_decoder_levels):
            for i_block in range(vae_model.num_blocks_per_level):
                h = vae_decode_stage_level_block(vae_params, h, i_level, i_block)
            h = vae_decode_stage_level_upsample(vae_params, h, i_level)
        return vae_decode_stage_out(vae_params, h)[0]  # (T, H, W, 3)

    def spatial_tiled_vae_decode(vae_params, latents_for_decode):
        b, t, h_lat, w_lat, c = latents_for_decode.shape
        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
        tile_pixel_size = tile_latent_min_size * ffs
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
            result_rows.append(jnp.concatenate(result_row, axis=-2))
        dec = jnp.concatenate(result_rows, axis=-3)
        return jnp.clip((dec + 1) * 127.5, 0, 255).astype(jnp.uint8)

    guidance_scale = jnp.asarray(args.guidance_scale, dtype=jnp.float32)
    step_fn = sampling_step_offloaded if args.offload_dit_weights else sampling_step
    for step in range(args.num_steps):
        t = jax.device_put(jnp.reshape(scheduler.timesteps[step], (1,)), replicated)
        dsigma = jax.device_put((scheduler.sigmas[step] - scheduler.sigmas[step + 1]).astype(jnp.float32), replicated)

        step_t0 = time.perf_counter()
        latents = step_fn(
            dit_params, latents, t, dsigma,
            text_states, text_mask, text_states_2, neg_text_states, neg_text_mask, neg_text_states_2,
            img_latents, embedded_guidance, guidance_scale)
        jax.block_until_ready(latents)
        logger.info("step %d/%d done (%.1fs)", step + 1, args.num_steps, time.perf_counter() - step_t0)

    # --- VAE decode ---
    latents_for_decode = jax.device_put((latents.astype(dtype) / scaling_factor), replicated)
    pixels = np.array(spatial_tiled_vae_decode(vae_params, latents_for_decode))

    save_video(pixels, args.output_path, fps=args.fps)
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Root dir containing hunyuan-video-i2v-720p/{transformers,vae}/ "
                              "(i.e. tencent/HunyuanVideo-I2V's downloaded layout).")
    parser.add_argument("--vae_checkpoint_dir", type=str, default=None,
                         help="Defaults to --checkpoint_dir (the I2V VAE checkpoint is byte-identical to T2V's, "
                              "see this module's docstring) -- override to point at a different downloaded root.")
    parser.add_argument("--llava_checkpoint_dir", type=str, required=True,
                         help="Path to the *full* xtuner/llava-llama-3-8b-v1_1-transformers downloaded root "
                              "(vision tower + projector + language model -- not the T2V-only extracted "
                              ".language_model subset generate_hunyuan_video.py uses).")
    parser.add_argument("--clip_checkpoint_dir", type=str, required=True,
                         help="Path to openai/clip-vit-large-patch14's downloaded root -- the plain pooled "
                              "CLIP-L text encoder (--text-encoder-2), used unconditionally (not disabled for "
                              "I2V) -- see this module's docstring.")
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill", choices=list(DIT_CONFIGS.keys()),
                         help="Named hyperparameter preset -- see this module's docstring on why cfgdistill is "
                              "used despite the reference's own example script passing a different --model value.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                         help="Number of devices to Megatron-shard the DiT's attention heads/FFN channels across. "
                              "Must divide num_devices and heads_num (24). Defaults to every local device.")
    parser.add_argument("--image_path", type=str, required=True, help="Reference/conditioning image.")
    parser.add_argument("--i2v_resolution", type=str, default="720p", choices=["720p", "540p", "360p"],
                         help="Target resolution bucket -- output height/width are derived from the reference "
                              "image's own aspect ratio snapped to this bucket's candidate sizes, not given "
                              "directly (matches the reference's own --i2v-resolution).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None,
                         help="Defaults to --negative_prompt_default when --guidance_scale != 1.0, else empty.")
    parser.add_argument("--negative_prompt_default", type=str,
                         default="deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs",
                         help="constants.py's NEGATIVE_PROMPT_I2V.")
    parser.add_argument("--num_frames", type=int, default=129, help="Must be `1 + 4k`.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=17.0,
                         help="config.py's --flow-shift; the real launch script's value (17.0), not the bare "
                              "argparse default -- I2V uses a very different shift than T2V's 7.0, confirmed "
                              "against scripts/run_sample_image2video_dynamic.sh.")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                         help="Real classifier-free guidance scale. Default 1.0 (off) -- this checkpoint's "
                              "embedded/distilled guidance (--embedded_guidance_scale) is the primary mechanism.")
    parser.add_argument("--embedded_guidance_scale", type=float, default=6.0,
                         help="Embedded/distilled guidance scale fed to `guidance_in` -- the real launch script's "
                              "value, confirmed against scripts/run_sample_image2video_dynamic.sh.")
    parser.add_argument("--hidden_state_skip_layer", type=int, default=2)
    parser.add_argument("--image_embed_interleave", type=int, default=4,
                         help="inference.py's real per-mode value for token_replace (vs. 1 for plain T2V-style "
                              "text-only encoding, 2 for latent_concat) -- subsamples every Nth projected image "
                              "patch row before splicing into the text-state sequence.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--vae_tile_latent_size", type=int, default=None,
                         help="Latent-space spatial tile size for the tiled VAE decode. Shrink (e.g. 8) if VAE "
                              "decode OOMs.")
    parser.add_argument("--offload_dit_weights", action="store_true",
                         help="Keep the double/single-stream blocks' weights host-resident and offload one "
                              "--offload_chunk_size_{double,single}-block group's worth into HBM at a time during "
                              "the sampling loop -- see generate_hunyuan_video.py's identical flag/docs/"
                              "weight_offloading.md.")
    parser.add_argument("--offload_chunk_size_double", type=int, default=None,
                         help="See generate_hunyuan_video.py's identical flag. Defaults to 20 (all in one chunk).")
    parser.add_argument("--offload_chunk_size_single", type=int, default=None,
                         help="See generate_hunyuan_video.py's identical flag. Defaults to 40 (all in one chunk).")
    args = parser.parse_args()
    main(args)
