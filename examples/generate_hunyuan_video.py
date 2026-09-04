# End-to-end text-to-video inference script for HunyuanVideo 1.0 on TPU
# (the `hunyuan-video-t2v-720p` checkpoint). I2V is a separate checkpoint
# and text encoder -- see `generate_hunyuan_video_i2v.py`.
#
# The reference ships one released checkpoint variant
# (`tencent/HunyuanVideo`'s `hunyuan-video-t2v-720p/`, the
# `"HYVideo-T/2-cfgdistill"` preset, `guidance_embed=True`) -- confirmed by
# reading `hyvideo/config.py`'s `--dit-weight` default and the real
# checkpoint's own `guidance_in.*` keys (see
# docs/lessons/hunyuan_video_debugging.md). `--model-resolution`/544p in
# the reference is dead code for the default CLI path (only ever consulted
# when `--dit-weight` is *not* given, which it always is) -- there is no
# separate 544p checkpoint, just a different runtime `--height`/`--width`
# on this one 720p-native checkpoint, so no `--resolution` flag here.
#
# Scope (see docs/models/hunyuan_video.md):
# - Real classifier-free guidance is optional (`--guidance_scale`, default
#   1.0 == off, matching the reference's own `sample_video.py` default) on
#   top of this checkpoint's embedded/distilled guidance
#   (`--embedded_guidance_scale`, default 6.0, always applied via
#   `guidance_in` since this checkpoint's `guidance_embed=True`) -- both
#   branches are always computed in one jitted program regardless of
#   `--guidance_scale`'s value, same convention as
#   `generate_hunyuan_video1_5.py`.
# - VAE decode is spatially tiled (`--vae_tile_latent_size`), same staged
#   per-level/per-block decode pattern as `generate_hunyuan_video1_5.py`
#   (see that script's module docstring + `hunyuan_video.vae`'s
#   `Decoder.stage_level`/`stage_level_block`/`stage_level_upsample`
#   docstrings for why a fused decode OOMs at real frame counts). No
#   temporal tiling (the reference VAE doesn't support it either).
# - Supports `--tensor_parallel_size` (Megatron-style, via
#   `vidax.core.sharding.shard_wan_params`) for the DiT, matching
#   `generate_hunyuan_video1_5.py`.

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
from vidax.models.hunyuan_video.hunyuan_video.llama_text import LlamaTextModel, extract_hunyuan_llm_embeddings
from vidax.models.hunyuan_video.hunyuan_video.vae import HunyuanVideoVAE, blend_h, blend_v
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}

# `constants.py`'s PROMPT_TEMPLATE["dit-llm-encode-video"] -- the only
# template `sample_video.py` ever actually uses (`data_type="video"` for
# any `video_length > 1`, see `hyvideo/inference.py`).
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
CROP_START = 95  # PROMPT_TEMPLATE["dit-llm-encode-video"]["crop_start"]
TEXT_LEN = 256  # config.py's `--text-len` default -- post-crop LLM sequence length.
LLM_TOKENIZE_MAX_LENGTH = TEXT_LEN + CROP_START  # inference.py: `max_length = args.text_len + crop_start`.
CLIP_MAX_LENGTH = 77  # config.py's `--text-len-2` default.
HIDDEN_STATE_SKIP_LAYER = 2  # config.py's `--hidden-state-skip-layer` default.
NEGATIVE_PROMPT = (
    "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, "
    "bad teeth, bad eyes, bad limbs, distortion"
)  # constants.py's NEGATIVE_PROMPT.


def cast_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    """frames: (T, H, W, 3) uint8."""
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def align_to(value: int, alignment: int) -> int:
    return int(np.ceil(value / alignment) * alignment)


class LlamaPromptTokenizer:
    """Wraps a caption in `PROMPT_TEMPLATE_ENCODE_VIDEO` and tokenizes it,
    right-padded to `LLM_TOKENIZE_MAX_LENGTH` -- matches
    `TextEncoder.text2tokens`'s `"dit-llm-encode-video"` path.
    """

    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")

    def __call__(self, texts):
        formatted = [PROMPT_TEMPLATE_ENCODE_VIDEO.format(t) for t in texts]
        encoded = self.tokenizer(
            formatted, return_tensors="np", padding="max_length", truncation=True,
            max_length=LLM_TOKENIZE_MAX_LENGTH)
        return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)


class ClipPromptTokenizer:
    def __init__(self, tokenizer_path: str):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=CLIP_MAX_LENGTH)

    def __call__(self, texts):
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True, max_length=CLIP_MAX_LENGTH)
        return encoded["input_ids"].astype(np.int32)


def main(args):
    configure_jax_cache()
    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]

    num_devices = jax.device_count()
    tp_size = args.tensor_parallel_size or num_devices
    assert num_devices % tp_size == 0, f"num_devices ({num_devices}) must be divisible by --tensor_parallel_size ({tp_size})."
    mesh = build_tpu_mesh(data_parallel_size=1, tensor_parallel_size=tp_size, sequence_parallel_size=1)
    replicated = get_replicated_sharding(mesh)

    # --- DiT ---
    dit_config = DIT_CONFIGS[args.model]
    dit_kwargs = dit_kwargs_from_config(dit_config)
    assert dit_kwargs["heads_num"] % tp_size == 0, (
        f"HunyuanVideoDiT's heads_num ({dit_kwargs['heads_num']}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    dit_model = HunyuanVideoDiT(**dit_kwargs, mesh=mesh)
    dit_params = load_torch_checkpoint_to_jax(
        os.path.join(args.checkpoint_dir, "hunyuan-video-t2v-720p", "transformers", "mp_rank_00_model_states.pt"),
        model_type="hunyuan_video_dit")
    dit_params = cast_to_dtype(dit_params, dit_dtype)
    dit_shardings = shard_wan_params(dit_params, mesh)

    # `--offload_dit_weights`: keep the double/single-stream blocks'
    # ("double_blocks_*"/"single_blocks_*") params as a host-resident numpy
    # pytree instead of `device_put`-ing the whole tree -- only offloaded
    # into a small HBM buffer, `--offload_chunk_size_{double,single}`
    # consecutive blocks of one stream at a time, inside the sampling loop
    # itself (see `single_step_offloaded` below and
    # docs/weight_offloading.md). Two
    # separate chunk pools (double vs. single) since the two block types
    # have different param shapes -- the "every chunk is interchangeable"
    # trick from docs/weight_offloading.md still holds *within* each pool.
    # The rest of the DiT's params (patchify/time/text-embed/txt_in/
    # final_layer) are a tiny fraction of the 60-block total, so they're
    # still `device_put` once, same as always.
    if args.offload_dit_weights:
        # The two streams' chunk sizes are independent -- they're offloaded
        # via entirely separate chunk pools/jit calls below, so there's no
        # actual reason to force one shared size that divides both 20 and
        # 40 (the old constraint here needlessly kept single-stream blocks
        # at 2 chunks of 20 instead of 1 chunk of 40 whenever
        # --offload_chunk_size was set to double's full depth). Each
        # defaults to its own full block depth (i.e. "offload once per
        # stream") unless overridden down for tighter HBM budgets.
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

    # --- VAE ---
    vae_dir = os.path.join(args.checkpoint_dir, "hunyuan-video-t2v-720p", "vae")
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

    # --- Llama text tower ---
    llama_model = LlamaTextModel()
    llama_params = load_torch_checkpoint_to_jax(
        os.path.join(args.text_encoder_dir, "model.safetensors.index.json"), model_type="hunyuan_video_llama_text")
    llama_params = jax.device_put(cast_to_dtype(llama_params, dtype), replicated)
    llama_tokenizer = LlamaPromptTokenizer(args.text_encoder_dir)

    # --- CLIP-L pooled text encoder ---
    clip_model = ClipTextModel()
    clip_params = load_torch_checkpoint_to_jax(
        os.path.join(args.clip_checkpoint_dir, "model.safetensors"), model_type="hunyuan_video_clip_text")
    clip_params = jax.device_put(cast_to_dtype(clip_params, dtype), replicated)
    clip_tokenizer = ClipPromptTokenizer(args.clip_checkpoint_dir)

    # --- Prompt encoding ---
    def encode(prompt):
        ids, mask = llama_tokenizer([prompt])
        ids_d = jax.device_put(jnp.array(ids), replicated)
        text_states = extract_hunyuan_llm_embeddings(
            llama_params, ids_d, jnp.array(mask), hidden_state_skip_layer=HIDDEN_STATE_SKIP_LAYER, model=llama_model)
        # Crop off the template's own instruction tokens (see module docstring).
        text_states = text_states[:, CROP_START:]
        text_mask = mask[:, CROP_START:]

        clip_ids = clip_tokenizer([prompt])
        clip_ids_d = jax.device_put(jnp.array(clip_ids), replicated)
        text_states_2 = extract_clip_pooled(clip_params, clip_ids_d, model=clip_model)

        return (jax.device_put(text_states.astype(dit_dtype), replicated),
                jax.device_put(jnp.array(text_mask), replicated),
                jax.device_put(text_states_2.astype(dit_dtype), replicated))

    text_states, text_mask, text_states_2 = encode(args.prompt)
    negative_prompt = args.negative_prompt if args.negative_prompt else (
        NEGATIVE_PROMPT if args.guidance_scale != 1.0 else "")
    neg_text_states, neg_text_mask, neg_text_states_2 = encode(negative_prompt)

    # `llama_params`/`clip_params` are `replicated` (a full copy on *every*
    # chip, not TP-sharded like the DiT) -- the 8B Llama tower alone is
    # ~16GB/chip in bf16. Nothing below this line needs them again, but
    # they (and `encode`, which closes over them) stay reachable as local
    # variables for the rest of `main()`'s scope, so without an explicit
    # `del` their device buffers sit resident in HBM through the entire
    # DiT sampling loop that follows -- confirmed to be the dominant cause
    # of this port's real-resolution OOMs (see docs/lessons/
    # hunyuan_video_debugging.md). Freeing them here, before the loop
    # allocates its own activations, trades a few seconds of Python/XLA
    # buffer-deletion overhead for tens of GB of HBM headroom.
    del encode, llama_model, llama_params, llama_tokenizer, clip_model, clip_params, clip_tokenizer
    gc.collect()

    # --- Resolution / frame count ---
    height = align_to(args.height, 16)
    width = align_to(args.width, 16)
    assert (args.num_frames - 1) % 4 == 0, "--num_frames must be `1 + 4k` (VAE temporal compression=4)."

    latent_channels = vae_kwargs["latent_channels"]
    lt = (args.num_frames - 1) // fft + 1
    lh, lw = height // ffs, width // ffs

    # --- Sampling loop ---
    key = jax.random.PRNGKey(args.seed)
    latents = jax.random.normal(key, (1, lt, lh, lw, latent_channels), dtype=jnp.float32).astype(dit_dtype)
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
        text_states, text_mask, text_states_2,
        neg_text_states, neg_text_mask, neg_text_states_2,
        embedded_guidance, guidance_scale,
    ):
        lat_cf = jnp.moveaxis(latents, -1, 1)

        v_cond = dit_model.apply(
            dit_params, lat_cf, timestep, text_states, text_mask, text_states_2, guidance=embedded_guidance)
        v_uncond = dit_model.apply(
            dit_params, lat_cf, timestep, neg_text_states, neg_text_mask, neg_text_states_2, guidance=embedded_guidance)
        v_cond = jnp.moveaxis(v_cond, 1, -1)
        v_uncond = jnp.moveaxis(v_uncond, 1, -1)
        v = (v_uncond + guidance_scale * (v_cond - v_uncond)).astype(jnp.float32)

        new_latents = latents.astype(jnp.float32) - v * dsigma
        return new_latents.astype(latents.dtype)

    if args.offload_dit_weights:
        # `pre_apply`/`mid_process`/`post_apply`/`chunk_forward_double`/
        # `chunk_forward_single` are each their own `jax.jit` (bar
        # `mid_process`, a trivial params-free concat run as plain Python),
        # compiled once and reused for every chunk and every step -- the
        # outer `range(num_chunks)` loops below must stay plain Python, not
        # jax.jit-traced, or every chunk's weights would get unrolled into
        # one HLO program's buffer space, defeating the entire point of
        # offloading (see docs/weight_offloading.md). `mesh=mesh` is not
        # optional on the block reconstructions below -- see
        # generate_wan2_1_t2v.py's identical comment on `attend`'s Pallas
        # flash-attention kernel needing `shard_map`, not a bare `jax.jit`,
        # once any part of the program is multi-device-partitioned.
        # Cond/uncond are batched together (batch=2) through the offloaded
        # path, not run as two separate passes like the non-offloaded
        # `sampling_step` above -- halves the number of host-to-device
        # weight transfers per step, the actual scarce resource here.
        # `img_len`/`tt`/`th`/`tw` can't be returned as-is from inside
        # `jax.jit` -- plain Python ints aren't valid jitted-function
        # *outputs* (unlike *inputs*, where static values are fine); JIT
        # would otherwise silently turn them into traced 0-d arrays, which
        # then fail `post_apply`'s `static_argnums` (a concrete-but-still-
        # ArrayImpl value isn't hashable) -- see generate_wan2_1_t2v.py's
        # identical comment on `input_dtype`/`grid`. So `_pre_process_body`
        # only returns the actual array outputs the block loops need;
        # `sampling_step_offloaded` below recomputes `img_len`/`tt`/`th`/
        # `tw` directly from `lat_cf_2b`'s own (static) shape instead --
        # exact, not an approximation, since `pre_process` computes them
        # the same way from the same shape.
        def _pre_process_body(params, hidden_states, timestep, text_states, text_mask, text_states_2, guidance):
            # `pre_process` now also returns `token_replace_vec`/
            # `first_frame_token_num` (I2V's `token_replace` mode -- see
            # generate_hunyuan_video_i2v.py) -- both `None` here since this
            # script never sets `i2v_condition_type`, discarded same as
            # the other static/unneeded values.
            img, txt, vec, freqs, key_valid, _img_len, _tt, _th, _tw, _tr_vec, _tr_n = dit_model.apply(
                params, hidden_states, timestep, text_states, text_mask, text_states_2,
                guidance=guidance, method=dit_model.pre_process)
            return img, txt, vec, freqs, key_valid

        pre_apply = jax.jit(_pre_process_body)

        def _chunk_forward_double_body(chunk_params, img, txt, vec, freqs, key_valid):
            for layer_params in chunk_params:
                img, txt = MMDoubleStreamBlock(
                    hidden_size=dit_model.hidden_size, heads_num=dit_model.heads_num,
                    mlp_width_ratio=dit_model.mlp_width_ratio, mlp_act_type=dit_model.mlp_act_type,
                    qk_norm=dit_model.qk_norm, qkv_bias=dit_model.qkv_bias, mesh=mesh,
                ).apply({"params": layer_params}, img, txt, vec, freqs, key_valid)
            return img, txt

        chunk_forward_double = jax.jit(_chunk_forward_double_body, donate_argnums=(0,))

        def _chunk_forward_single_body(chunk_params, x, vec, txt_len, freqs, key_valid):
            for layer_params in chunk_params:
                x = MMSingleStreamBlock(
                    hidden_size=dit_model.hidden_size, heads_num=dit_model.heads_num,
                    mlp_width_ratio=dit_model.mlp_width_ratio, mlp_act_type=dit_model.mlp_act_type,
                    qk_norm=dit_model.qk_norm, mesh=mesh,
                ).apply({"params": layer_params}, x, vec, txt_len, freqs, key_valid)
            return x

        # `txt_len` must be a static arg: `MMSingleStreamBlock` uses it for
        # slice bounds (`q[:, :-txt_len]`), which JAX requires to be a
        # concrete Python int, not a traced jitted value -- constant across
        # every call here (one fixed text-token count per run), so this
        # never triggers a retrace.
        chunk_forward_single = jax.jit(_chunk_forward_single_body, static_argnums=(3,), donate_argnums=(0,))

        # `img_len`/`tt`/`th`/`tw` must be `static_argnums`, not ordinary
        # jitted arguments (plain Python ints, not abstractable arrays) --
        # constant across every call in this script (one fixed resolution
        # per run), so this never triggers a retrace.
        post_apply = jax.jit(
            lambda params, x, vec, img_len, tt, th, tw: dit_model.apply(
                params, x, vec, img_len, tt, th, tw, method=dit_model.post_process),
            static_argnums=(3, 4, 5, 6))

        def sampling_step_offloaded(
            nonblock_params, latents, timestep, dsigma,
            text_states, text_mask, text_states_2,
            neg_text_states, neg_text_mask, neg_text_states_2,
            embedded_guidance, guidance_scale,
        ):
            lat_cf = jnp.moveaxis(latents, -1, 1)
            lat_cf_2b = jnp.concatenate([lat_cf, lat_cf], axis=0)
            t_2b = jnp.concatenate([timestep, timestep], axis=0)
            text_states_2b = jnp.concatenate([text_states, neg_text_states], axis=0)
            text_mask_2b = jnp.concatenate([text_mask, neg_text_mask], axis=0)
            text_states_2_2b = jnp.concatenate([text_states_2, neg_text_states_2], axis=0)
            guidance_2b = (
                jnp.concatenate([embedded_guidance, embedded_guidance], axis=0)
                if embedded_guidance is not None else None)

            # Matches `pre_process`'s own internal computation exactly, from
            # the same (static) shape -- see the comment on `_pre_process_body`.
            pt, ph, pw = dit_model.patch_size
            _, _, ot, oh, ow = lat_cf_2b.shape
            tt, th, tw = ot // pt, oh // ph, ow // pw

            img, txt, vec, freqs, key_valid = pre_apply(
                nonblock_params, lat_cf_2b, t_2b, text_states_2b, text_mask_2b, text_states_2_2b, guidance_2b)
            img_len = img.shape[1]
            for chunk_host in double_chunks_host:
                chunk_params = jax.device_put(chunk_host, double_chunk_sharding)
                img, txt = chunk_forward_double(chunk_params, img, txt, vec, freqs, key_valid)

            x, txt_len = HunyuanVideoDiT.mid_process(img, txt)

            for chunk_host in single_chunks_host:
                chunk_params = jax.device_put(chunk_host, single_chunk_sharding)
                x = chunk_forward_single(chunk_params, x, vec, txt_len, freqs, key_valid)

            v_2b = post_apply(nonblock_params, x, vec, img_len, tt, th, tw)
            v_cond, v_uncond = jnp.split(v_2b, 2, axis=0)
            v_cond = jnp.moveaxis(v_cond, 1, -1)
            v_uncond = jnp.moveaxis(v_uncond, 1, -1)
            v = (v_uncond + guidance_scale * (v_cond - v_uncond)).astype(jnp.float32)

            new_latents = latents.astype(jnp.float32) - v * dsigma
            return new_latents.astype(latents.dtype)

    # --- Staged (per-decoder-level) VAE decode -- see module docstring and
    # `hunyuan_video.vae.Decoder.stage_level`'s docstring for why real
    # frame counts need this instead of one fused `Decoder.__call__`. ---
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
            text_states, text_mask, text_states_2,
            neg_text_states, neg_text_mask, neg_text_states_2,
            embedded_guidance, guidance_scale)
        jax.block_until_ready(latents)  # otherwise this only times async dispatch, not the real step.
        logger.info("step %d/%d done (%.1fs)", step + 1, args.num_steps, time.perf_counter() - step_t0)

    # --- VAE decode ---
    latents_for_decode = jax.device_put((latents.astype(dtype) / scaling_factor), replicated)
    pixels = np.array(spatial_tiled_vae_decode(vae_params, latents_for_decode))

    save_video(pixels, args.output_path, fps=args.fps)
    logger.info("Saved %s", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="Root dir containing config.json, hunyuan-video-t2v-720p/{transformers,vae}/ "
                              "(i.e. tencent/HunyuanVideo's downloaded layout).")
    parser.add_argument("--text_encoder_dir", type=str, required=True,
                         help="Path to the extracted Llama text-decoder tower (see "
                              "hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py -- "
                              "xtuner/llava-llama-3-8b-v1_1-transformers's .language_model).")
    parser.add_argument("--clip_checkpoint_dir", type=str, required=True,
                         help="Path to openai/clip-vit-large-patch14's downloaded root.")
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill", choices=list(DIT_CONFIGS.keys()),
                         help="Named hyperparameter preset -- the released checkpoint is the "
                              "cfgdistill (guidance_embed=True) variant.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                         help="Number of devices to Megatron-shard the DiT's attention heads/FFN channels across. "
                              "Must divide num_devices and heads_num (24). Defaults to every local device.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None,
                         help="Defaults to the reference's own NEGATIVE_PROMPT when --guidance_scale != 1.0, else empty.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=129, help="Must be `1 + 4k`.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=7.0, help="config.py's `--flow-shift` default.")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                         help="Real classifier-free guidance scale. Default 1.0 (off) matches the reference's own "
                              "sample_video.py default -- this checkpoint's embedded/distilled guidance "
                              "(--embedded_guidance_scale) is the primary guidance mechanism.")
    parser.add_argument("--embedded_guidance_scale", type=float, default=6.0,
                         help="Embedded/distilled guidance scale fed to `guidance_in` (config.py's "
                              "`--embedded-cfg-scale` default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--vae_tile_latent_size", type=int, default=None,
                         help="Latent-space spatial tile size for the tiled VAE decode. Defaults to the "
                              "reference's own `sample_size // ffactor_spatial`; shrink (e.g. 8) if VAE decode OOMs.")
    parser.add_argument("--offload_dit_weights", action="store_true",
                         help="Keep the double/single-stream blocks' weights host-resident and offload one "
                              "--offload_chunk_size_{double,single}-block group's worth into HBM at a time during "
                              "the sampling loop, instead of the whole DiT tree staying HBM-resident for the "
                              "entire script (same idea as DeepSpeed's ZeRO-Offload / diffusers' "
                              "enable_sequential_cpu_offload, applied per-layer here). The 13B DiT's own compute "
                              "already fits at native 720p/129 frames under --tensor_parallel_size 4 -- what "
                              "doesn't fit is the DiT staying fully HBM-resident *plus* the sampling loop's own "
                              "activations at that scale (only 4 TPU chips means TP can't be pushed further); "
                              "see docs/weight_offloading.md and docs/lessons/hunyuan_video_debugging.md.")
    parser.add_argument("--offload_chunk_size_double", type=int, default=None,
                         help="Number of consecutive double-stream blocks grouped into one offloaded HBM buffer / "
                              "one jax.jit compile when --offload_dit_weights is set (ignored otherwise). Must "
                              "divide mm_double_blocks_depth (20). Defaults to 20 (all double-stream blocks in "
                              "one chunk -- confirmed to fit at the real 129-frame/720p default; the single/double "
                              "streams' chunk sizes are independent, unlike the shared --offload_chunk_size this "
                              "replaces). Shrink for a tighter HBM budget, at the cost of more separate "
                              "host-to-device transfers and jax.jit dispatches per step.")
    parser.add_argument("--offload_chunk_size_single", type=int, default=None,
                         help="Same as --offload_chunk_size_double, for the single-stream blocks. Must divide "
                              "mm_single_blocks_depth (40). Defaults to 40 (all single-stream blocks in one chunk).")
    args = parser.parse_args()
    main(args)
