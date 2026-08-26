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
    DIFFUSION_VAE_CONFIG, VAE_CONFIG, connector_kwargs_from_transformer_config,
    dit_kwargs_from_transformer_config, gemma4_text_model_kwargs, load_gemma4_config, load_ltx2_5_metadata,
    vae_scale_factors,
)
from vidax.models.ltx2_5.connector import Embeddings1DConnector
from vidax.models.ltx2_5.diffusion_vae import DiffusionVideoDecoder, crop_temporal_pad
from vidax.models.ltx2_5.dit import LTXDiT, LTXDiTBlock
from vidax.models.ltx2_5.gemma4 import Gemma4Tokenizer, Gemma4TextModel, extract_video_features
from vidax.models.ltx2_5.patchifier import get_latent_coord_bounds, latent_to_pixel_coord_bounds, patchify, unpatchify
from vidax.models.ltx2_5.vae import Encoder, LTXVAE
from vidax.schedulers.ltx2_5_ancestral_euler import DEV_CFG_GUIDANCE_SCALE, DEV_CFG_RESCALE_SCALE, AncestralEulerScheduler
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


# The real `ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors`
# checkpoints ship every `scale_shift_table`/`prompt_scale_shift_table`
# (the AdaLN modulation tables driving every block's self-attn/cross-attn/
# FFN scale-shift-gate, and cross-attention-AdaLN's key/value modulation)
# in **float32**, not bf16, unlike the rest of the DiT's weights -- the
# checkpoint's own file metadata confirms this deliberately (290 float32
# tensors out of 4349 total, every one of them one of these two names).
# A blanket `cast_to_dtype(dit_params, dit_dtype)` silently throws this
# away, downcasting these small (9x4096 / 2x4096 per block, negligible
# memory either way) but numerically load-bearing tables to bf16 along
# with everything else -- these values directly scale/shift/gate the
# residual stream at every one of 48 blocks, so bf16 rounding here
# compounds across the whole depth of the network in a way it wouldn't
# for an ordinary large matmul weight. Kept at float32 regardless of
# `--dit_dtype` (still promoted further to float32 explicitly inside
# `LTXDiTBlock`/`LTXDiT`'s own AdaLN math either way -- this only fixes
# the *storage* dtype these promotions start from).
_DIT_FLOAT32_LEAF_NAMES = frozenset({"scale_shift_table", "prompt_scale_shift_table"})


def cast_dit_params(tree, dtype):
    def cast_leaf(path, x):
        leaf_name = path[-1].key if path and hasattr(path[-1], "key") else None
        if leaf_name in _DIT_FLOAT32_LEAF_NAMES:
            return x.astype(jnp.float32) if x.dtype != jnp.float32 else x
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map_with_path(cast_leaf, tree)


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
    video_feats = extract_video_features(
        hidden_states, mask, video_kernel, video_bias, embedding_dim=gemma_model.hidden_size).astype(dtype)

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

    dit_model = LTXDiT(**dit_kwargs_from_transformer_config(dit_metadata), compute_dtype=dit_dtype, mesh=mesh)
    connector_model = Embeddings1DConnector(
        **connector_kwargs_from_transformer_config(dit_metadata), compute_dtype=dtype)

    # `vidax.models.ltx2_5.vae.Encoder` is shared between both VAE variants
    # unchanged (confirmed from both checkpoints' embedded `config.vae.
    # encoder` -- see vidax.models.ltx2_5.diffusion_vae's module docstring),
    # so `encoder_model`/its kwargs come from the same `VAE_CONFIG` either
    # way; only the decoder differs.
    encoder_kwargs = dict(
        in_channels=3, base_channels=VAE_CONFIG["base_channels"], encoder_blocks=VAE_CONFIG["encoder_blocks"],
        patch_size=VAE_CONFIG["patch_size"], latent_channels=VAE_CONFIG["latent_channels"])
    temporal_scale, spatial_scale = vae_scale_factors(VAE_CONFIG)
    if args.vae_variant == "conv":
        vae_kwargs = dict(
            latent_channels=VAE_CONFIG["latent_channels"], encoder_blocks=VAE_CONFIG["encoder_blocks"],
            decoder_blocks=VAE_CONFIG["decoder_blocks"], patch_size=VAE_CONFIG["patch_size"],
            base_channels=VAE_CONFIG["base_channels"], causal_decoder=VAE_CONFIG["causal_decoder"],
            timestep_conditioning=VAE_CONFIG["timestep_conditioning"])
        vae_model = LTXVAE(**vae_kwargs)
        encoder_model = vae_model
    else:
        # `DiffusionVideoDecoder` has no `encode` -- `vidax.models.ltx2_5.
        # vae.Encoder` is shared/unchanged between both VAE variants (see
        # its docstring), applied separately here to the same checkpoint's
        # `params["encoder"]` subtree (`map_ltx2_5_diffusion_decoder_keys`
        # produces it in that shape -- see that mapper's docstring).
        vae_model = DiffusionVideoDecoder(**DIFFUSION_VAE_CONFIG)
        encoder_model = Encoder(**encoder_kwargs)

    gemma_config = load_gemma4_config(args.text_encoder_checkpoint_path)
    gemma_model = Gemma4TextModel(**gemma4_text_model_kwargs(gemma_config), compute_dtype=dtype)
    tokenizer = Gemma4Tokenizer(args.text_encoder_checkpoint_path, seq_len=args.text_max_tokens)

    assert dit_model.num_attention_heads % tp_size == 0, (
        f"LTXDiT.num_attention_heads ({dit_model.num_attention_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    assert gemma_model.num_attention_heads % tp_size == 0, (
        f"Gemma4TextModel.num_attention_heads ({gemma_model.num_attention_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")
    if args.vae_variant == "diffusion":
        for c in (*vae_model.stage_channels, vae_model.stage5_channels or vae_model.stage_channels[-1]):
            assert (c // vae_model.head_dim) % tp_size == 0, (
                f"DiffusionVideoDecoder: every stage's num_heads (channels // head_dim) must be divisible by "
                f"--tensor_parallel_size ({tp_size}) for Megatron-TP -- got {c} // {vae_model.head_dim} "
                f"= {c // vae_model.head_dim} heads.")

    # --- Load weights ---
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="ltx2_5_dit")
    connector_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="ltx2_5_connector")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(
        args.vae_checkpoint_path,
        model_type="ltx2_5_vae" if args.vae_variant == "conv" else "ltx2_5_diffusion_decoder")
    logging.info(f"Loading Gemma-4 weights from {args.text_encoder_checkpoint_path}...")
    gemma_params = load_torch_checkpoint_to_jax(args.text_encoder_checkpoint_path, model_type="gemma4_text")
    import safetensors.numpy
    gemma_sd = safetensors.numpy.load_file(args.text_encoder_checkpoint_path)
    video_kernel, video_bias = load_gemma4_video_aggregate_embed(gemma_sd)
    video_kernel = jnp.asarray(video_kernel, dtype=dtype)
    video_bias = jnp.asarray(video_bias, dtype=dtype)
    del gemma_sd

    replicated = get_replicated_sharding(mesh)
    dit_params = cast_dit_params(dit_params, dit_dtype)
    connector_params = cast_to_dtype(connector_params, dtype)
    gemma_params = cast_to_dtype(gemma_params, dtype)
    dit_shardings = shard_wan_params(dit_params, mesh)
    # `--offload_dit_weights`: keep the DiT's per-block ("blocks_*") params
    # host-resident and stream `--offload_chunk_size` consecutive blocks'
    # worth into HBM at a time during the sampling loop instead of
    # `device_put`-ing the whole 48-block tree at once -- see
    # docs/weight_offloading.md and this model's own entry there. Unlike
    # every other model this technique has been applied to in this repo,
    # LTX-2.5's DiT weights were never the actual HBM bottleneck (~6.6GB/chip
    # at tp=4, comfortably resident) -- what this buys here is instead
    # closing the JIT compilation boundary once per chunk: with the whole
    # 48-block loop traced into one fused program, the per-block AdaLN/
    # attention/FFN intermediates were measured to *not* get freed across
    # blocks (temp memory scaled almost linearly with block count, ~1.8GB per
    # extra block), so the fused program's peak activation memory grew far
    # past this chip's budget even though no single block's own compute
    # needs more than a few GB. Chunking closes that gap by construction:
    # each `chunk_forward` call is its own separately-compiled program, so
    # only one chunk's temporaries are ever live at once, regardless of
    # `num_layers`. See docs/lessons/ltx2_5_debugging.md.
    if args.offload_dit_weights:
        chunk_size = args.offload_chunk_size
        assert dit_model.num_layers % chunk_size == 0, (
            f"--offload_chunk_size ({chunk_size}) must divide LTXDiT.num_layers "
            f"({dit_model.num_layers}) -- see docs/weight_offloading.md.")
        num_layers = dit_model.num_layers
        chunk_params_host = [
            [dit_params["params"][f"blocks_{i}"] for i in range(c, c + chunk_size)]
            for c in range(0, num_layers, chunk_size)
        ]
        layer_sharding = dit_shardings["params"]["blocks_0"]
        chunk_sharding = [layer_sharding] * chunk_size
        nonblock_params = {k: v for k, v in dit_params["params"].items() if not k.startswith("blocks_")}
        nonblock_shardings = {k: v for k, v in dit_shardings["params"].items() if not k.startswith("blocks_")}
        dit_params = jax.device_put({"params": nonblock_params}, {"params": nonblock_shardings})
    else:
        dit_params = jax.device_put(dit_params, dit_shardings)
    connector_params = jax.device_put(connector_params, shard_wan_params(connector_params, mesh))
    gemma_params = jax.device_put(gemma_params, shard_wan_params(gemma_params, mesh))
    vae_params = cast_to_dtype(vae_params, dtype)
    if args.vae_variant == "diffusion":
        # Megatron-TP the diffusion decoder's attention heads (`to_q`/
        # `to_k`/`to_v`/`to_out`) and SwiGLU MLPs (`w_gate`/`w_up`/`w_down`)
        # across the same mesh already used for the DiT/Gemma-4 -- needed
        # for real, not just consistency: a fully-replicated (single-chip)
        # stage-5 block alone needed 32.53GB of HLO temporaries (30.75GB
        # available on one v4 chip), even after every other memory fix (see
        # docs/lessons/ltx2_5_debugging.md). `Encoder`'s conv submodule
        # names (`conv1`/`conv2`/`conv_in`/`conv_out`) aren't in
        # `shard_wan_params`'s name tables, so its params fall back to
        # (correct, unchanged) full replication automatically.
        vae_params = jax.device_put(vae_params, shard_wan_params(vae_params, mesh))
    else:
        vae_params = jax.device_put(vae_params, replicated)
    encoder_params = (
        vae_params if args.vae_variant == "conv"
        else {"params": vae_params["params"]["encoder"]})
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
        eta=args.eta)
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None
        else (1.0 if args.sampler == "distilled" else DEV_CFG_GUIDANCE_SCALE))
    guidance_rescale = (
        args.guidance_rescale if args.guidance_rescale is not None
        else (0.0 if args.sampler == "distilled" else DEV_CFG_RESCALE_SCALE))
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
        latent_coord_bounds, temporal_scale, spatial_scale, causal_fix=causal_fix, fps=float(args.fps))

    # --- I2V conditioning: VAE-encode the image into the first latent frame,
    # `clean_latent`/`denoise_mask` grids (all-ones mask = plain T2V, see
    # file docstring) ---
    denoise_mask_grid = jnp.ones((batch_size, latent_f, latent_h, latent_w, 1), dtype=jnp.float32)
    clean_latent = jnp.zeros(latent_shape, dtype=jnp.float32)
    if args.image_path is not None:
        logging.info(f"I2V: encoding conditioning image {args.image_path}")
        image = load_conditioning_image(args.image_path, args.height, args.width)
        image = np.broadcast_to(image, (batch_size,) + image.shape[1:])
        encode_method = encoder_model.encode if args.vae_variant == "conv" else encoder_model.__call__
        cond_latent = encoder_model.apply(encoder_params, jnp.asarray(image, dtype=dtype), method=encode_method)
        clean_latent = clean_latent.at[:, :1].set(cond_latent.astype(jnp.float32))
        strength = args.conditioning_strength
        denoise_mask_grid = denoise_mask_grid.at[:, :1].set(1.0 - strength)

    denoise_mask = patchify(denoise_mask_grid)  # (B, N, 1)
    clean_tokens = patchify(clean_latent)  # (B, N, C)

    noise_rng, rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, clean_tokens.shape, dtype=jnp.float32)
    init_tokens = denoise_mask * noise + (1.0 - denoise_mask) * clean_tokens
    tokens = init_tokens.astype(dit_dtype)

    if args.offload_dit_weights:
        # `pre_apply`/`post_apply`/`chunk_forward` are each their own
        # `jax.jit`, compiled once and reused for every chunk and every step
        # (identical shape/dtype/sharding signature throughout). Neither
        # `chunk_forward`'s cross-chunk Python loop nor the whole per-step
        # computation may be wrapped in an outer `jax.jit`: doing so would
        # trace/unroll every chunk into one HLO program again, defeating the
        # entire point (see the comment above `if args.offload_dit_weights`
        # near the weight-loading code, and docs/hardware_and_sharding.md's
        # JIT Compilation Safety section).
        def _pre_process_body(params, tokens, coords, timestep, sigma, context):
            x, freqs, ctx, bias, timestep_mod, prompt_timestep, embedded_timestep, _input_dtype = dit_model.apply(
                params, tokens, coords, timestep, sigma, context, method=dit_model.pre_process)
            return x, freqs, ctx, bias, timestep_mod, prompt_timestep, embedded_timestep

        pre_apply = jax.jit(_pre_process_body)
        post_apply = jax.jit(
            lambda params, x, embedded_timestep, input_dtype: dit_model.apply(
                params, x, embedded_timestep, input_dtype, method=dit_model.post_process),
            static_argnums=(3,))

        inner_dim = dit_model.num_attention_heads * dit_model.attention_head_dim

        def _chunk_forward_body(chunk_params, x, freqs, ctx, bias, timestep_mod, prompt_timestep):
            for layer_params in chunk_params:
                x = LTXDiTBlock(
                    dim=inner_dim, num_heads=dit_model.num_attention_heads, head_dim=dit_model.attention_head_dim,
                    ff_inner_dim=inner_dim * 4, cross_attention_dim=dit_model.cross_attention_dim,
                    eps=dit_model.eps, ff_bias=dit_model.ff_bias,
                    cross_attention_adaln=dit_model.cross_attention_adaln,
                    apply_gated_attention=dit_model.apply_gated_attention,
                    compute_dtype=dit_dtype, mesh=mesh,
                ).apply({"params": layer_params}, x, freqs, ctx, bias, timestep_mod, prompt_timestep)
            return x

        chunk_forward = jax.jit(_chunk_forward_body, donate_argnums=(0,))

        def dit_apply_offloaded(tokens, coords, timestep, sigma, context):
            x, freqs, ctx, bias, timestep_mod, prompt_timestep, embedded_timestep = pre_apply(
                dit_params, tokens, coords, timestep, sigma, context)
            for chunk_host in chunk_params_host:
                chunk_params = jax.device_put(chunk_host, chunk_sharding)
                x = chunk_forward(chunk_params, x, freqs, ctx, bias, timestep_mod, prompt_timestep)
            return post_apply(dit_params, x, embedded_timestep, tokens.dtype)

        def single_step_offloaded(current_tokens, step_index, context, negative_context, pixel_coord_bounds,
                                   denoise_mask, clean_tokens, noise_rng, guidance_scale, guidance_rescale, use_cfg):
            b = current_tokens.shape[0]
            sigma_val = scheduler.sigmas[step_index]
            timestep = denoise_mask[..., 0] * sigma_val

            if use_cfg:
                tokens_2b = jnp.concatenate([current_tokens, current_tokens], axis=0)
                timestep_2b = jnp.concatenate([timestep, timestep], axis=0)
                sigma_2b = jnp.full((2 * b,), sigma_val, dtype=jnp.float32)
                coords_2b = jnp.concatenate([pixel_coord_bounds, pixel_coord_bounds], axis=0)
                context_2b = jnp.concatenate([context, negative_context], axis=0)
                v_2b = dit_apply_offloaded(tokens_2b, coords_2b, timestep_2b, sigma_2b, context_2b)
                v_cond, v_uncond = v_2b[:b].astype(jnp.float32), v_2b[b:].astype(jnp.float32)
                velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
                # Guidance rescale (`ltx_core.components.guiders.MultiModalGuider
                # .calculate`): corrects CFG over-saturation. A true no-op at
                # `guidance_rescale=0.0` (factor=1), so applied unconditionally
                # -- avoids branching on a traced value under `jax.jit`.
                rescale_factor = guidance_rescale * (v_cond.std() / velocity.std()) + (1.0 - guidance_rescale)
                velocity = velocity * rescale_factor
            else:
                sigma = jnp.full((b,), sigma_val, dtype=jnp.float32)
                velocity = dit_apply_offloaded(current_tokens, pixel_coord_bounds, timestep, sigma, context)

            denoised = current_tokens.astype(jnp.float32) - velocity.astype(jnp.float32) * sigma_val
            denoised = denoise_mask * denoised + (1.0 - denoise_mask) * clean_tokens

            noise = jax.random.normal(noise_rng, current_tokens.shape, dtype=jnp.float32)
            next_tokens = scheduler.step(denoised, current_tokens, step_index, noise)
            next_tokens = denoise_mask * next_tokens + (1.0 - denoise_mask) * clean_tokens
            return next_tokens.astype(current_tokens.dtype)
    else:
        dit_apply = jax.jit(lambda params, tokens, coords, timestep, sigma, context: dit_model.apply(
            params, tokens, coords, timestep, sigma, context))

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("use_cfg",))
        def single_step(current_tokens, step_index, context, negative_context, pixel_coord_bounds,
                         denoise_mask, clean_tokens, params, noise_rng, guidance_scale, guidance_rescale, use_cfg):
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
                v_cond, v_uncond = v_2b[:b].astype(jnp.float32), v_2b[b:].astype(jnp.float32)
                velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
                # Guidance rescale (`ltx_core.components.guiders.MultiModalGuider
                # .calculate`): corrects CFG over-saturation. A true no-op at
                # `guidance_rescale=0.0` (factor=1), so applied unconditionally
                # -- avoids branching on a traced value under `jax.jit`.
                rescale_factor = guidance_rescale * (v_cond.std() / velocity.std()) + (1.0 - guidance_rescale)
                velocity = velocity * rescale_factor
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
        f"guidance_scale={guidance_scale}, guidance_rescale={guidance_rescale})...")
    for step_index in range(scheduler.num_steps):
        step_rng, rng = jax.random.split(rng)
        if args.offload_dit_weights:
            tokens = single_step_offloaded(
                tokens, step_index, context, (negative_context if use_cfg else context), pixel_coord_bounds,
                denoise_mask, clean_tokens, step_rng, guidance_scale, guidance_rescale, use_cfg)
        else:
            tokens = single_step(
                tokens, step_index, context, (negative_context if use_cfg else context), pixel_coord_bounds,
                denoise_mask, clean_tokens, dit_params, step_rng, guidance_scale, guidance_rescale, use_cfg)

    # --- Decode latents to video frames ---
    latents = unpatchify(tokens, latent_f, latent_h, latent_w).astype(dtype)
    # Drop every DiT/Gemma/connector-side reference (weights, and every
    # closure that captured them: `single_step`/`dit_apply` or
    # `single_step_offloaded`/`pre_apply`/`post_apply`/`chunk_forward`/
    # `chunk_params_host`) before decoding -- CPython's refcounting then
    # frees their HBM immediately, rather than leaving it resident
    # alongside VAE decode's own activation memory for the rest of the
    # script. Same fix as Wan2.1's native-720P OOM (see
    # docs/weight_offloading.md's "Wan2.1: fixing two real OOMs" section):
    # confirmed necessary here too -- decoding a larger-than-reference
    # frame count OOM'd in exactly this call, at only ~3.5GB short, before
    # this fix freed the DiT/Gemma/connector residency that had nothing
    # left to do by this point in the script.
    if args.offload_dit_weights:
        del pre_apply, post_apply, chunk_forward, dit_apply_offloaded, single_step_offloaded, chunk_params_host
    else:
        del dit_apply, single_step
    del dit_params, connector_params, gemma_params
    logging.info("Decoding final latents into video frames...")
    decode_rng, rng = jax.random.split(rng)
    if args.vae_variant == "conv":
        decode_fn = jax.jit(lambda params, z: vae_model.apply(params, z, method=vae_model.decode))
        frames = decode_fn(vae_params, latents)
    else:
        # Two separate `jax.jit` calls, not one `decode_fn` wrapping the
        # whole thing -- `DiffusionVideoDecoder`'s `context`/`diffuse` split
        # exists specifically so each stage's own temporaries are freed
        # before the next begins; re-fusing these into one jitted call here
        # would silently reintroduce a real OOM. See `DiffusionVideoDecoder`'s
        # own class docstring and docs/lessons/ltx2_5_debugging.md.
        context_fn = jax.jit(lambda params, z: vae_model.apply(params, z, method=vae_model.context))
        context, t_pad = context_fn(vae_params, latents)
        # `t_pad` comes back as a 0-d traced array (jax.jit converts every
        # returned Python scalar), but `diffuse`'s `t_pad` arg controls
        # Python-level control flow (`if t_pad: ...`), so it must be a real
        # Python int again before being passed back in as a static arg.
        t_pad = int(t_pad)
        noise_shape = (
            context.shape[0], context.shape[1], context.shape[2] * vae_model.patch_size,
            context.shape[3] * vae_model.patch_size, vae_model.out_channels)
        x_t = jax.random.normal(decode_rng, noise_shape, dtype=context.dtype)
        # `--vae_variant diffusion` only wires up the checkpoint's own real
        # single-step `x0` recipe here (see `DiffusionVideoDecoder.diffuse`'s
        # multi-step Euler fallback for the general case, not memory-split
        # below) -- one `jax.jit` per stage-5 block (`block_idx` static) so
        # each block's own temporaries are freed before the next begins,
        # same fix shape as the DiT's own --offload_dit_weights. See
        # `DiffusionVideoDecoder`'s class docstring for why this granularity.
        assert vae_model.default_num_inference_steps == 1 and vae_model.model_output_type == "x0", (
            "examples/generate_ltx2_5.py's --vae_variant diffusion decode path only supports the checkpoint's "
            "own real single-step x0 recipe.")
        prepare_fn = jax.jit(
            lambda params, context, x_t, t_now: vae_model.apply(
                params, context, x_t, t_now, method=vae_model.diffuse_prepare),
            static_argnums=3)
        step_fn = jax.jit(
            lambda params, context, x_half, modulation, block_idx: vae_model.apply(
                params, context, x_half, modulation, block_idx, method=vae_model.diffuse_step),
            static_argnums=4)
        finalize_fn = jax.jit(
            lambda params, x_half: vae_model.apply(params, x_half, method=vae_model.diffuse_finalize))

        x_half, modulation = prepare_fn(vae_params, context, x_t, 1.0)
        for block_idx in range(vae_model.stage_depths[-1]):
            x_half = step_fn(vae_params, context, x_half, modulation, block_idx)
        frames = finalize_fn(vae_params, x_half)
        frames = crop_temporal_pad(frames, t_pad, vae_model.upsamples)

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
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to ltx-2.5-video-vae-conv-bf16.safetensors (--vae_variant conv) or ltx-2.5-video-vae-bf16.safetensors (--vae_variant diffusion).")
    parser.add_argument("--vae_variant", type=str, default="conv", choices=["conv", "diffusion"], help="VAE decoder architecture: 'conv' (default, ResNet + pixel-shuffle-upsample) or 'diffusion' (vidax.models.ltx2_5.diffusion_vae.DiffusionVideoDecoder, NATTEN-based, matches the official demos' decoder). Both have a real, checkpoint-inherent periodic artifact, 'diffusion' about half as severe -- single full-volume tile only so far. See docs/lessons/ltx2_5_debugging.md.")
    parser.add_argument("--text_encoder_checkpoint_path", type=str, required=True, help="Path to gemma4-12b-with-proj-ltx-2.5-bf16.safetensors (bundles the Gemma-4 text tower, its embedded tokenizer, and the video_aggregate_embed feature projection).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of devices to Megatron-shard the DiT/connector/Gemma-4's attention heads and FFN channels across. Required at any value >1 device -- neither the 22B DiT nor the 12B Gemma encoder's bf16 weights fit replicated on a single TPU v4 chip.")
    parser.add_argument("--offload_dit_weights", action="store_true", help="Keep the DiT's per-block weights host-resident and stream one --offload_chunk_size-block group's worth into HBM at a time during the sampling loop, instead of the whole 48-block tree being resident and traced into one fused forward pass. Needed at higher resolutions/frame counts: the DiT's own weights already fit resident (~6.6GB/chip at tp=4), but the fused forward pass's per-block activations were measured not to be freed across blocks, so peak activation memory grows with num_layers regardless of resolution -- see docs/weight_offloading.md and docs/lessons/ltx2_5_debugging.md.")
    parser.add_argument("--offload_chunk_size", type=int, default=1, help="Number of consecutive DiT blocks grouped into one offloaded HBM buffer / one jax.jit compile when --offload_dit_weights is set (ignored otherwise). Must divide the DiT's num_layers (48 for the 22B checkpoints).")
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
    parser.add_argument("--guidance_rescale", type=float, default=None, help="CFG guidance-rescale strength (corrects over-saturation at high --guidance_scale by rescaling toward the conditioned-only prediction's std): factor = guidance_rescale * (cond.std()/pred.std()) + (1 - guidance_rescale). 0.0 disables (a true no-op). Only applied when CFG is active. Defaults to --sampler's own real value (0.0 for distilled, which uses no CFG at all; 0.7 for dev).")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance (dev checkpoint / guidance_scale != 1.0 only).")
    parser.add_argument("--height", type=int, default=704, help="Output height in pixels.")
    parser.add_argument("--width", type=int, default=1216, help="Output width in pixels.")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of output frames (1 + 8*k).")
    parser.add_argument("--fps", type=int, default=24, help="Output video frame rate.")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path.")
    main(parser.parse_args())
