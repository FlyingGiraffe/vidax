# Text-to-video / image-to-video inference script for Cosmos3 (Nano or Edge)
# on TPU.
#
# Cosmos3 is architecturally unrelated to Wan/Cosmos-Predict2.5: a
# Mixture-of-Transformers combining a causal "understanding" (text) pathway
# with a full-attention "generation" (diffusion) pathway inside one shared
# transformer, no AdaLN modulation anywhere (the timestep is injected once,
# additively, directly into the noisy vision tokens). See
# `vidax.models.cosmos3.dit`'s module docstring for the full architecture
# summary, and `vidax.models.cosmos3.configs` for the Nano/Edge presets.
#
# Ported against a fixed-shape `(B, seq_len, hidden)` packed sequence instead
# of the reference's ragged flat-buffer-with-global-indices design (ragged
# batching only matters for the reference's flexible multi-item *training*
# setup) -- text is padded to `--max_text_len` tokens with an explicit
# validity mask (`und_valid_mask`) so `gen`'s cross-attention over the text
# segment excludes padding positions; the vision (patchified video latent)
# segment is always exactly `T*Hp*Wp` tokens, no padding needed there.
#
# The raw transformer output *is* the velocity prediction (`v = noise - x0`),
# fed straight into the scheduler's `x0 = sample - sigma_t * model_output` --
# no EDM-style `c_in`/`c_skip`/`c_out` preconditioning wrapper, matching
# `refs/diffusers-cosmos3/pipeline_cosmos3_omni.py`'s own denoising loop.

import argparse
import json
import logging
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vidax.core.sharding import build_tpu_mesh, get_replicated_sharding, shard_wan_params
from vidax.models.cosmos3.configs import EDGE_CONFIG, NANO_CONFIG
from vidax.models.cosmos3.mrope import get_mrope_ids_text_tokens, get_mrope_ids_vision_tokens
from vidax.models.cosmos3.dit import Cosmos3Transformer
from vidax.models.wan.wan2_2.vae import (
    Decoder3d, Encoder3d, PATCH_SIZE, WanVAEDecoder, WanVAEEncoder,
    _count_causal_convs, _count_causal_convs_encoder, unpatchify,
)
from vidax.schedulers.unipc import FlowUniPCMultistepScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
MODEL_SIZE_CONFIGS = {"nano": NANO_CONFIG, "edge": EDGE_CONFIG}
BASE_FPS = 24.0
TEMPORAL_COMPRESSION_FACTOR = 4  # Wan2.2 VAE's latent temporal downsample.
TEMPORAL_MARGIN = 15000  # `unified_3d_mrope_temporal_modality_margin`, separates text/vision mRoPE ranges.

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
_RESOLUTION_TEMPLATE_VIDEO = "This video is of {height}x{width} resolution."
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_INVERSE_RESOLUTION_TEMPLATE_VIDEO = "This video is not of {height}x{width} resolution."
_INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _load_json_prompt_asset(filename: str) -> str:
    """Loads a JSON-structured prompt asset and re-serializes it compactly
    (matching `refs/cosmos-main`'s own `compact_json_file` helper) -- these
    JSON-structured prompts are the checkpoint's *documented* recipe
    ("For optimal quality, prompts should be upsampled into a specific JSON
    structure", `Cosmos3-Edge/README.md`), not just one option among many. A
    short plain-text prompt/negative-prompt (this file's own previous
    default) is measurably worse, especially for Edge -- see
    docs/models/cosmos3.md#prompting.
    """
    with open(os.path.join(_ASSETS_DIR, filename)) as f:
        return json.dumps(json.load(f), ensure_ascii=True, separators=(",", ":"))


DEFAULT_NEGATIVE_PROMPT = _load_json_prompt_asset("cosmos3_t2v_negative_prompt.json")
# JSON-upsampled version of `benchmarks/common.py`'s plain-text
# `STANDARD_T2V_PROMPT` -- the exact same red panda/bamboo scene every other
# model's benchmark uses (for cross-model comparability), just re-described
# in the JSON structure Cosmos3 is documented to need, schema-matched to
# `refs/cosmos-main`'s own JSON prompt assets. Used by
# `benchmarks/run_cosmos3.py` in place of the shared plain-text prompt
# (changing `STANDARD_T2V_PROMPT` itself would require re-benchmarking every
# other model family, not just Cosmos3).
EXAMPLE_T2V_PROMPT = _load_json_prompt_asset("cosmos3_t2v_prompt.json")


def save_video(frames: np.ndarray, output_path: str, fps: int = 24):
    logging.info(f"Saving {frames.shape[0]} frames to {output_path}...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info("Video saved successfully.")


def cast_to_dtype(tree, dtype):
    def cast_leaf(x):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def tokenize_prompt(tokenizer, prompt: str, height: int, width: int, num_frames: int, fps: float,
                     max_text_len: int, is_negative: bool = False,
                     add_duration_template: bool = True, add_resolution_template: bool = True):
    """Matches the reference's `Cosmos3OmniPipeline.tokenize_prompt`: system
    prompt + resolution/duration metadata templates, chat-templated, with
    `<|vision_start|>` + eos appended -- then padded to `max_text_len` with
    the tokenizer's own pad token, returning `(ids, valid_mask)`.

    `add_duration_template`/`add_resolution_template` match the reference
    pipeline's own `__call__` kwargs of the same name (both default `True`
    there too) -- every real usage example in `refs/cosmos-main` (Nano,
    Super, and Edge alike) explicitly passes both as `False`.
    """
    text = prompt.rstrip(".")
    if add_duration_template:
        duration_template = _INVERSE_DURATION_TEMPLATE if is_negative else _DURATION_TEMPLATE
        text = f"{text}. {duration_template.format(duration=num_frames / fps, fps=fps)}" if text else \
            duration_template.format(duration=num_frames / fps, fps=fps)
    if add_resolution_template:
        resolution_template = _INVERSE_RESOLUTION_TEMPLATE_VIDEO if is_negative else _RESOLUTION_TEMPLATE_VIDEO
        text = f"{text}. {resolution_template.format(height=height, width=width)}" if text else \
            resolution_template.format(height=height, width=width)

    conversations = [
        {"role": "system", "content": _SYSTEM_PROMPT_VIDEO},
        {"role": "user", "content": text},
    ]
    encodings = tokenizer.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=True)
    ids = list(encodings["input_ids"])
    start_of_generation = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ids = ids + [tokenizer.eos_token_id, start_of_generation]

    assert len(ids) <= max_text_len, (
        f"Prompt tokenized to {len(ids)} tokens, exceeding --max_text_len={max_text_len}; "
        f"pass a longer --max_text_len or a shorter prompt.")
    valid_len = len(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = ids + [pad_id] * (max_text_len - valid_len)
    valid_mask = np.array([True] * valid_len + [False] * (max_text_len - valid_len))
    return np.array(ids, dtype=np.int32), valid_mask


def main(args):
    devices = jax.devices()
    tp_size = args.tensor_parallel_size
    dp_size = len(devices) // tp_size
    mesh = build_tpu_mesh(data_parallel_size=dp_size, tensor_parallel_size=tp_size)
    replicated = get_replicated_sharding(mesh)
    dtype = DTYPES[args.dtype]

    logging.info(f"Using {len(devices)} devices: {dp_size}-way data // {tp_size}-way tensor parallel.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dit_model = Cosmos3Transformer(mesh=mesh, **MODEL_SIZE_CONFIGS[args.model_size])
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder() if args.image_path else None
    scheduler = FlowUniPCMultistepScheduler(
        num_steps=args.num_steps, num_train_timesteps=1000,
        use_karras_sigmas=args.use_karras_sigmas, shift=args.shift,
        karras_sigma_min=args.karras_sigma_min, karras_sigma_max=args.karras_sigma_max)

    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="cosmos3_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(args.vae_checkpoint_path, model_type="wan2.2_vae_diffusers")

    dit_params = cast_to_dtype(dit_params, dtype)
    if tp_size > 1:
        dit_params = jax.device_put(dit_params, shard_wan_params(dit_params, mesh))
    else:
        dit_params = jax.device_put(dit_params, replicated)
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    b = dp_size
    latent_h, latent_w = args.height // 16, args.width // 16
    latent_t = 1 + (args.num_frames - 1) // TEMPORAL_COMPRESSION_FACTOR
    patch_h, patch_w = latent_h // 2, latent_w // 2

    # --- Tokenize prompt (cond + uncond), pad to a fixed length ---
    cond_ids, cond_valid = tokenize_prompt(
        tokenizer, args.prompt, args.height, args.width, args.num_frames, args.fps, args.max_text_len,
        add_duration_template=args.add_duration_template, add_resolution_template=args.add_resolution_template)
    uncond_ids, uncond_valid = tokenize_prompt(
        tokenizer, args.negative_prompt, args.height, args.width, args.num_frames, args.fps,
        args.max_text_len, is_negative=True,
        add_duration_template=args.add_duration_template, add_resolution_template=args.add_resolution_template)
    und_len = args.max_text_len
    cond_input_ids = jnp.asarray(np.broadcast_to(cond_ids, (b, und_len)))
    uncond_input_ids = jnp.asarray(np.broadcast_to(uncond_ids, (b, und_len)))
    cond_valid_mask = jnp.asarray(np.broadcast_to(cond_valid, (b, und_len)))
    uncond_valid_mask = jnp.asarray(np.broadcast_to(uncond_valid, (b, und_len)))

    text_position_ids = get_mrope_ids_text_tokens(und_len, temporal_offset=0.0)
    text_position_ids = jnp.broadcast_to(text_position_ids[:, None, :], (3, b, und_len))

    def _vision_position_ids_for(valid_mask: np.ndarray) -> jnp.ndarray:
        # Offset from each prompt's real (unpadded) token count, matching the
        # reference's ragged, unpadded design (pipeline_cosmos3_omni.py's
        # `_prepare_text_segment`: `und_len = len(input_ids)`) -- using the
        # padded `--max_text_len` here instead would inflate the text<->vision
        # RoPE gap by `max_text_len - real_length`, differently per cond/uncond
        # pass whenever their real lengths differ.
        real_len = float(valid_mask.sum())
        vision_temporal_offset = real_len + TEMPORAL_MARGIN
        vis_pos = get_mrope_ids_vision_tokens(
            latent_t, patch_h, patch_w, temporal_offset=vision_temporal_offset,
            fps=args.fps, base_fps=BASE_FPS, temporal_compression_factor=TEMPORAL_COMPRESSION_FACTOR)
        return jnp.broadcast_to(vis_pos[:, None, :], (3, b, latent_t * patch_h * patch_w))

    cond_vision_position_ids = _vision_position_ids_for(cond_valid)
    uncond_vision_position_ids = _vision_position_ids_for(uncond_valid)

    # --- Vision conditioning: I2V anchors latent frame 0 to the real image; T2V is pure noise ---
    vision_condition_mask = jnp.zeros((b, latent_t), dtype=jnp.float32)  # 1.0 = clean/conditioned frame
    z_cond = None
    if args.image_path:
        img = Image.open(args.image_path).convert("RGB").resize((args.width, args.height))
        img_np = (np.array(img).astype(np.float32) / 127.5) - 1.0
        x_pixels = jnp.asarray(np.broadcast_to(img_np[None, None], (b, 1, args.height, args.width, 3)), dtype=dtype)
        x_pixels = vae_encoder.apply(vae_params, x_pixels, method=vae_encoder.pre_process)
        encoder_cfg = Encoder3d(vae_encoder.dim, vae_encoder.z_dim, vae_encoder.dim_mult,
                                 vae_encoder.num_res_blocks, vae_encoder.temperal_downsample, vae_encoder.eps)
        cache_list = [None] * _count_causal_convs_encoder(encoder_cfg)
        raw_out, cache_list = vae_encoder.apply(vae_params, x_pixels, cache_list, method=vae_encoder.encode_chunk)
        mu = vae_encoder.apply(vae_params, raw_out, method=vae_encoder.post_process)
        z_cond = jnp.zeros((b, latent_t, latent_h, latent_w, dit_model.latent_channel), dtype=jnp.float32)
        z_cond = z_cond.at[:, 0:1].set(mu.astype(jnp.float32))
        vision_condition_mask = vision_condition_mask.at[:, 0].set(1.0)

    key = jax.random.PRNGKey(args.seed)
    noise = jax.random.normal(key, (b, latent_t, latent_h, latent_w, dit_model.latent_channel), dtype=jnp.float32)
    if z_cond is not None:
        cm = vision_condition_mask[:, :, None, None, None]
        latents = cm * z_cond + (1.0 - cm) * noise
    else:
        latents = noise
    vision_noisy_mask = 1.0 - vision_condition_mask  # (B, T): 1.0 = noisy, 0.0 = clean/conditioned.

    def _dit_apply(params, input_ids, und_pos, und_valid, vis_latents, vis_pos, vis_sigma, vis_noisy):
        return dit_model.apply(
            params, input_ids=input_ids, und_position_ids=und_pos, und_valid_mask=und_valid,
            vision_latents=vis_latents, gen_position_ids=vis_pos, vision_sigma=vis_sigma,
            vision_noisy_mask=vis_noisy)

    @jax.jit
    def compute_velocity(current_latents, sigma_val, params, guide_scale):
        sigma_vec = jnp.broadcast_to(jnp.asarray(sigma_val, dtype=jnp.float32), (b, latent_t))
        vision_sigma = sigma_vec * scheduler.num_train_timesteps

        v_cond = _dit_apply(
            params, cond_input_ids, text_position_ids, cond_valid_mask,
            current_latents.astype(dtype), cond_vision_position_ids, vision_sigma, vision_noisy_mask)
        v_uncond = _dit_apply(
            params, uncond_input_ids, text_position_ids, uncond_valid_mask,
            current_latents.astype(dtype), uncond_vision_position_ids, vision_sigma, vision_noisy_mask)
        v_cond = v_cond.astype(jnp.float32) * vision_noisy_mask[:, :, None, None, None]
        v_uncond = v_uncond.astype(jnp.float32) * vision_noisy_mask[:, :, None, None, None]
        return v_uncond + guide_scale * (v_cond - v_uncond)

    logging.info(
        f"Running UniPC sampling for {args.num_steps} steps "
        f"(Karras sigmas, solver_order=2, guide_scale={args.guide_scale})...")
    unipc_state = scheduler.init_state()
    for step_index in range(scheduler.num_steps):
        sigma_val = scheduler.sigmas[step_index]
        velocity = compute_velocity(latents, sigma_val, dit_params, args.guide_scale)
        unipc_state, latents = scheduler.step(unipc_state, velocity, step_index, latents)
        if z_cond is not None:
            cm = vision_condition_mask[:, :, None, None, None]
            latents = cm * z_cond + (1.0 - cm) * latents

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    x_full = vae_decoder.apply(vae_params, latents.astype(dtype), method=vae_decoder.pre_process)
    decoder_cfg = Decoder3d(vae_decoder.dim, vae_decoder.z_dim, vae_decoder.dim_mult,
                             vae_decoder.num_res_blocks, vae_decoder.temperal_upsample, vae_decoder.eps)
    cache_list = [None] * _count_causal_convs(decoder_cfg)
    frames = []
    for i in range(x_full.shape[1]):
        out_chunk, cache_list = vae_decoder.apply(
            vae_params, x_full[:, i:i + 1], cache_list, i == 0, method=vae_decoder.decode_chunk)
        frames.append(out_chunk)
    video = jnp.concatenate(frames, axis=1)
    video = unpatchify(video, PATCH_SIZE)

    for i in range(b):
        out_np = np.array(video[i], dtype=np.float32)
        out_np = np.clip(out_np * 0.5 + 0.5, 0, 1)
        out_np = (out_np * 255).astype(np.uint8)
        out_path = args.output_path if b == 1 else f"{args.output_path.rsplit('.', 1)[0]}_{i}.mp4"
        save_video(out_np, out_path, fps=int(args.fps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2video / image2video generation with Cosmos3 on TPU.")
    parser.add_argument("--model_size", type=str, default="nano", choices=list(MODEL_SIZE_CONFIGS.keys()),
                         help="Which released Cosmos3 config to build (must match --dit_checkpoint_path's "
                              "actual checkpoint: Cosmos3-Nano or Cosmos3-Edge).")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True,
                         help="Path to the DiT .safetensors.index.json manifest "
                              "(checkpoints/Cosmos3-<Nano|Edge>/transformer/diffusion_pytorch_model.safetensors.index.json).")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True,
                         help="Path to the VAE .safetensors (checkpoints/Cosmos3-<Nano|Edge>/vae/diffusion_pytorch_model.safetensors).")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                         help="Path to the text tokenizer directory (checkpoints/Cosmos3-<Nano|Edge>/text_tokenizer).")
    parser.add_argument("--image_path", type=str, default=None,
                         help="Optional conditioning image for image2video (anchors latent frame 0).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--add_duration_template", type=lambda s: s.lower() != "false", default=True,
                         help="Append the reference's duration/FPS metadata sentence to the prompt "
                              "(matches Cosmos3OmniPipeline.__call__'s own default). Every real usage "
                              "example in refs/cosmos-main passes false for this.")
    parser.add_argument("--add_resolution_template", type=lambda s: s.lower() != "false", default=True,
                         help="Append the reference's resolution metadata sentence to the prompt "
                              "(matches Cosmos3OmniPipeline.__call__'s own default). Every real usage "
                              "example in refs/cosmos-main passes false for this.")
    parser.add_argument("--max_text_len", type=int, default=3072,
                         help="Fixed padded text-token length (JAX needs a static shape; the reference uses the "
                              "prompt's exact tokenized length instead). Must comfortably fit the *negative* "
                              "prompt too -- the default negative prompt (a JSON-structured, checkpoint-realistic "
                              "prompt, see docs/models/cosmos3.md#prompting) tokenizes to ~2800 tokens.")
    parser.add_argument("--guide_scale", type=float, default=6.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                         help="Number of devices to shard the DiT's attention heads/FFN channels "
                              "(Megatron-style) across. Must divide num_devices and num_attention_heads/"
                              "num_key_value_heads (32/8 for Nano, 16/8 for Edge).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--use_karras_sigmas", type=lambda s: s.lower() != "false", default=True,
                         help="Nano's default schedule (its scheduler_config.json: "
                              "use_karras_sigmas=true). Edge's own recipe instead uses a plain "
                              "shift-warped linear schedule -- pass --use_karras_sigmas false "
                              "--shift 12.0 for Edge.")
    parser.add_argument("--shift", type=float, default=5.0,
                         help="Only used when --use_karras_sigmas false.")
    parser.add_argument("--karras_sigma_min", type=float, default=0.147)
    parser.add_argument("--karras_sigma_max", type=float, default=200.0)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--output_path", type=str, default="output_cosmos3.mp4")
    args = parser.parse_args()
    main(args)
