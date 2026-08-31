# End-to-end text-to-video / image-to-video inference for CogVideoX on TPU.
#
# One script/pipeline for every released checkpoint (--variant):
#   2b          THUDM/CogVideoX-2b        (t2v, no RoPE, learned 3D sincos pos-embed, snr_shift 3)
#   5b          THUDM/CogVideoX-5b        (t2v, 3D RoPE, v-prediction)          [default]
#   5b-i2v      THUDM/CogVideoX-5b-I2V    (i2v, image latent concat, learned pos-embed)
#   1.5-5b      THUDM/CogVideoX1.5-5B     (t2v, patch_size_t=2, "slice" RoPE)
#   1.5-5b-i2v  THUDM/CogVideoX1.5-5B-I2V (i2v, + ofs embedding)
#
# T2V and I2V share this script exactly as the diffusers pipelines do; pass
# --image_path for I2V (only meaningful for the *-i2v variants), omit it for T2V.
#
# Megatron-style tensor parallelism (--tensor_parallel_size, see
# vidax.core.sharding) shards the DiT's / T5's attention heads; the FFN stays
# replicated (its Flax submodule names -- net_0_proj / net_2 -- aren't in
# sharding.py's column/row-parallel name lists), which is fine since the 5B
# checkpoint's bf16 weights fit replicated on a single TPU v4 chip. The VAE is
# always replicated (same as every other model's script).

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
    build_tpu_mesh, configure_jax_cache, get_batch_sharding, get_replicated_sharding,
    shard_wan_params, to_partition_specs,
)
from vidax.models.cogvideo.configs import CONFIGS, RESOLUTION_MAP, dit_kwargs
from vidax.models.cogvideo.dit import CogVideoXDiT
from vidax.models.cogvideo.rope import prepare_rotary_positional_embeddings
from vidax.models.cogvideo.t5 import CogVideoXT5Tokenizer, T5Encoder, MAX_TEXT_SEQ_LENGTH
from vidax.models.cogvideo.vae import CogVideoXVAE
from vidax.schedulers.cogvideox import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
VAE_SCALE_SPATIAL = 8
VAE_SCALE_TEMPORAL = 4

# The diffusers CogVideoX examples' own default negative prompt is the empty
# string (`negative_prompt = negative_prompt or ""`).
DEFAULT_NEGATIVE_PROMPT = ""


def save_video(frames: np.ndarray, output_path: str, fps: int):
    logging.info(f"Saving {frames.shape[0]} frames to {output_path}...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def cast_to_dtype(tree, dtype):
    def cast_leaf(x):
        if jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x
    return jax.tree_util.tree_map(cast_leaf, tree)


def dynamic_cfg_scale(guidance_scale: float, t: int, num_steps: int) -> float:
    """diffusers pipeline's `use_dynamic_cfg` schedule."""
    return 1.0 + guidance_scale * (
        (1.0 - np.cos(np.pi * ((num_steps - t) / num_steps) ** 5.0)) / 2.0)


def encode_prompts(prompts, t5_model, t5_params, tokenizer, out_dtype):
    """T5-encode, padded to 226 tokens, *no* attention mask -- matching
    diffusers `_get_t5_prompt_embeds` (which never passes a mask to the
    encoder, so padding tokens are attended).

    The encoder itself runs in float32 (see `main`): T5-XXL's intermediate
    activations reach ~1e5, so bf16 through the unmasked 226-token sequence
    loses all precision (verified: ~20-40% rel error vs an fp32 reference,
    vs 3e-5 in fp32). The bf16 cast happens only on the *output* embeddings
    the DiT consumes.
    """
    ids, _ = tokenizer(prompts)
    return t5_model.apply(t5_params, jnp.asarray(ids)).astype(out_dtype)


def load_conditioning_image(path, height, width):
    """Loads and resizes the conditioning frame to (width, height) -> a
    (1, 1, H, W, 3) array in [-1, 1]. Returns `(arr, (orig_w, orig_h))` so the
    caller can restore the source aspect ratio on the output."""
    from PIL import Image
    src = Image.open(path).convert("RGB")
    orig_w, orig_h = src.size
    img = src.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr[None, None], (orig_w, orig_h)


def _snap(v, m):
    return max(m, int(round(v / m)) * m)


def i2v_output_size(orig_w, orig_h, gen_w, gen_h):
    """The size to rescale the generated (gen_w x gen_h) video to so it carries
    the conditioning image's aspect ratio -- preserving the generated pixel
    budget, snapped to a multiple of 16 (video-codec macro-block size, so
    imageio-ffmpeg doesn't silently re-pad the frames)."""
    budget = gen_w * gen_h
    ar = orig_w / orig_h
    h = _snap((budget / ar) ** 0.5, 16)
    w = _snap(h * ar, 16)
    return w, h


def resize_video(frames: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """frames: (T, H, W, 3) uint8 -> (T, out_h, out_w, 3) uint8, LANCZOS."""
    from PIL import Image
    if frames.shape[2] == out_w and frames.shape[1] == out_h:
        return frames
    return np.stack([
        np.asarray(Image.fromarray(f).resize((out_w, out_h), Image.LANCZOS)) for f in frames])


def main(args):
    configure_jax_cache()
    num_devices = jax.device_count()
    preset = CONFIGS[args.variant]
    sp_size = args.sequence_parallel_size or 1
    sequence_parallel = sp_size > 1
    if sequence_parallel:
        # DeepSpeed-Ulysses over the DiT's visual token sequence -- for
        # CogVideoX-1.5 at its native 1360x768 (~45k visual tokens), whose
        # per-block activations don't fit a v4 chip otherwise. CogVideoX keeps
        # this mutually exclusive with Megatron TP (no combined column/row-
        # parallel shape juggling in the DiT), so `--tensor_parallel_size` must
        # be 1 (or unset) alongside `--sequence_parallel_size`.
        if args.tensor_parallel_size not in (None, 1):
            raise SystemExit("--sequence_parallel_size > 1 requires --tensor_parallel_size 1 "
                             "for CogVideoX (TP and SP are mutually exclusive here).")
        assert num_devices % sp_size == 0, (
            f"--sequence_parallel_size ({sp_size}) must divide num_devices ({num_devices}).")
        assert preset["num_attention_heads"] % sp_size == 0, (
            f"--sequence_parallel_size ({sp_size}) must divide num_attention_heads "
            f"({preset['num_attention_heads']}).")
        tp_size = 1
        dp_size = num_devices // sp_size
    else:
        tp_size = args.tensor_parallel_size or num_devices
        # CogVideoX-2b has 30 attention heads (not a multiple of 4) -- cap tp at
        # the largest divisor of both the device count and the head count, and
        # give the leftover devices to data parallelism (dp * tp == num_devices).
        while num_devices % tp_size or preset["num_attention_heads"] % tp_size:
            tp_size -= 1
        dp_size = num_devices // tp_size
        if tp_size != (args.tensor_parallel_size or num_devices):
            logging.warning("Reduced --tensor_parallel_size to %d (must divide num_devices=%d and "
                            "num_attention_heads=%d); data_parallel_size=%d.",
                            tp_size, num_devices, preset["num_attention_heads"], dp_size)
    mesh = build_tpu_mesh(
        data_parallel_size=dp_size, tensor_parallel_size=tp_size, sequence_parallel_size=sp_size)
    rng = jax.random.PRNGKey(args.seed)

    dtype = DTYPES[args.dtype]
    dit_dtype = DTYPES[args.dit_dtype]
    is_i2v = args.image_path is not None
    patch_size_t = preset["patch_size_t"]
    # 5b-I2V's learned positional-embedding buffer locks it to 720x480; 1.5
    # and 2b take any resolution (VAE ÷8, patchify ÷2 -> ÷16 overall).
    res_locked = preset["use_learned_positional_embeddings"]

    src_size = None
    if is_i2v:
        from PIL import Image
        src_size = Image.open(args.image_path).size  # (w, h)

    if args.height is None or args.width is None:
        rw, rh = RESOLUTION_MAP[args.variant]
        if is_i2v and not res_locked and not sequence_parallel and src_size is not None:
            # Flexible I2V *without* sequence parallelism (1.5-5B-I2V on a
            # single-mesh run): generate at the conditioning image's aspect
            # ratio, pixel budget near 720x480 rather than 1.5's native
            # 1360x768 -- the ~45k-token DiT graph doesn't compile in practical
            # time under plain Megatron TP (see docs/lessons/cogvideox_debugging
            # .md §5). Snap to /16, keep each side within the slice-RoPE table.
            ow, oh = src_size
            ar = ow / oh
            budget = 720 * 480
            h = min(_snap((budget / ar) ** 0.5, 16), rh)
            w = min(_snap(h * ar, 16), rw)
            args.height, args.width = h, w
        else:
            # T2V, res-locked I2V (5b-i2v -> 720x480), or 1.5 I2V under
            # --sequence_parallel_size (which makes native 1360x768 tractable):
            # generate at the variant's reference resolution. For I2V the saved
            # video is then rescaled to the image's aspect (--match_image_aspect).
            args.height, args.width = rh, rw

    # --- models ---
    # The T5 (`t5-v1.1-xxl`) weights + tokenizer are byte-identical across every
    # CogVideoX repo (only the stored dtype differs, and bf16 -> f32 is exact),
    # so `--t5_dir` can point at any one downloaded copy.
    t5_dir = args.t5_dir or os.path.join(args.model_dir, "text_encoder")
    tok_dir = args.tokenizer_dir or os.path.join(args.model_dir, "tokenizer")

    dit_model = CogVideoXDiT(**dit_kwargs(args.variant), compute_dtype=dit_dtype, mesh=mesh,
                             sequence_parallel=sequence_parallel)
    vae_model = CogVideoXVAE()
    t5_model = T5Encoder()
    tokenizer = CogVideoXT5Tokenizer(tok_dir, seq_len=MAX_TEXT_SEQ_LENGTH)

    if args.scheduler == "ddim":
        scheduler = CogVideoXDDIMScheduler(args.num_inference_steps, snr_shift_scale=preset["snr_shift_scale"])
    else:
        scheduler = CogVideoXDPMScheduler(args.num_inference_steps, snr_shift_scale=preset["snr_shift_scale"])

    # --- weights ---
    def _ckpt(dir_, stem):
        """<stem>.safetensors.index.json (sharded) else <stem>.safetensors."""
        idx = os.path.join(dir_, f"{stem}.safetensors.index.json")
        return idx if os.path.exists(idx) else os.path.join(dir_, f"{stem}.safetensors")

    logging.info(f"Loading CogVideoX-{args.variant} weights from {args.model_dir}...")
    dit_params = load_torch_checkpoint_to_jax(
        _ckpt(os.path.join(args.model_dir, "transformer"), "diffusion_pytorch_model"),
        model_type="cogvideox_dit")
    vae_params = load_torch_checkpoint_to_jax(
        _ckpt(os.path.join(args.model_dir, "vae"), "diffusion_pytorch_model"),
        model_type="cogvideox_vae")
    t5_params = load_torch_checkpoint_to_jax(_ckpt(t5_dir, "model"), model_type="ltx_video_t5")

    replicated = get_replicated_sharding(mesh)
    dit_params = jax.device_put(cast_to_dtype(dit_params, dit_dtype), shard_wan_params(dit_params, mesh))
    # T5 runs in float32 (see encode_prompts); Megatron-sharded like the DiT.
    t5_params = jax.device_put(cast_to_dtype(t5_params, jnp.float32), shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(cast_to_dtype(vae_params, jnp.float32), replicated)
    logging.info("Weights loaded, cast, and sharded.")

    # --- text ---
    batch_size = len(args.prompt)
    prompt_embeds = encode_prompts(args.prompt, t5_model, t5_params, tokenizer, dit_dtype)
    negative_embeds = encode_prompts([args.negative_prompt] * batch_size, t5_model, t5_params, tokenizer, dit_dtype)
    context_2b = jnp.concatenate([negative_embeds, prompt_embeds], axis=0)  # uncond first
    del t5_params  # ~19 GB in fp32; not needed after the one-time prompt encode

    # --- latents (mirrors CogVideoX{,ImageToVideo}Pipeline.prepare_latents) ---
    latent_frames_base = (args.num_frames - 1) // VAE_SCALE_TEMPORAL + 1
    # CogVideoX-1.5 (patch_size_t=2) pads the latent-frame count up to a
    # multiple of patch_size_t; those pad frames are stripped from the front
    # again before VAE decode. diffusers' i2v `additional_frames = pt - n % pt`
    # is guarded by `n % pt != 0`, i.e. exactly the `% pt` below.
    extra_t = 0 if patch_size_t is None else \
        (patch_size_t - latent_frames_base % patch_size_t) % patch_size_t
    latent_frames = latent_frames_base + extra_t
    additional_frames = extra_t  # stripped from the front again before VAE decode
    latent_h = args.height // VAE_SCALE_SPATIAL
    latent_w = args.width // VAE_SCALE_SPATIAL
    latch = preset["in_channels"] if not is_i2v else preset["out_channels"]

    rng, lat_rng = jax.random.split(rng)
    latents = jax.random.normal(
        lat_rng, (batch_size, latent_frames, latch, latent_h, latent_w), dtype=jnp.float32)
    latents = latents * scheduler.init_noise_sigma

    # --- I2V conditioning (mirrors CogVideoXImageToVideoPipeline.prepare_latents) ---
    image_latents = None
    ofs = None
    if is_i2v:
        # Encode the SINGLE conditioning frame, then pad with zero *latents*
        # out to `latent_frames` -> image_latents = [real, zeros x (N-1)],
        # matching CogVideoXImageToVideoPipeline.prepare_latents. (That method's
        # `first_frame` prepend is dead code -- `additional_frames` makes its
        # slice empty -- so don't re-add a prepend here.)
        img, src_size = load_conditioning_image(args.image_path, args.height, args.width)  # (1,1,H,W,3), (w,h)
        img = np.broadcast_to(img, (batch_size, 1, args.height, args.width, 3)).astype(np.float32)
        mean, _ = vae_model.apply(vae_params, jnp.asarray(img), method=vae_model.encode)  # (B, 1, Hl, Wl, 16)
        sf = preset["scaling_factor"]
        # `invert_scale_latents` (1.5): the CogVideoX team forgot the training-
        # time scaling, so I2V un-scales instead of scales.
        cond = mean / sf if preset["invert_scale_latents"] else mean * sf
        cond = jnp.transpose(cond, (0, 1, 4, 2, 3))  # (B, F=1, C=16, Hl, Wl) -- matches `latents` layout
        pad_t = latent_frames - 1
        if pad_t > 0:
            cond = jnp.concatenate(
                [cond, jnp.zeros((batch_size, pad_t) + cond.shape[2:], cond.dtype)], axis=1)
        image_latents = cond  # (B, latent_frames, 16, Hl, Wl), matching `latents`
        if preset["ofs_embed_dim"] is not None:
            # diffusers: `latents.new_full((1,), fill_value=2.0)` -- constant 2.0,
            # broadcast over the batch inside the ofs embedding.
            ofs = jnp.full((1,), 2.0, dtype=jnp.float32)

    # --- RoPE ---
    rope_cos = rope_sin = None
    if preset["use_rotary_positional_embeddings"]:
        c, s = prepare_rotary_positional_embeddings(
            args.height, args.width, latent_frames,
            vae_scale_factor_spatial=VAE_SCALE_SPATIAL, patch_size=2, patch_size_t=patch_size_t,
            attention_head_dim=64, sample_height=preset["sample_height"], sample_width=preset["sample_width"])
        rope_cos, rope_sin = jnp.asarray(c), jnp.asarray(s)

    def _dit_call(p, hs, ehs, ts, rc, rs, of):
        return dit_model.apply(p, hs, ehs, ts, rc, rs, of)

    if sequence_parallel:
        # `sequence_parallel=True` chunks the visual token sequence across the
        # 'sp' mesh axis inside `CogVideoXDiT` itself (in `pre_process`, with an
        # all-to-all around the joint attention and an all-gather before
        # unpatchify) -- those are collectives, so the call must run inside
        # `shard_map`, not a plain `jax.jit`. `in_specs` mirror the shardings of
        # the arguments: `dit_params` keep their (here trivially tp=1) per-leaf
        # spec via `to_partition_specs`, latents/context/timestep are batch-
        # sharded on 'dp', the RoPE tables are replicated (chunked internally).
        rope_spec = P() if rope_cos is not None else None
        ofs_spec = P() if ofs is not None else None
        dit_apply = jax.jit(shard_map(
            _dit_call, mesh=mesh,
            in_specs=(to_partition_specs(shard_wan_params(dit_params, mesh)),
                      P('dp', None, None, None, None),   # hidden_states (B, F, C, H, W)
                      P('dp', None, None),               # context (B, 226, 4096)
                      P('dp'),                           # timestep (B,)
                      rope_spec, rope_spec, ofs_spec),
            out_specs=P('dp', None, None, None, None),
            check_rep=False))
        context_2b = jax.device_put(context_2b, get_batch_sharding(mesh, context_2b.ndim))
        if rope_cos is not None:
            rope_cos = jax.device_put(rope_cos, replicated)
            rope_sin = jax.device_put(rope_sin, replicated)
        if ofs is not None:
            ofs = jax.device_put(ofs, replicated)
    else:
        dit_apply = jax.jit(_dit_call)

    # --- denoising loop (plain Python, like generate_ltx_video.py) ---
    timesteps = scheduler.timesteps
    old_pred = None
    logging.info(f"Running {len(timesteps)} {args.scheduler} steps (guidance_scale={args.guidance_scale})...")
    for i, t in enumerate(timesteps):
        lmi = jnp.concatenate([latents, latents], axis=0).astype(dit_dtype)
        if is_i2v:
            lmi = jnp.concatenate([lmi, jnp.concatenate([image_latents, image_latents], axis=0).astype(dit_dtype)], axis=2)
        ts_2b = jnp.full((2 * batch_size,), float(t), dtype=jnp.float32)
        noise_pred = dit_apply(dit_params, lmi, context_2b, ts_2b, rope_cos, rope_sin, ofs)
        noise_pred = np.asarray(noise_pred, dtype=np.float32)

        gs = dynamic_cfg_scale(args.guidance_scale, int(t), args.num_inference_steps) if args.use_dynamic_cfg \
            else args.guidance_scale
        uncond, text = noise_pred[:batch_size], noise_pred[batch_size:]
        noise_pred = uncond + gs * (text - uncond)

        lat32 = np.asarray(latents, dtype=np.float32)
        if args.scheduler == "ddim":
            latents, _ = scheduler.step(noise_pred, int(t), lat32)
        else:
            rng, k1, k2 = jax.random.split(rng, 3)
            n1 = np.asarray(jax.random.normal(k1, lat32.shape, dtype=jnp.float32))
            n2 = np.asarray(jax.random.normal(k2, lat32.shape, dtype=jnp.float32))
            latents, old_pred = scheduler.step(
                noise_pred, old_pred, int(t), int(timesteps[i - 1]) if i > 0 else None, lat32, n1, n2)
        latents = jnp.asarray(np.asarray(latents, dtype=np.float32))

    # --- decode ---
    if additional_frames:
        latents = latents[:, additional_frames:]
    sf = preset["scaling_factor"]
    # latents: (B, Tlat, C, Hlat, Wlat) -> channels-last (B, Tlat, Hlat, Wlat, C) for CogVideoXVAE.
    # The VAE decode runs *eagerly* (not jax.jit-wrapped): the tiled/chunked
    # decode loop, unrolled by jit, holds every tile's 512-channel 3D-conv
    # activations live at once and OOMs a v4 chip -- same rationale as
    # vidax.models.wan.wan2_2.vae's docstring.
    z = jnp.transpose(latents, (0, 1, 3, 4, 2)) / sf
    frames = vae_model.apply(vae_params, z.astype(jnp.float32), method=vae_model.decode)  # (B, Tpix, H, W, 3)

    # I2V: give the output the conditioning image's aspect ratio. 5b-I2V is
    # locked to 720x480, so its conditioning frame was squished -- rescaling
    # the output back to the source aspect un-squishes it. 1.5-5B-I2V was
    # already generated at the source aspect (above), so this is a no-op.
    out_w, out_h = args.width, args.height
    if is_i2v and getattr(args, "match_image_aspect", True):
        out_w, out_h = i2v_output_size(src_size[0], src_size[1], args.width, args.height)
        if (out_w, out_h) != (args.width, args.height):
            logging.info("Rescaling I2V output %dx%d -> %dx%d to match the image's aspect ratio.",
                         args.width, args.height, out_w, out_h)

    base, ext = os.path.splitext(args.output_path)
    for b in range(batch_size):
        vid = np.clip(np.asarray(frames[b], np.float32) * 0.5 + 0.5, 0, 1)
        vid = (vid * 255).astype(np.uint8)
        vid = resize_video(vid, out_w, out_h)
        out = args.output_path if batch_size == 1 else f"{base}_{b}{ext}"
        save_video(vid, out, fps=args.fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="End-to-end CogVideoX T2V/I2V generation on TPU.")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Path to a downloaded diffusers CogVideoX repo (with transformer/ vae/ text_encoder/ tokenizer/).")
    p.add_argument("--variant", type=str, default="5b", choices=list(CONFIGS.keys()))
    p.add_argument("--t5_dir", type=str, default=None,
                   help="Override for the T5 text_encoder dir (default: <model_dir>/text_encoder). "
                        "The t5-v1.1-xxl weights are identical across all CogVideoX repos.")
    p.add_argument("--tokenizer_dir", type=str, default=None,
                   help="Override for the tokenizer dir (default: <model_dir>/tokenizer).")
    p.add_argument("--prompt", type=str, required=True, nargs="+")
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--image_path", type=str, default=None, help="Conditioning image for the *-i2v variants.")
    p.add_argument("--match_image_aspect", action="store_true", default=True,
                   help="I2V: rescale the output video to the conditioning image's aspect ratio "
                        "(default on; 5b-i2v is locked to 720x480 so its output is otherwise squished).")
    p.add_argument("--no_match_image_aspect", dest="match_image_aspect", action="store_false")
    p.add_argument("--num_frames", type=int, default=None, help="Default: 49 (1.0) / 81 (1.5).")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--scheduler", type=str, default="dpm", choices=["ddim", "dpm"])
    p.add_argument("--use_dynamic_cfg", action="store_true", default=True)
    p.add_argument("--no_dynamic_cfg", dest="use_dynamic_cfg", action="store_false")
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="Megatron shard of DiT/T5 attention heads. Default: all devices "
                        "(capped to a divisor of the head count; 2b -> 2 on a v4-8). "
                        "Must be 1 (or unset) when --sequence_parallel_size > 1.")
    p.add_argument("--sequence_parallel_size", type=int, default=1,
                   help="DeepSpeed-Ulysses shard of the DiT's visual token sequence across "
                        "this many devices (mutually exclusive with --tensor_parallel_size for "
                        "CogVideoX). This is what fits CogVideoX-1.5 at its native 1360x768 "
                        "(~45k visual tokens) on a v4-8 -- e.g. --sequence_parallel_size 4. "
                        "Must divide num_devices and num_attention_heads, and the DiT's visual "
                        "token count must be evenly divisible by it.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    p.add_argument("--dit_dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--output_path", type=str, default="output_cogvideox.mp4")
    args = p.parse_args()
    if args.num_frames is None:
        args.num_frames = 81 if args.variant.startswith("1.5") else 49
    main(args)
