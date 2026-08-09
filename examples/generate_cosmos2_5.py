# End-to-end text2world / image2world / video2world inference script for
# Cosmos-Predict2.5-2B on TPU.
#
# One checkpoint, three tasks, selected by what conditioning you pass:
# nothing (text2world, i.e. plain t2v), --image_path (image2world, a single
# conditioning frame), or --video_path (video2world, the first few frames of
# an existing clip). Unlike Wan2.1's i2v (a separate model, CLIP
# cross-attention + an extra concatenated `y` channel) but *like* Wan2.2
# TI2V-5B's i2v, conditioning here works by substituting the known frames'
# latents back into `x` between sampling steps -- except Cosmos additionally
# tells the DiT which frames are conditioning via a concatenated mask
# channel *and* a nonzero-but-tiny per-frame timestep for those frames
# (`CosmosDiT`'s `condition_video_mask` + per-frame `timesteps`), not just
# frame substitution alone. See `vidax.models.cosmos.cosmos2_5.dit`'s module
# docstring and `docs/models/cosmos.md`'s "Architecture notes" section.
#
# Known simplification vs. the reference: the reference's `denoise()` also
# forces the *predicted* x0 at conditioning-frame positions back to ground
# truth before it's consumed by the sampler's internal state
# (`denoise_replace_gt_frames=True` -- for UniPC specifically, this would
# mean patching the rolling `model_outputs` history, not just `x` itself).
# This script instead re-clamps only `x` (the latents) after every step,
# matching the precedent already established by
# `examples/generate_wan2_2_ti2v.py`'s i2v path (Euler, no internal solver
# state to patch) -- for UniPC's higher orders this is a real, if probably
# small, deviation from the reference: the corrector's short lookback window
# sees the model's own (slightly-off) prediction at conditioning positions
# for a step or two, not the exact ground truth. Worth revisiting if
# generated output looks off specifically near conditioning frames.

import argparse
import logging
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from vidax.core.sharding import (
    build_tpu_mesh, get_batch_sharding, get_replicated_sharding, shard_wan_params,
    to_partition_specs, configure_jax_cache,
)
from vidax.models.cosmos.common.reason1 import (
    NUM_EMBEDDING_PADDING_TOKENS, Qwen2TextModel, Reason1Tokenizer, compute_reason1_embeddings,
)
from vidax.models.cosmos.cosmos2_5.dit import CosmosDiT
from vidax.models.wan.wan2_1.vae import Decoder3d, WanVAEDecoder, WanVAEEncoder, _count_causal_convs
from vidax.schedulers.unipc import FlowUniPCMultistepScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}

# See `vidax.schedulers.unipc`'s / the reference's `sigma_conditional`: the
# tiny (not exactly zero) noise level conditioning frames are told they're
# at, rather than 0 -- matches `Video2WorldModel.denoise`'s
# `sigma_conditional=0.0001`.
CONDITIONAL_SIGMA = 0.0001

DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with "
    "no motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, pixelated images, poorly lit areas, underexposed and "
    "overexposed scenes, poor color balance, washed out colors, choppy "
    "sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
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


def resolve_batch_prompts(prompts: list, batch_size: int) -> list:
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) == batch_size:
        return prompts
    raise ValueError(
        f"Got {len(prompts)} prompts but the batch size is {batch_size}. "
        f"Pass exactly 1 prompt (broadcast to all replicas) or exactly {batch_size}.")


def encode_prompts(
    prompts: list, reason1_model: Qwen2TextModel, reason1_params, tokenizer: Reason1Tokenizer, dtype,
) -> jnp.ndarray:
    ids = jnp.asarray(tokenizer(prompts))
    context = compute_reason1_embeddings(reason1_params, ids, reason1_model)
    return context.astype(dtype)


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> tuple:
    """Picks the (width, height) closest to `w:h`'s aspect ratio, both
    divisible by (dw, dh), with area <= expected_area. Same algorithm as
    Wan2.1's/Wan2.2's own i2v resolution derivation (ported from the
    reference's `wan/utils/utils.py:best_output_size` -- Cosmos-Predict2.5
    reuses Wan2.1's VAE, so the same spatial-stride/patch-size divisibility
    constraints apply unchanged) -- tries rounding width-first and
    height-first, keeping whichever stays closer to the true ratio.
    """
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def resize_and_crop(orig_image: np.ndarray, pixel_h: int, pixel_w: int) -> np.ndarray:
    """Resizes (preserving aspect ratio, then center-cropping) a single RGB
    frame to exactly (pixel_h, pixel_w) -- same recipe as
    `generate_wan2_2_ti2v.py`'s i2v path."""
    orig_h, orig_w = orig_image.shape[0], orig_image.shape[1]
    scale = max(pixel_w / orig_w, pixel_h / orig_h)
    resized = Image.fromarray(orig_image).resize(
        (round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    x1, y1 = (resized.width - pixel_w) // 2, (resized.height - pixel_h) // 2
    return np.array(resized.crop((x1, y1, x1 + pixel_w, y1 + pixel_h)))


def build_cosmos_conditioning(
    pixel_frames: np.ndarray, vae_model: WanVAEEncoder, vae_params, dtype,
) -> jnp.ndarray:
    """Encodes `num_pixel_frames` conditioning frames (already resized/
    cropped) into a `(1, num_latent_frames, lat_h, lat_w, z_dim)` latent.

    `num_pixel_frames` must be `1 + 4 * (num_latent_frames - 1)` (the VAE's
    causal "1 + 4k" temporal grouping -- see `WanVAEEncoder`'s docstring);
    callers pick pixel-frame counts of 1 or 5 for 1 or 2 conditioning latent
    frames respectively, matching the reference's training distribution over
    `{0, 1, 2}` conditioning latent frames.

    Unlike `WanVAEDecoder.decode_chunk`'s ~20-chunk loop (needed because a
    full video's worth of decoded frames doesn't fit as one fused program),
    encoding 1-5 conditioning frames is small enough to just call
    `WanVAEEncoder.__call__` directly (which internally does its own,
    correctly-causal chunking) under a single `jax.jit`.
    """
    video = jnp.asarray(pixel_frames, dtype=jnp.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
    video = video[None].astype(dtype)  # (1, num_pixel_frames, H, W, 3)
    encode_jit = jax.jit(lambda params, x: vae_model.apply(params, x))
    return encode_jit(vae_params, video)


def main(args):
    configure_jax_cache()
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
    sequence_parallel = sp_size > 1
    assert not (args.image_path and args.video_path), (
        "Pass at most one of --image_path (image2world) / --video_path (video2world).")
    has_conditioning = args.image_path is not None or args.video_path is not None

    # `--sequence_parallel_size > 1` shards the DiT's token sequence itself
    # across the mesh's 'sp' axis (DeepSpeed-Ulysses -- see `CosmosDiT`'s
    # module docstring, ported directly from Wan's DiTs); `--tensor_parallel_size`
    # shards its weights (Megatron-style, attention heads/FFN channels)
    # across the independent 'tp' axis -- the two compose freely, see
    # `docs/hardware_and_sharding.md`'s "Combining both" section. Sequence
    # parallelism is off (size 1) by default: at the 2B model's typical
    # resolutions Megatron TP already fits fine on its own; it's there for
    # pushing to much higher resolution/frame counts, the same tradeoff
    # Wan2.1's t2v script documents.
    dit_model = CosmosDiT(mesh=mesh, sequence_parallel=sequence_parallel)
    vae_decoder = WanVAEDecoder()
    vae_encoder = WanVAEEncoder() if has_conditioning else None
    reason1_model = Qwen2TextModel()
    scheduler = FlowUniPCMultistepScheduler(
        num_steps=args.num_steps, shift=args.shift, solver_order=args.solver_order)

    assert dit_model.num_heads % (tp_size * sp_size) == 0, (
        f"CosmosDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size * --sequence_parallel_size ({tp_size} * {sp_size}) "
        f"(product constraint -- see generate_wan2_2_ti2v.py's identical assert).")
    # Reason1 always uses ordinary Megatron TP regardless of `--sequence_parallel_size`
    # (that flag only controls the DiT): its 512-token sequence is far too
    # short for sequence parallelism to matter, but at 7B params it's by far
    # the largest of the three checkpoints, so weight-sharding it still pays
    # off. `num_key_value_heads` (4, GQA) is the binding constraint --
    # `num_attention_heads` (28) divides evenly whenever it does too.
    assert reason1_model.num_key_value_heads % tp_size == 0, (
        f"Qwen2TextModel.num_key_value_heads ({reason1_model.num_key_value_heads}) must be "
        f"divisible by --tensor_parallel_size ({tp_size}); tp in {{1,2,4}}.")

    # Defaults to the Reason1 checkpoint's own directory: the released
    # Cosmos-Reason-1-7B repo bundles `tokenizer.json`/`tokenizer_config.json`/
    # `chat_template.json` right alongside the sharded `model-*.safetensors`
    # weights, so this Just Works for the common case (both paths pointing
    # into the same downloaded repo) without a network call to the HF hub.
    tokenizer_path = args.tokenizer_path or os.path.dirname(args.reason1_checkpoint_path)
    tokenizer = Reason1Tokenizer(tokenizer_path, seq_len=NUM_EMBEDDING_PADDING_TOKENS)

    # --- Load weights ---
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(args.dit_checkpoint_path, model_type="cosmos2.5_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(args.vae_checkpoint_path, model_type="wan2.1_vae")
    logging.info(f"Loading Reason1 text-encoder weights from {args.reason1_checkpoint_path}...")
    reason1_params = load_torch_checkpoint_to_jax(
        args.reason1_checkpoint_path, model_type="reason1_text_encoder")

    # The DiT is always Megatron tensor-sharded regardless of
    # `sequence_parallel` (weight-sharding and token-sharding are
    # independent mesh axes now -- see `generate_wan2_1_t2v.py`'s identical
    # comment). Reason1 is always Megatron tensor-sharded too (see the
    # assert above); the VAE is small and stays replicated either way.
    replicated = get_replicated_sharding(mesh)
    dit_params = cast_to_dtype(dit_params, dtype)
    dit_shardings = shard_wan_params(dit_params, mesh)
    dit_params = jax.device_put(dit_params, dit_shardings)
    vae_params = jax.device_put(cast_to_dtype(vae_params, dtype), replicated)
    reason1_params = jax.device_put(
        cast_to_dtype(reason1_params, dtype), shard_wan_params(reason1_params, mesh))
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Resolve output resolution + conditioning frames ---
    pt, ph, pw = dit_model.patch_size
    latent_t = 1 + (args.num_frames - 1) // 4

    num_cond_latent = 0
    z_cond = None
    if has_conditioning:
        num_cond_latent = 1 if args.image_path else args.num_conditional_latent_frames
        assert num_cond_latent in (1, 2), "--num_conditional_latent_frames must be 1 or 2."
        assert num_cond_latent <= latent_t, (
            f"--num_conditional_latent_frames ({num_cond_latent}) must be < the output's "
            f"latent frame count ({latent_t}) -- increase --num_frames.")
        num_cond_pixel = 1 + 4 * (num_cond_latent - 1)

        # Divisibility target for the *pixel* resolution: the VAE's 8x
        # spatial stride times the DiT's own patch_size, so that the
        # resulting *latent* grid (pixel / 8) comes out evenly divisible by
        # patch_size in turn. (Not `pw * 16`/`ph * 16` -- that's Wan2.2's
        # own constant, baking in *its* VAE's 16x compression; Cosmos reuses
        # Wan2.1's VAE, which is only 8x.)
        dw, dh = pw * 8, ph * 8
        if args.image_path:
            orig = np.array(Image.open(args.image_path).convert("RGB"))
        else:
            reader = imageio.get_reader(args.video_path)
            orig_frames = [reader.get_data(i) for i in range(num_cond_pixel)]
            reader.close()
            orig = orig_frames[0]
        orig_h, orig_w = orig.shape[0], orig.shape[1]
        pixel_w, pixel_h = best_output_size(orig_w, orig_h, dw, dh, args.max_area)
        latent_h, latent_w = pixel_h // 8, pixel_w // 8  # VAE's spatial stride only.

        if args.image_path:
            cond_frames = np.broadcast_to(
                resize_and_crop(orig, pixel_h, pixel_w)[None], (1, pixel_h, pixel_w, 3))
        else:
            cond_frames = np.stack([resize_and_crop(f, pixel_h, pixel_w) for f in orig_frames])
        logging.info(
            f"Conditioning: {'image2world' if args.image_path else 'video2world'}, "
            f"{orig_w}x{orig_h} -> {pixel_w}x{pixel_h}, {num_cond_latent} conditioning "
            f"latent frame(s) (--height/--width ignored).")

        logging.info("Encoding conditioning frame(s) with the VAE...")
        z_cond = build_cosmos_conditioning(cond_frames, vae_encoder, vae_params, dtype)
        z_cond = jax.device_put(z_cond, replicated)
    else:
        pixel_h, pixel_w = args.height, args.width
        latent_h, latent_w = pixel_h // 8, pixel_w // 8  # VAE's spatial stride only.

    assert latent_t % pt == 0 and latent_h % ph == 0 and latent_w % pw == 0, (
        f"latent grid {(latent_t, latent_h, latent_w)} must be divisible by "
        f"patch_size {dit_model.patch_size} -- adjust --num_frames/--height/--width/--max_area.")

    prompts = resolve_batch_prompts(args.prompt, dp_size)
    batch_size = len(prompts)
    latents_shape = (batch_size, latent_t, latent_h, latent_w, dit_model.in_channels)

    # --- Conditioning mask (1 = given, 0 = to-be-generated), broadcast to
    # the batch -- fed to the DiT's `condition_video_mask` input, and used
    # to both clamp the initial noise and re-clamp after every step (see
    # module docstring for the frame-substitution mechanism / its known
    # simplification vs. the reference). ---
    cond_mask = jnp.zeros((batch_size, latent_t, 1, 1, 1), dtype=jnp.float32)
    z_cond_padded = None
    if has_conditioning:
        cond_mask = cond_mask.at[:, :num_cond_latent].set(1.0)
        z_cond_padded = jnp.zeros(latents_shape, dtype=dtype)
        z_cond_padded = z_cond_padded.at[:, :num_cond_latent].set(
            jnp.broadcast_to(z_cond, (batch_size,) + z_cond.shape[1:]))
    # Per-frame (not per-token) mask for the *timestep* blend below, which
    # stays float32 (matching `scheduler.timesteps`'s dtype -- `CosmosDiT`
    # runs its own sinusoidal timestep embedding in float32 regardless, so
    # there's no reason to narrow this one).
    cond_frame_mask = cond_mask[:, :, 0, 0, 0]  # (B, T)
    # `cond_mask_full`, by contrast, is blended directly against `latents`/
    # `z_cond_padded` (both `dtype`, e.g. bf16) below and fed to the DiT as
    # `condition_video_mask` -- cast to `dtype` so neither of those silently
    # upcasts to float32 (`jnp.concatenate`/elementwise ops promote to the
    # widest operand dtype), which would otherwise pollute every activation
    # in the DiT forward pass *and* change `latents`' dtype mid-sampling-loop
    # (breaking `single_step`'s `donate_argnums=(0,)`, which requires the
    # donated buffer's dtype to stay fixed across calls).
    cond_mask_full = jnp.broadcast_to(cond_mask, (batch_size, latent_t, latent_h, latent_w, 1)).astype(dtype)
    cond_mask_full = jax.device_put(cond_mask_full, get_batch_sharding(mesh, cond_mask_full.ndim))

    latents_rng, rng = jax.random.split(rng)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=dtype)
    if has_conditioning:
        latents = cond_mask_full * z_cond_padded + (1.0 - cond_mask_full) * latents
    latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))

    logging.info(f"Encoding {batch_size} prompt(s) with Reason1: {prompts}")
    prompt_embeds = encode_prompts(prompts, reason1_model, reason1_params, tokenizer, dtype)
    prompt_embeds = jax.device_put(prompt_embeds, get_batch_sharding(mesh, prompt_embeds.ndim))

    negative_prompts = resolve_batch_prompts([args.negative_prompt], dp_size)
    negative_embeds = encode_prompts(negative_prompts, reason1_model, reason1_params, tokenizer, dtype)
    negative_embeds = jax.device_put(negative_embeds, get_batch_sharding(mesh, negative_embeds.ndim))

    # `sequence_parallel=True` reshapes/reshuffles activations across the
    # 'sp' mesh axis inside `CosmosDiT.__call__` itself -- these are
    # collectives, so the call needs to run inside `shard_map`, not a plain
    # `jax.jit` (which only sees ordinary per-device-local ops). `in_specs`
    # mirror the shardings already applied above: `get_batch_sharding`
    # shards the leading batch axis on 'dp' and replicates the rest,
    # `replicated`/`P()` fully replicated, and `dit_shardings` (computed
    # above) is the *real* per-leaf Megatron sharding, converted via
    # `to_partition_specs` -- not a blanket "fully replicated" `P()`, so
    # weight-sharding and sequence-parallel token-sharding both take effect
    # at once. `shard_map` requires the actual input array shardings to
    # agree with what it's told here. `cond_mask_full` is passed explicitly
    # (not closed over) for the same reason: `shard_map` needs every array
    # its traced function touches to be declared as an argument.
    def _dit_apply(params, latents, t, context, cond_mask):
        return dit_model.apply(
            params, latents=latents, timesteps=t, context=context,
            condition_video_mask=cond_mask)

    if sequence_parallel:
        dit_apply = shard_map(
            _dit_apply, mesh=mesh,
            in_specs=(
                to_partition_specs(dit_shardings), P('dp', None, None, None, None), P('dp', None),
                P('dp', None, None), P('dp', None, None, None, None),
            ),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

    # --- Timestep conditioning, no input/output preconditioning ---
    #
    # No EDM-style `c_in`/`c_skip`/`c_out` preconditioning here -- the
    # reference's actual inference loop (`generate_samples_from_batch`,
    # text2world_model_rectified_flow.py:493-583) passes the raw noisy
    # latent `xt` to the DiT unscaled and uses the DiT's raw output directly
    # as `velocity_pred`, fed straight into `self.sample_scheduler.step(...)`
    # -- no reconstruction step. (`RectifiedFlowScaling`, an EDM-style
    # preconditioning transform, belongs to a different reference model
    # class the rectified-flow checkpoint this script targets never uses.)
    #   - The timestep passed to both the DiT and the scheduler's own
    #     `.step()` is the *same* value, `self.sample_scheduler.timesteps`
    #     (i.e. `sigma * num_train_timesteps`, this module's own
    #     `scheduler.timesteps` attribute) -- matching `MinimalV1LVGDiT.
    #     forward`'s `timesteps_B_T * self.timestep_scale` (0.001): the
    #     `*1000` (schedule) and `*0.001` (DiT-internal) cancel, so the DiT
    #     ends up conditioned on plain, unscaled `sigma`.
    # This lines up exactly with `FlowUniPCMultistepScheduler.step`'s own
    # `convert_model_output` (`x0 = sample - sigma_t * model_output`,
    # docstring: "model_output: The predicted velocity (v_t) from the DiT")
    # -- the scheduler already consumes the DiT's raw output directly as
    # velocity, so no extra reconstruction step is needed on this side either.

    # Only the DiT forward pass (`compute_velocity`) is jitted, and its
    # signature has *no* static/step-dependent argument at all -- `sigma_vec`
    # is an ordinary traced array (its per-step *value* changes, not its
    # shape or dtype), so this compiles exactly **once** and is reused for
    # every step, the same "value varies, shape doesn't" reasoning the Wan
    # scripts already rely on for their own per-step jit.
    #
    # `scheduler.step(...)` (UniPC's predictor/corrector) is deliberately
    # called *eagerly*, outside any `jax.jit`, from the Python loop below,
    # not fused into `compute_velocity`'s jitted function. UniPC's `step()`
    # has genuine Python-level branching on `step_index`'s concrete value
    # (`if step_index > 0`, the `lower_order_final` ramp's
    # `min(self.num_steps - step_index, ...)`), which requires `step_index`
    # to be a *static* argument if this were jitted -- but marking an
    # argument static forces JAX to retrace (and recompile) the *entire*
    # enclosing jitted function every time that argument's value changes.
    # Fusing `compute_velocity`'s two full DiT forward passes into that same
    # jitted function would mean every one of `num_steps` steps triggers a
    # full DiT recompile. UniPC's own arithmetic (a handful of small einsums
    # over `solver_order`-many cached model outputs) is cheap enough that
    # running it eagerly, with ordinary per-op JAX dispatch overhead, costs
    # nothing that matters next to a 2B-parameter forward pass.
    # CFG needs two DiT forward passes (conditional and unconditional) that
    # are identical in every input except the text context -- rather than
    # two separate `dit_apply` calls (two dispatches, two sets of collective
    # ops under `sequence_parallel`/tensor parallelism), they're batched
    # into *one* call over a `2*B` batch (latents/mask/timestep each
    # duplicated, contexts concatenated), then split back apart. Total FLOPs
    # are the same either way, but this halves dispatch/collective-launch
    # overhead per step -- for a `num_steps`-step loop that's a real,
    # cumulative saving, not a micro-optimization.
    @jax.jit
    def compute_velocity(current_latents, sigma_vec, prompt_embeds, negative_embeds, params, guide_scale):
        # `c_noise` = `sigma * num_train_timesteps`, matching the reference's
        # `self.sample_scheduler.timesteps` fed to the DiT -- its internal
        # `timestep_scale=0.001` divides this back down to plain `sigma`.
        c_noise = sigma_vec * scheduler.num_train_timesteps  # (B, T)
        net_in = current_latents  # unscaled -- no c_in preconditioning.

        b = current_latents.shape[0]
        net_in_2b = jnp.concatenate([net_in, net_in], axis=0)
        c_noise_2b = jnp.concatenate([c_noise, c_noise], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        cond_mask_2b = jnp.concatenate([cond_mask_full, cond_mask_full], axis=0)

        net_out_2b = dit_apply(params, net_in_2b, c_noise_2b, context_2b, cond_mask_2b)
        v_cond, v_uncond = net_out_2b[:b], net_out_2b[b:]
        return v_uncond + guide_scale * (v_cond - v_uncond)

    logging.info(
        f"Running UniPC sampling for {args.num_steps} steps "
        f"(solver_order={args.solver_order}, shift={args.shift}, guide_scale={args.guide_scale})...")
    unipc_state = scheduler.init_state()
    for step_index in range(scheduler.num_steps):
        sigma_val = scheduler.sigmas[step_index]
        # Per-frame sigma: conditioning frames sit at a fixed, tiny noise
        # level (`CONDITIONAL_SIGMA`) instead of the current sampling sigma
        # -- lets a single forward pass see both "known, barely-noised"
        # frames and "being generated, at the true current noise level"
        # frames, matching `CosmosDiT`'s per-frame `(B, T)` timestep input
        # (`compute_velocity` turns this raw sigma into the DiT's actual
        # `c_noise` input via `sigma * num_train_timesteps`).
        sigma_vec = cond_frame_mask * CONDITIONAL_SIGMA + (1.0 - cond_frame_mask) * sigma_val

        velocity = compute_velocity(latents, sigma_vec, prompt_embeds, negative_embeds, dit_params, args.guide_scale)
        unipc_state, latents = scheduler.step(unipc_state, velocity, step_index, latents)

        if has_conditioning:
            # Re-clamp the conditioning frames' latents back to the known
            # VAE-encoded value -- matches the reference's frame-replacement
            # in `denoise()`, applied to `x` (see module docstring for the
            # one place this simplifies vs. the reference: the UniPC
            # corrector's internal history isn't separately patched).
            latents = cond_mask_full * z_cond_padded + (1.0 - cond_mask_full) * latents

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
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
    for i in range(batch_size):
        video_frames = np.array(decoded_frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end text2world / image2world / video2world generation with Cosmos-Predict2.5-2B on TPU.")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the DiT .pt checkpoint (e.g. checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/.../model_ema_bf16.pt).")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the (Wan2.1) VAE .pth checkpoint (checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth).")
    parser.add_argument("--reason1_checkpoint_path", type=str, required=True, help="Path to the Reason1 (Qwen2.5-VL-7B text tower) checkpoint's model.safetensors.index.json -- a separate download from Cosmos-Predict2.5-2B itself, from the nvidia/Cosmos-Reason1-7B repo.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="HuggingFace tokenizer id/path for Reason1. Defaults to the directory of --reason1_checkpoint_path (the released repo bundles the tokenizer files alongside the model shards).")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a conditioning image, for image2world generation (a single conditioning latent frame). Mutually exclusive with --video_path. When given, output resolution is derived from the image's aspect ratio + --max_area instead of --height/--width (which are then ignored).")
    parser.add_argument("--video_path", type=str, default=None, help="Path to a conditioning video, for video2world generation (its first --num_conditional_latent_frames-implied pixel frames are used). Mutually exclusive with --image_path. Same resolution-derivation behavior as --image_path.")
    parser.add_argument("--num_conditional_latent_frames", type=int, default=1, choices=[1, 2], help="video2world only: how many latent frames (1 or 2 -- the reference's own training distribution over {0,1,2}, minus the degenerate 0=t2v case) of the input video to condition on. Forced to 1 for --image_path.")
    parser.add_argument("--max_area", type=int, default=704 * 1280, help="image2world/video2world only: target output resolution's pixel area; actual (height, width) are derived from this and the conditioning frame's aspect ratio (see best_output_size).")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance.")
    parser.add_argument("--guide_scale", type=float, default=7.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond). Matches the reference's default of 7.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard the DiT's attention heads / FFN channels (weights) across. Also always used for Reason1's own (Megatron-only) weight sharding, independent of --sequence_parallel_size. Must divide num_devices, CosmosDiT.num_heads (16), and Qwen2TextModel.num_key_value_heads (4, the binding GQA constraint -- so tp in {1,2,4} if Reason1 is being sharded meaningfully, though the DiT alone would tolerate {1,2,4,8,16}). Composes independently with --sequence_parallel_size (their product is the DiT's real head-divisibility constraint).")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Number of devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of --tensor_parallel_size's weight-sharding. 1 (off) by default since Megatron TP already fits the 2B model fine at typical resolutions; useful for pushing to much higher resolution/frame counts, where self-attention activation memory (not weight memory) becomes the bottleneck. Also requires the latent frame count (`1 + (num_frames - 1) // 4`) to be evenly divisible by this value.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the DiT, VAE, and Reason1 encoder.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of UniPC sampling steps. The reference's default for the 2B base checkpoint is 35.")
    parser.add_argument("--solver_order", type=int, default=2, help="UniPC solver order. The reference's default is 2.")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-matching noise-schedule shift. Matches the reference's default.")
    parser.add_argument("--height", type=int, default=704, help="Output video height (must be divisible by 16). Ignored if --image_path/--video_path is given.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width (must be divisible by 16). Ignored if --image_path/--video_path is given.")
    parser.add_argument("--num_frames", type=int, default=93, help="Number of frames in the output video. The reference trains around 93-frame clips at 720p.")
    parser.add_argument("--fps", type=int, default=16, help="Output video frame rate. Matches the reference's 16 fps training/inference setting.")
    parser.add_argument("--output_path", type=str, default="output_cosmos2_5.mp4", help="Path to save the output MP4 video(s). With multiple prompts, each video is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
