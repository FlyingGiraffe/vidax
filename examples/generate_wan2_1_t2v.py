# End-to-end text-to-video inference script for Wan2.1 on TPU.
#
# Supports Megatron-style 1D tensor parallelism (splitting attention heads
# and FFN channels across chips) in addition to data parallelism across the
# batch, since full-resolution DiT self-attention (tens of thousands of
# patches) needs more per-chip HBM than a single TPU v4 chip has if run
# purely data-parallel / replicated. See vidax.core.sharding for the
# sharding scheme.

import argparse
from functools import partial
import logging
import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding

from vidax.core.sharding import (
    build_tpu_mesh, shard_wan_params, get_replicated_sharding, get_batch_sharding,
    to_partition_specs, configure_jax_cache,
)
from vidax.core.rope3d import create_rope3d_freqs
from vidax.models.wan.wan2_1.configs import T2V_1_3B_CONFIG, T2V_14B_CONFIG
from vidax.models.wan.wan2_1.dit import WanDiT
from vidax.models.wan.wan2_1.vae import WanVAEDecoder, Decoder3d, _count_causal_convs
from vidax.models.wan.common.t5 import T5Encoder, Umt5Tokenizer
from vidax.schedulers.flow_match import RectifiedFlowScheduler
from vidax.translator.mappings import load_torch_checkpoint_to_jax

logging.basicConfig(level=logging.INFO)

DTYPES = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
MODEL_SIZE_CONFIGS = {"1.3B": T2V_1_3B_CONFIG, "14B": T2V_14B_CONFIG}

# Wan2.1's default negative prompt (Wan2.1-main/wan/configs/shared_config.py,
# `sample_neg_prompt`), used for classifier-free guidance.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
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


def resolve_batch_prompts(prompts: list, batch_size: int) -> list:
    """Broadcasts a single prompt to fill the batch, or requires one prompt per slot.

    This lets `batch_size` (== the data-parallel mesh size) double as either
    "N distinct prompts" (one video per prompt) or "N noise samples of one
    prompt" (each data-parallel replica gets independent noise).
    """
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) == batch_size:
        return prompts
    raise ValueError(
        f"Got {len(prompts)} prompts but the data-parallel batch size is "
        f"{batch_size} (num_devices // tensor_parallel_size). Pass exactly 1 "
        f"prompt (broadcast to all replicas) or exactly {batch_size}.")


def encode_prompts(prompts: list, t5_model: T5Encoder, t5_params, tokenizer: Umt5Tokenizer,
                    dtype) -> jnp.ndarray:
    """Tokenizes and T5-encodes one prompt per batch element.

    Matches the reference `T5EncoderModel.__call__`: the encoder runs over
    the full padded sequence with an attention mask, then everything past
    each sample's real (unpadded) length is zeroed out -- WanDiT's cross
    attention has no length masking of its own, so what's actually in those
    trailing positions matters, not just what attends to them.
    """
    ids, mask = tokenizer(prompts)
    ids, mask = jnp.asarray(ids), jnp.asarray(mask)

    context = t5_model.apply(t5_params, ids, mask)
    seq_lens = mask.sum(axis=1)
    positions = jnp.arange(context.shape[1])[None, :]
    keep = positions < seq_lens[:, None]
    return jnp.where(keep[..., None], context, 0.0).astype(dtype)


def main(args):
    """Main inference function."""
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

    # --- Initialize models and scheduler ---
    # `--sequence_parallel_size > 1` shards the DiT's token sequence itself
    # across the mesh's 'sp' axis (DeepSpeed-Ulysses -- see `WanDiT`'s
    # module docstring); `--tensor_parallel_size` shards its weights
    # (Megatron-style, attention heads/FFN channels) across the independent
    # 'tp' axis. The two compose freely -- see `docs/hardware_and_sharding
    # .md`'s "Combining both" section. Sequence parallelism is off (size 1)
    # by default since Megatron TP already fits the 1.3B model fine at
    # typical resolutions; it's what the much larger 14B model needs at
    # higher resolutions instead, the same issue Wan2.2's 5B model hit first
    # (see `vidax.models.wan.wan2_2.dit`'s module docstring for the
    # underlying reasoning -- Wan2.1's timestep modulation isn't per-token
    # the way Wan2.2's is, so it doesn't contribute to that specific
    # problem, but self-attention activation memory for a big-enough DiT
    # still doesn't shrink under Megatron TP alone).
    dit_model = WanDiT(
        mesh=mesh, sequence_parallel=sequence_parallel,
        **MODEL_SIZE_CONFIGS[args.model_size])
    vae_model = WanVAEDecoder()
    t5_model = T5Encoder()
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps, shift=args.shift)

    assert dit_model.num_heads % tp_size == 0, (
        f"WanDiT.num_heads ({dit_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}); e.g. tp in {{1,2,3,4,6,12}} for the 1.3B model, "
        f"{{1,2,4,5,8,10,20,40}} for the 14B model.")
    assert t5_model.num_heads % tp_size == 0, (
        f"T5Encoder.num_heads ({t5_model.num_heads}) must be divisible by "
        f"--tensor_parallel_size ({tp_size}).")

    tokenizer_path = args.tokenizer_path or os.path.join(
        os.path.dirname(args.t5_checkpoint_path), "google", "umt5-xxl")
    tokenizer = Umt5Tokenizer(tokenizer_path, seq_len=dit_model.text_len)

    # --- Load weights (DiT, VAE, and T5 ship as separate checkpoints) ---
    logging.info(f"Loading DiT weights from {args.dit_checkpoint_path}...")
    dit_params = load_torch_checkpoint_to_jax(
        args.dit_checkpoint_path, model_type="wan2.1_dit")
    logging.info(f"Loading VAE weights from {args.vae_checkpoint_path}...")
    vae_params = load_torch_checkpoint_to_jax(
        args.vae_checkpoint_path, model_type="wan2.1_vae")
    logging.info(f"Loading T5 weights from {args.t5_checkpoint_path}...")
    t5_params = load_torch_checkpoint_to_jax(
        args.t5_checkpoint_path, model_type="wan_t5")

    # Shard onto devices *before* dtype-casting: casting is elementwise and
    # preserves sharding (each device casts only its own local shard), so
    # doing it after sharding avoids ever holding two full-size copies of a
    # multi-GB param tree on a single device. Casting itself happens on the
    # host (numpy), before any of this -- see
    # `vidax.translator.converter.convert_pt_tensor_to_jax`'s docstring for
    # why (matters most for the DiT once it's large enough to ship as raw
    # float32, e.g. Wan2.2's 5B/14B; Wan2.1's checkpoints are already bf16).
    # T5 and the DiT are both tensor-parallel sharded (attention heads / FFN
    # channels split across the 'tp' axis) regardless of
    # `sequence_parallel`: weight-sharding and token-sharding are
    # independent mesh axes now, so there's no more "replicate weights
    # instead" special case for sequence parallelism -- `shard_wan_params`
    # degenerates to full replication on its own whenever 'tp' has size 1,
    # so this one code path covers every combination. The VAE is
    # comparatively small and stays replicated regardless.
    replicated = get_replicated_sharding(mesh)
    dit_params = cast_to_dtype(dit_params, dtype)
    vae_params = cast_to_dtype(vae_params, dtype)
    t5_params = cast_to_dtype(t5_params, dtype)

    dit_shardings = shard_wan_params(dit_params, mesh)
    dit_params = jax.device_put(dit_params, dit_shardings)
    t5_params = jax.device_put(t5_params, shard_wan_params(t5_params, mesh))
    vae_params = jax.device_put(vae_params, replicated)
    logging.info("Weights loaded, cast, and sharded across devices.")

    # --- Prepare inputs ---
    prompts = resolve_batch_prompts(args.prompt, dp_size)
    batch_size = len(prompts)

    # Wan2.1's causal VAE compresses time by 4x (with a "+1" for the leading
    # frame) and space by 8x: see `vae_stride = (4, 8, 8)` in the reference
    # configs, and vidax.models.wan.wan2_1.vae.WanVAEDecoder's docstring.
    latent_t = 1 + (args.num_frames - 1) // 4
    latent_h = args.height // 8
    latent_w = args.width // 8
    latents_shape = (batch_size, latent_t, latent_h, latent_w, dit_model.in_dim)

    # Independent noise per batch slot -- when one prompt is broadcast across
    # multiple data-parallel replicas, this gives multiple distinct samples.
    latents_rng, rng = jax.random.split(rng)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=dtype)
    latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))

    logging.info(f"Encoding {batch_size} prompt(s) with T5: {prompts}")
    prompt_embeds = encode_prompts(prompts, t5_model, t5_params, tokenizer, dtype)
    prompt_embeds = jax.device_put(prompt_embeds, get_batch_sharding(mesh, prompt_embeds.ndim))

    # Classifier-free guidance: the reference *always* runs this (default
    # guide_scale=5.0) -- skipping it isn't just a quality knob turned down,
    # it's the difference between the amplified, prompt-aligned
    # `uncond + guide_scale * (cond - uncond)` signal the model was tuned
    # around and the raw conditional-only prediction, which on its own
    # regresses hard toward a low-contrast, washed-out "average video".
    negative_prompts = resolve_batch_prompts([args.negative_prompt], dp_size)
    negative_embeds = encode_prompts(negative_prompts, t5_model, t5_params, tokenizer, dtype)
    negative_embeds = jax.device_put(negative_embeds, get_batch_sharding(mesh, negative_embeds.ndim))

    # RoPE frequencies for the DiT's patchified (T, H, W) grid.
    pt, ph, pw = dit_model.patch_size
    head_dim = dit_model.dim // dit_model.num_heads
    freqs = create_rope3d_freqs(
        t=latent_t // pt, h=latent_h // ph, w=latent_w // pw, head_dim=head_dim)
    freqs = jax.device_put(freqs, replicated)

    # `sequence_parallel=True` reshapes/reshuffles activations across the
    # 'sp' mesh axis inside `WanDiT.__call__` itself -- these are
    # collectives, so the call needs to run inside `shard_map`, not a plain
    # `jax.jit` (which only sees ordinary per-device-local ops). `in_specs`
    # mirror the shardings already applied above: `get_batch_sharding`
    # shards the leading batch axis on 'dp' and replicates the rest,
    # `replicated` is fully replicated, and `dit_shardings` (computed above)
    # is the *real* per-leaf Megatron sharding, converted from
    # `NamedSharding` to `PartitionSpec` via `to_partition_specs` -- not a
    # blanket "fully replicated" `P()`, so weight-sharding and
    # sequence-parallel token-sharding both take effect at once. `shard_map`
    # requires the actual input array shardings to agree with what it's
    # told here. When `sequence_parallel` is off, this is just
    # `dit_model.apply` directly (the original, unmodified path).
    def _dit_apply(params, latents, t, freqs, context):
        return dit_model.apply(params, latents=latents, t=t, freqs=freqs, context=context)

    if sequence_parallel:
        dit_apply = shard_map(
            _dit_apply, mesh=mesh,
            in_specs=(to_partition_specs(dit_shardings), P('dp', None, None, None, None),
                      P('dp'), (P(), P()), P('dp', None, None)),
            out_specs=P('dp', None, None, None, None),
            check_rep=False,
        )
    else:
        dit_apply = _dit_apply

    # --- Euler sampling loop ---
    # `single_step` is jit-compiled once (t_val varies by *value*, not shape
    # or dtype, so this never recompiles) and called from a plain Python
    # loop, rather than jax.jit-ing the whole num_steps-iteration loop as
    # one fused program: unrolling every step into a single HLO program
    # means every step's intermediate activations need to coexist in that
    # program's buffer space, instead of being freed between dispatches (the
    # same issue as VAE decode's chunk loop, see the comment below -- here
    # it doesn't cause an outright OOM the way VAE decode's ~20x unroll did,
    # but it's wasted memory and compile time for no benefit, since there's
    # no cross-step fusion opportunity to actually gain from unrolling).
    # `donate_argnums=(0,)` additionally lets XLA overwrite the previous
    # step's latents buffer in place instead of allocating a fresh one.
    @partial(jax.jit, donate_argnums=(0,))
    def single_step(current_latents, step_index, prompt_embeds, negative_embeds, freqs, params, guide_scale):
        b_size = current_latents.shape[0]
        # `timesteps` are on the ~[0, num_train_timesteps] scale the model
        # was trained on (see RectifiedFlowScheduler), not the raw [0, 1]
        # flow-matching sigma -- conflating the two feeds the DiT a
        # conditioning signal ~1000x smaller than anything it saw in
        # training, producing incoherent output regardless of step count.
        t_val = scheduler.timesteps[step_index]
        t_vec = jnp.full((b_size,), t_val, dtype=jnp.float32)

        # Classifier-free guidance needs two forward passes (conditional and
        # unconditional/negative-prompt) that are identical except for
        # `context` -- batched into *one* `dit_apply` call over a `2*B`
        # batch (latents/timesteps duplicated, contexts concatenated) rather
        # than two separate dispatches, halving per-step dispatch/collective
        # overhead (`freqs` needs no duplication: its leading dim is
        # already 1 and broadcasts against any batch size). Not optional in
        # the reference pipeline -- see the comment above `negative_prompts`
        # in main().
        latents_2b = jnp.concatenate([current_latents, current_latents], axis=0)
        t_vec_2b = jnp.concatenate([t_vec, t_vec], axis=0)
        context_2b = jnp.concatenate([prompt_embeds, negative_embeds], axis=0)
        v_2b = dit_apply(params, latents_2b, t_vec_2b, freqs, context_2b)
        v_cond, v_uncond = v_2b[:b_size], v_2b[b_size:]
        velocity = v_uncond + guide_scale * (v_cond - v_uncond)
        return scheduler.step(velocity, step_index, current_latents)

    logging.info(
        f"Running sampling for {args.num_steps} steps "
        f"(shift={args.shift}, guide_scale={args.guide_scale})...")
    for step_index in range(scheduler.num_steps):
        latents = single_step(
            latents, step_index, prompt_embeds, negative_embeds, freqs, dit_params, args.guide_scale)

    # --- Decode latents to video frames ---
    logging.info("Decoding final latents into video frames...")
    # Uses `WanVAEDecoder.decode_chunk`, not a single `vae_model.apply(
    # vae_params, latents)` call: jit-ing the *whole* per-chunk loop in one
    # call would unroll all ~20 chunks into a single HLO program (each
    # chunk's conv activations needing to coexist as one giant program's
    # buffers instead of being freed between chunks -- what actually blows
    # the HBM budget at full resolution), while not jit-ing anything at all
    # means every individual op inside the decoder triggers its own,
    # separate XLA compilation (tolerable at Wan2.1's default resolution,
    # but not a good habit -- see `vidax.models.wan.wan2_2.vae.WanVAEDecoder
    # .decode_chunk`'s docstring for where this stopped being tolerable).
    # `decode_chunk` (jit-wrapped here, called from a plain Python loop)
    # avoids both: each per-frame computation is compiled as one fused
    # program, and only ever one chunk's activations are live in any given
    # jit call.
    x_full = vae_model.apply(vae_params, latents.astype(dtype), method=vae_model.pre_process)
    decode_chunk_jit = jax.jit(
        lambda params, x_chunk, cache_list: vae_model.apply(
            params, x_chunk, cache_list, method=vae_model.decode_chunk))

    decoder_cfg = Decoder3d(
        vae_model.dim, vae_model.z_dim, vae_model.dim_mult, vae_model.num_res_blocks,
        vae_model.attn_scales, vae_model.temperal_upsample, vae_model.eps)
    cache_list = [None] * _count_causal_convs(decoder_cfg)
    decoded_chunks = []
    for i in range(x_full.shape[1]):
        out_chunk, cache_list = decode_chunk_jit(vae_params, x_full[:, i:i + 1], cache_list)
        decoded_chunks.append(out_chunk)
    decoded_frames = jnp.concatenate(decoded_chunks, axis=1)

    # One output video per batch element.
    base, ext = os.path.splitext(args.output_path)
    for i in range(batch_size):
        video_frames = np.array(decoded_frames[i], dtype=np.float32)
        video_frames = np.clip(video_frames * 0.5 + 0.5, 0, 1)  # [-1, 1] -> [0, 1]
        video_frames = (video_frames * 255).astype(np.uint8)
        out_path = args.output_path if batch_size == 1 else f"{base}_{i}{ext}"
        save_video(video_frames, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end video generation with Wan 2.1 on TPU.")
    parser.add_argument("--model_size", type=str, default="1.3B", choices=list(MODEL_SIZE_CONFIGS.keys()), help="Which released Wan2.1 T2V config to build (must match --dit_checkpoint_path's actual size).")
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the DiT .safetensors checkpoint.")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True, help="Path to the VAE .pth checkpoint.")
    parser.add_argument("--t5_checkpoint_path", type=str, required=True, help="Path to the T5 (umt5-xxl encoder) .pth checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the umt5-xxl HuggingFace tokenizer directory. Defaults to '<t5_checkpoint_dir>/google/umt5-xxl'.")
    parser.add_argument("--prompt", type=str, required=True, nargs="+", help="One text prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for classifier-free guidance. Defaults to the reference's `sample_neg_prompt`.")
    parser.add_argument("--guide_scale", type=float, default=5.0, help="Classifier-free guidance scale: velocity = uncond + guide_scale * (cond - uncond). The reference's default is 5.0; skipping CFG (there is no flag to do so here, matching the reference always running it) produces washed-out, low-contrast output.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of devices to Megatron-shard each model's attention heads / FFN channels (weights) across. Must divide num_heads (12 for the 1.3B DiT, 40 for the 14B DiT, 64 for the T5 encoder) and num_devices. Composes independently with --sequence_parallel_size (see docs/hardware_and_sharding.md); num_devices must equal --tensor_parallel_size * --sequence_parallel_size * data-parallel size.")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Number of devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of --tensor_parallel_size's weight-sharding. 1 (off) by default since Megatron TP already fits the 1.3B model fine at typical resolutions; intended for the much larger 14B model at higher resolutions, where it's expected to be necessary the way it was for Wan2.2's 5B model (see WanDiT's module docstring). Also requires the DiT's patch token count to be evenly divisible by this value.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES.keys()), help="Compute dtype for the DiT, VAE, and T5 (and cast target for their loaded checkpoints). The reference uses bfloat16 for the DiT/T5 and float32 for the VAE; vidax uses one unified dtype for simplicity. Note: TPU's XLA backend does not implement float16 matmuls (a hardware/compiler limitation, not a vidax one) -- float16 will fail at runtime on TPU.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the initial noise.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps for the scheduler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Flow-matching noise-schedule shift (see RectifiedFlowScheduler). The reference's default for t2v is 5.0 regardless of resolution.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--width", type=int, default=832, help="Output video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the output video.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", help="Path to save the output MP4 video(s). With multiple prompts, each video is saved as '<output_path>_<i>.mp4'.")

    args = parser.parse_args()
    main(args)
