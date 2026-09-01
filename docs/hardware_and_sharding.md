# Hardware, Sharding & JAX Engineering Notes

Technical conventions, sharding strategy, and the debugging history behind
them. This is the "why", written down so it survives past the commit that
fixed it — several of these were found the hard way (OOMs, a 45-minute
"hang" that turned out to be a compiler issue, silently-wrong output), and
the reasoning is easy to accidentally undo by a well-intentioned refactor
that doesn't know the history.

## 1. Tensor Memory Layouts

- **JAX/XLA prefers channels-last (`NHWC` / `NDHWC`):**
  - PyTorch Conv3D `[Batch, Channels, Time, Height, Width]` → JAX Conv3D
    `[Batch, Time, Height, Width, Channels]`
  - PyTorch Linear `[Out_Features, In_Features]` → JAX Dense kernel
    `[In_Features, Out_Features]`
- `vidax.translator.converter` handles these layout conversions dynamically
  during weight ingestion, keyed off each tensor's PyTorch state_dict name
  and rank (see `convert_pt_tensor_to_jax`).
- Checkpoints are converted to **plain numpy arrays**, not `jnp.array`s:
  `jnp.array(...)` forces immediate placement on a single default device
  (unsharded). For a multi-GB param tree, converting the whole thing this
  way piles it onto one device's HBM before `jax.device_put(...,
  sharding)` ever runs — this is what was silently OOM-ing checkpoint
  loading for the 5B+ Wan2.2 DiT/T5, even though the *sharded* result
  comfortably fits across a v4-8's 4 chips. Numpy arrays live in host RAM
  instead, so `jax.device_put(param_tree, sharding_tree)` is the only point
  a device-resident copy is ever created, and it creates it already
  correctly sharded. See `convert_pt_tensor_to_jax`'s docstring.

## 2. Sharding & TPU Topology (Megatron-style tensor parallelism)

- **`vN-M` names TensorCores, not chips.** A v4 chip has 2 TensorCores, so
  `v4-8` is a 4-*chip* slice (`jax.device_count() == 4`), not 8 — this repo
  currently runs on a v4-8. `--tensor_parallel_size`/`--sequence_parallel_size`
  values throughout this repo's docs are chip counts (what
  `jax.device_count()` reports and what `tensor_parallel_size` must divide
  into), so "full width on a v4-8" means 4, not 8. v5e and v6e chips have 1
  TensorCore each, so `vN-M` there *does* equal chip count — don't assume
  the /2 rule carries over once those are benchmarked.
- `vidax.core.sharding.build_tpu_mesh` builds a 2D `(dp, tp)` device mesh:
  `dp` (data-parallel) shards the batch, `tp` (tensor-parallel) shards
  attention heads and FFN channels within each DiT/T5 layer, Megatron-1D
  style.
- `shard_wan_params` assigns the actual `NamedSharding`s: attention Q/K/V
  and the FFN up-projection are column-parallel (shard their output),
  attention-output and the FFN down-projection are row-parallel (shard
  their input — GSPMD auto-inserts the resulting all-reduce). Everything
  else (norms, embeddings, modulation) stays replicated, since it's small
  and elementwise ops against it are free regardless of the other
  operand's sharding.
- This exists because full-resolution DiT self-attention runs over tens of
  thousands of patches; without tensor parallelism the O(S² × num_heads)
  attention matrix alone can exceed a single TPU v4 chip's HBM.
  `tensor_parallel_size` must divide both `num_devices` and `num_heads` (12
  for the 1.3B DiT, 64 for the T5 encoder, 40 for the 14B DiT, 24 for
  Wan2.2's 5B DiT).

### Where Megatron TP stops being enough

Megatron TP shards *weights* (and the activations that are direct
projections of them) but keeps the **full token sequence** resident on
every device. That's fine as long as per-token activation memory is small
relative to weight memory. It stops being fine when:

- The model is wide enough (Wan2.2's 5B DiT) that per-token modulation
  tensors are themselves large, **and**
- The sequence is long enough (TI2V-5B's one supported resolution,
  704×1280×121 frames, patchifies to ~27k tokens) that "full sequence on
  every device" dominates.

At that point, quartering weight memory via 4-way Megatron TP genuinely
doesn't help, because the thing that doesn't fit was never the weights.

## 3. Sequence Parallelism (DeepSpeed-Ulysses)

`WanDiT(sequence_parallel=True)` (both Wan2.1 and Wan2.2) shards the
**token sequence itself** across the `tp` mesh axis between blocks, instead
of sharding weights. This is the DeepSpeed-Ulysses scheme
([arxiv.org/abs/2309.14509](https://arxiv.org/abs/2309.14509)), matching
the reference's own `wan/distributed/sequence_parallel.py` + `ulysses.py`.

**Mechanism**, implemented in `vidax.core.attention.sequence_parallel_self_attention`
and wired through `WanDiT`'s module docstring (`wan2_2/dit.py` has the most
detailed version, since it was built first for Wan2.2):

1. Before the block loop, chunk `x` (and, for Wan2.2 specifically, the
   per-token timestep-modulation state `e`/`e0` — Wan2.1's modulation is
   per-*sample*, not per-token, so it needs no chunking at all, a genuine
   simplification over Wan2.2) along the sequence axis. Each device now
   holds only `seq_len / sp_size` tokens for the FFN/norm/modulation-heavy
   part of every block — this is exactly the memory that was overflowing.
2. Cross-attention against the (small, already fully-replicated) text
   context runs as an ordinary local call — no reshuffle needed.
3. Self-attention needs every token to see every other token, so it
   reshuffles to a *head-sharded, full-sequence* view just for the
   attention op itself: each device already holds every head for its
   local sequence chunk (having just computed q/k/v locally); an
   `all_to_all` redistributes that into every device holding every
   sequence position for its local head chunk (a pure data reshuffle, no
   device recomputes another's tokens); local flash attention runs; a
   second `all_to_all` reshuffles back.
4. After the head, `all_gather` reassembles the full sequence before
   unpatchify.

This requires the **whole** `WanDiT.apply(...)` call to run inside
`jax.experimental.shard_map.shard_map(..., mesh=mesh)`, not just the
attention op — the chunk-before/gather-after logic needs the mesh axis
bound too. See `examples/generate_wan2_2_ti2v.py` for the wiring
(`dit_apply`, built once with `in_specs` matching the shardings already
applied to its arguments).

**Weights compose independently via `--tensor_parallel_size`** (originally
this section said SP forced weights fully replicated — that was true only
because nothing combined the two mesh axes yet; see "Combining with
Megatron TP" below for how that changed and why it was worth doing).

**CogVideoX** (`vidax.models.cogvideo.dit.CogVideoXDiT(sequence_parallel=True)`)
uses the same scheme with one twist: its attention is a single **joint**
self-attention over `[text(226); visual]` (no separate cross-attention), so
`sequence_parallel_self_attention` can't be used directly — naively
all-to-all-ing the concatenated `[text; visual_chunk]` would replicate the
text tokens `sp_size` times in the reshuffled KV.
`vidax.core.attention.sequence_parallel_joint_self_attention` handles it:
only the visual q/k/v go through the head↔sequence all-to-all, the small
replicated text q/k/v is sliced to this device's local head range and
concatenated in before one local flash-attention call over
`[text(full); visual(full)]`, then the visual output is reshuffled back and
the text output is `all_gather`ed over the head axis. Per-sample (not
per-token) AdaLN modulation means only the visual token sequence and the
RoPE tables need chunking — nothing analogous to Wan2.2's `e0`. CogVideoX
keeps SP and Megatron TP **mutually exclusive** (`generate_cogvideox.py`
asserts `--tensor_parallel_size 1` under `--sequence_parallel_size`): the 5B
DiT fits replicated per chip in bf16, so none of the column/row-parallel
shape juggling below is threaded through `CogVideoXDiT`. This is what runs
CogVideoX-1.5 at its native 1360×768 (~45k visual tokens, which the non-SP
graph never finished compiling) — see
[`docs/lessons/cogvideox_debugging.md`](lessons/cogvideox_debugging.md).

**A real bug found here:** `attend()`'s i2v CLIP image cross-attention
branch always dispatched through the mesh-based `dot_product_attention`
regardless of `sequence_parallel` — which would break inside `shard_map`
(the same "already inside a sharded body, but the dispatch heuristic can't
tell" issue as ordinary cross-attention, see §4 below). Fixed by routing it
through the same local, non-mesh-dispatched path as text cross-attention.
Verified numerically identical to the non-SP path on real TPU hardware for
both t2v and i2v (including this CLIP branch) using synthetic-dimension
models.

### Combining with Megatron TP

Wan2.2 A14B (14B params, two MoE experts) is heavy enough that neither
scheme alone fits this repo's 4-chip machine at real resolutions: Megatron
TP shards weights but not the (large, per-token-modulated) activations;
sequence parallelism shards activations but replicates weights, and one
14B expert unsharded is already most of a chip's HBM budget by itself.
`--tensor_parallel_size` and `--sequence_parallel_size` now compose freely
instead of being mutually exclusive — `vidax.core.sharding.build_tpu_mesh`
builds a 3-axis `(dp, tp, sp)` mesh (previously 2-axis, with SP overloading
the `tp` axis name for token-chunking), and every `sequence_parallel`
example script's `dit_apply` now feeds `shard_map`'s `in_specs` the *real*
per-leaf Megatron sharding (`to_partition_specs(shard_wan_params(...))`)
instead of a blanket "fully replicated" `P()` for the params argument.
`--sequence_parallel_size 1` (the default) is a complete no-op for
everything below — nothing changes for callers that don't ask for this.

Two things had to be fixed to make this actually correct, not just wired
up — both stem from the same root cause: **GSPMD's auto-partitioner
doesn't run inside `shard_map`.** Outside it (plain Megatron TP, no SP),
GSPMD transparently reconciles a layer's *declared* (global) shape against
its *physically* Megatron-sharded weight — the model code stays written in
global-shape terms, and the compiler figures out the rest. Inside
`shard_map`, there is no such magic: every op sees the true local array,
and Flax's `self.param(...)` (used by both `nn.Dense` and `RMSNorm`)
validates the existing weight's actual shape against a *static* Python
shape argument you give it — if that argument still says the old global
width, `shard_map` raises a shape mismatch immediately.

1. **Column-parallel Dense layers** (`self_attn_q/k/v`, `cross_attn_q/k/v`,
   `ffn_0`/`mlp_layer1`) now pass `features = global_width // tp_size`
   instead of the global width, whenever `sequence_parallel` is set
   (`tp_size` read from `mesh.shape['tp']`, a static Python int at trace
   time — 1, a no-op, when TP isn't in use). Row-parallel layers
   (`self_attn_o`/`cross_attn_o`, `ffn_2`/`mlp_layer2`) need no shape
   change: Flax infers a Dense layer's *input* width from the actual local
   array already, only its *output* width is a static argument, and that
   correctly stays the full global width (matching the value after the
   manual `psum` below).
2. **The manual all-reduce row-parallel layers need.** Outside `shard_map`,
   GSPMD auto-inserts the cross-device sum a row-parallel layer's output
   needs; inside it, nothing does, so every row-parallel output projection
   and FFN down-projection now sums its per-device partial output by hand
   whenever `sequence_parallel` is set — a no-op when `'tp'` has size 1.
   `vidax.models.cosmos2_5.dit_layers.cosmos_attend` calls plain
   `jax.lax.psum(x, 'tp')` directly (its row-parallel Dense layers are
   `use_bias=False`); `vidax.models.wan.common.dit_layers.attend` and both
   Wan DiTs' FFN down-projection instead call
   `vidax.models.wan.common.dit_layers.psum_row_parallel`, a thin wrapper
   around the same `psum` that also corrects for `nn.Dense`'s replicated
   bias otherwise getting summed once per `tp`-way device instead of once
   — see
   [`docs/lessons/wan2_1_debugging.md`](lessons/wan2_1_debugging.md#row-parallel-psum-double-counted-nndenses-bias-under-sequence_parallel)
   for why Wan needs the extra correction and Cosmos doesn't.
3. **Wan's Q/K-RMSNorm needed more than a shape fix.** It normalizes over
   the *entire* projected `dim` (before splitting into heads) — under TP
   sharding, that axis is now split across devices, so a naive per-device
   local mean-square would be a mathematically different (wrong) number
   from the reference's, not just a differently-shaped one. Fixed with a
   new `vidax.core.attention.TPShardedRMSNorm`: sums each device's local
   sum-of-squares via `jax.lax.psum('tp')` before normalizing, applying
   this device's own (now Megatron-sharded, via new `COLUMN_PARALLEL_NAMES`
   entries in `shard_wan_params`) local slice of the scale parameter.
   Cosmos's Q/K-RMSNorm needed no equivalent: it normalizes per-*head*
   (over `head_dim`, after splitting into heads), and Megatron sharding
   always keeps whole heads on one device, so that axis is never split.

**Verified two ways.** First, numerically: a small synthetic-dimension
`WanDiT`/`CosmosDiT`, real random weights, forward pass computed once
unsharded as ground truth, then again under `(tp=4,sp=1)` (regression),
`(tp=1,sp=4)` (regression), and `(tp=2,sp=2)` (new) on this repo's real 4
chips — all three matched the ground truth (Wan: bit-exact; Cosmos: matched
each other and the ground truth to the same small floating-point-
associativity tolerance real multi-device reductions always carry, not a
tolerance specific to the new combined path). Second, end-to-end against
real A14B checkpoints: `--tensor_parallel_size 2 --sequence_parallel_size 2`
ran correctly (including the two-expert boundary switch) at a noticeably
larger resolution than either trick alone reached on this 4-chip machine,
though still short of the reference's full 720x1280x81 — see
[`docs/models/wan2_2.md`](models/wan2_2.md)'s A14B sections for the actual
numbers.

Wan2.1's i2v CLIP image-cross-attention (`image_context`) isn't threaded
through the column-parallel halving above — combining it with TP weight-
sharding under `sequence_parallel` would give `k_img`/`v_img` a different
head count than `q` (they aren't Megatron-sharded at all, being small and
outside `COLUMN_PARALLEL_NAMES`). `attend()` raises `NotImplementedError`
rather than silently computing something wrong if this combination is
attempted; use one parallelism scheme or the other for i2v with CLIP
conditioning, or `--tensor_parallel_size 1` alongside
`--sequence_parallel_size` (or vice versa).

## 4. Flash Attention (TPU)

- `vidax.core.attention.dot_product_attention` dispatches to a real Pallas
  flash-attention kernel (`jax.experimental.pallas.ops.tpu.flash_attention`)
  on TPU whenever no `bias`/`mask` is given — i.e. for DiT's self- and
  cross-attention. **This matters**: `jax.nn.dot_product_attention`'s
  default `"xla"` implementation, and the `jax.nn` API in general, has no
  automatic TPU flash-attention path — it always fully materializes the
  `(B, num_heads, S_q, S_k)` logits matrix, which for DiT self-attention
  over tens of thousands of video patches is the dominant memory cost
  (well beyond what tensor parallelism alone divides down to a usable
  size). T5's self-attention keeps its relative-position `bias` and stays
  on the materializing path, since its sequence length is small and fixed
  (512) and doesn't need flash attention's O(S) memory anyway.
- Mosaic (Pallas TPU) kernels are opaque custom calls that **GSPMD cannot
  auto-partition** — running one on a sharded array of any kind
  (tensor-parallel *or* plain data-parallel-batched) raises `"Mosaic
  kernels cannot be automatically partitioned"` unless it's explicitly
  wrapped in `jax.experimental.shard_map.shard_map`. `dot_product_attention`
  does this itself given a `mesh` argument (threaded through
  `WanDiT`/`WanDiTBlock` as a `mesh` field) — every caller running on more
  than one device must pass one, or the call falls back to the (correct,
  just slower/O(S²)) XLA path.
- Under `sequence_parallel`, `local_attention`/`sequence_parallel_self_attention`
  bypass `dot_product_attention`'s own dispatch heuristics entirely and call
  the local flash-attention primitive directly — those heuristics check
  `jax.device_count() > 1` with no `mesh` given and would otherwise pick the
  slow XLA path, since they can't tell they're already running inside a
  `shard_map`-sharded body.
- **The `shard_map` requirement is per-*program*, not per-call.** Once
  *any* part of a `jax.jit`-compiled function is multi-device
  GSPMD-partitioned (true the moment any of its params carry TP sharding
  at all), *every* Pallas/Mosaic call anywhere in that same function needs
  a `shard_map` wrapper — including calls whose own operands are fully
  replicated. HunyuanVideo-1.5's port hit this directly: its DiT's
  double/single-stream blocks' attention was correctly wrapped for their
  TP-sharded Q/K/V, but the small text-refiner submodule (`SingleTokenRefiner`,
  intentionally left un-sharded — a fused-QKV Dense's contiguous TP split
  doesn't align with clean per-head boundaries) crashed with the same
  `NotImplementedError` anyway, since it shares the same jitted program as
  the (now-partitioned) main blocks. Fixed with a second, fully-replicated
  `shard_map` wrapper for that caller specifically (`PartitionSpec()` on
  every arg — a per-device no-op replica, matching how its already-
  replicated activations are actually laid out) rather than skipping
  `shard_map` for it — see `vidax.models.hunyuan_video.common.dit_layers
  ._flash_attention_tpu_segment_masked_replicated` and
  `docs/lessons/hunyuan_video1_5_debugging.md`.

## 5. JIT Compilation Safety

- Keep sequence lengths, frame counts, and spatial dimensions static or
  explicitly padded.
- Avoid dynamic Python branching on array values inside functions decorated
  with `@jax.jit`.
- **Don't `jax.jit` a Python loop over many repeated forward passes**
  (diffusion sampling steps, VAE decode's per-latent-frame chunks). `jax.jit`
  traces the loop by fully unrolling it into *one* HLO program, so every
  iteration's intermediate buffers can end up needing to coexist in that
  program's memory footprint instead of being freed between iterations —
  this is what caused whole-video VAE decode (~20 latent-frame chunks) to
  OOM even though DiT sampling (a much bigger model) had already succeeded.
  Instead, jit only the per-iteration function (its shape/dtype signature
  is identical every call, so this never recompiles) and call it from a
  plain Python loop: the sampling loops in all three `examples/generate_*`
  scripts do exactly this (with `donate_argnums` on the latents carry, so
  each step reuses the previous step's buffer in place).

### The VAE decode "hang" that wasn't a hang

`WanVAEDecoder`/`WanVAEEncoder`'s `decode_chunk`/`encode_chunk` methods
exist because of a real incident, not a hypothetical: a first attempt at
running Wan2.2 TI2V-5B decode appeared to hang for 45+ minutes. `py-spy`
attached to the running process showed it stuck in `backend_compile_and_load`
— genuine XLA compilation, not a deadlock. Running the decoder eagerly
(calling `.apply()` directly, one op at a time, no `jax.jit` anywhere) means
**every individual op** inside the decoder triggers its own separate XLA
compilation. This is tolerable at Wan2.1's default resolution and channel
width (384 at the widest) but not at all tolerable for Wan2.2's much
wider, deeper decoder (1024 channels at the bottleneck) at TI2V-5B's one
supported resolution.

The fix (`WanVAEDecoder`/`WanVAEEncoder` converted from `@nn.compact` to
`setup()`-based, exposing `pre_process`/`decode_chunk` (or
`pre_process`/`encode_chunk`/`post_process`) as independently callable
methods) lets the *whole* per-frame computation be jit-compiled as one
fused program, called from a plain Python loop over latent frames — at
most 2-3 distinct compiles total (the cache state's pytree structure
stabilizes after the first frame or two), reused for every remaining chunk,
instead of one compile per op per frame. See `WanVAEDecoder.decode_chunk`'s
docstring in either `wan2_1/vae.py` or `wan2_2/vae.py` for the full
reasoning, and `WanVAEEncoder.encode_chunk`'s docstring for the encode-side
mirror (used by I2V's conditioning-image path, which — for Wan2.1 — encodes
a full, mostly-zero, `num_frames`-long video, not just the one real frame,
so it hits the same ~20-chunk loop).

A prerequisite fix for jit-compatibility: the reference's causal-conv cache
uses a `"Rep"` string sentinel (meaning "no real history yet, this is the
first temporal-upsample call") mixed into an otherwise array-valued cache
list. A raw Python string can't cross a `jax.jit` boundary as a traced
argument. Replaced with a **zero-length-along-time** real array — value-
preserving in eager mode too (verified by regression test), and now
JIT-compatible, since JAX's `jax.jit` already handles `None`-containing
pytree arguments and automatically compiles/caches a separate variant per
distinct structure — the only actual blocker was the string.

### The double-buffering OOM (checkpoint casting)

A second, related bug: casting the DiT checkpoint to `bfloat16` *after*
`jax.device_put` (the natural-looking order) needs the pre-cast (float32)
and post-cast (bfloat16) copies to coexist on-device for the duration of
the cast op. Wan2.2's DiT ships as raw float32 — 5B params × 4 bytes = 20GB,
and under sequence parallelism that's **replicated per device** (not a
`1/tp_size` Megatron shard). 20GB (float32) + 10GB (bfloat16, transiently
live) is already past a v4 chip's ~30.75GB HBM budget before accounting for
anything else. Fixed by casting on the host (`cast_numpy_tree_to_dtype`,
operating on the numpy pytree) **before** `device_put` — only the
already-small target-dtype array is ever placed on a device at all.

### i2v resolution-divisibility

`--tensor_parallel_size`'s exact-divisibility requirement (the DiT's patch
token count must divide evenly for sequence-parallel chunking) is
guaranteed by construction at Wan2.2 TI2V-5B's fixed 704×1280 t2v
resolution, but **not** for i2v: an arbitrary input image's aspect ratio
gives no guarantee its derived patch token count divides evenly by
`tensor_parallel_size`. `generate_wan2_2_ti2v.py` grows the derived width
in 32px steps (up to `tensor_parallel_size - 1` times — guaranteed to find
a divisible value, since that many consecutive integers cover every
residue mod `tensor_parallel_size`) and logs when it does, rather than
failing outright.

### Host-transfer for decoded chunks

Verifying i2v end-to-end (real checkpoint, bundled example image, 4-way
sequence parallelism) surfaced one more device-memory bug: the final
`jnp.concatenate` of all ~31 decoded chunks needed its own large
contiguous on-device allocation, which no longer fit once the conditioning
image's encoded latent and masks were *also* device-resident alongside the
5B DiT/T5/VAE params. Fixed by moving each decoded chunk to the host
immediately after decoding, instead of keeping all of them device-resident
for one big on-device concatenate at the end.

## 6. Model-specific debugging postmortems

Bug-by-bug narratives for specific models (Cosmos-Predict2.5's real-
checkpoint bugs, the Wan2.1 fp32/bf16 precision corruption) have moved to
[`docs/lessons/`](lessons/) to keep this file focused on general, reusable
sharding/JIT/dtype engineering conventions rather than growing without
bound as more models are ported. See:

- [`docs/lessons/cosmos2_5_debugging.md`](lessons/cosmos2_5_debugging.md) —
  a "grid of random colors" output symptom traced to a wrongly-added EDM
  preconditioning wrapper borrowed from a reference class this checkpoint's
  training config never actually uses, and the real-photo low-noise
  denoising probe that found it.
- [`docs/lessons/wan2_1_debugging.md`](lessons/wan2_1_debugging.md) —
  three compounding bugs (checkpoint weight rounding, `compute_dtype`/
  `dit_dtype` decoupling, latents/output re-quantization) behind severely
  corrupted Wan2.1 I2V output at large token counts and why it only showed
  up at scale, plus the row-parallel `psum` double-counting `nn.Dense`'s
  bias under `sequence_parallel` in code Wan2.1 and Wan2.2 share (found
  while combining offloading with sequence parallelism on A14B).
- [`docs/lessons/cosmos3_debugging.md`](lessons/cosmos3_debugging.md) —
  Edge silently running at Nano's resolution/scheduler defaults (wrong
  config, no error, just degraded output), and both models needing
  JSON-structured prompts for good quality.
- [`docs/lessons/ltx_video_debugging.md`](lessons/ltx_video_debugging.md) —
  two bugs caught by bit-exact numerical comparison against the actual
  reference PyTorch implementation (a throwaway conda env, not the main
  dev env) before any end-to-end generation was even attempted: the VAE's
  top-level `patchify`/`unpatchify` using a different width/height merge
  order than its own internal `PixelShuffleND`, and the decoder's
  `causal_decoder=False` config meaning symmetric (not causal) temporal
  padding.
- [`docs/lessons/ltx2_5_debugging.md`](lessons/ltx2_5_debugging.md) —
  findings from porting LTX-2.5 that generalize beyond this one model: a
  fully-fused 48-block `jax.jit` trace not freeing per-block activations
  (and per-layer weight offloading fixing that as a side effect), AdaLN
  modulation tables shipped at float32 in an otherwise-bf16 checkpoint, a
  periodic VAE-decoder artifact that turned out checkpoint-inherent rather
  than a porting bug, a silently under-scaled conditioning signal that
  produced prompt-disconnected video with no crash to flag it, and the
  multi-stage memory/compile-time story behind the NATTEN-based diffusion
  VAE decoder.
- [`docs/lessons/cogvideox_debugging.md`](lessons/cogvideox_debugging.md) —
  bf16 T5-XXL going 16–37% off when the encoder is run without an attention
  mask (CogVideoX passes none), the VAE's stateful spatial-tiling blend, and
  why CogVideoX's single joint `[text; visual]` attention needs a bespoke
  DeepSpeed-Ulysses variant.
- [`docs/weight_offloading.md`](weight_offloading.md) — per-layer weight
  offloading (host RAM, streamed into a small fixed-shape HBM buffer one
  block at a time), implemented for every DiT in this repo that doesn't fit
  fully device-resident at some resolution; the measured throughput cost
  and when it's actually worth reaching for.

## Summary of hard-won lessons

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Checkpoint loading OOMs on a 5B+ model | `jnp.array(...)` in the converter forces unsharded single-device placement | Convert to numpy, not JAX arrays; only `device_put` creates device-resident (sharded) copies |
| Sampling OOMs even after checkpoint loads | Megatron TP shards weights, not the token sequence; Wan2.2's per-token state doesn't shrink | Sequence parallelism (DeepSpeed-Ulysses) — shard activations instead |
| VAE decode "hangs" for 45+ minutes | Eager (unjitted) execution compiles every op separately | `decode_chunk`/`encode_chunk`, jit-compiled once per chunk-shape, called from a Python loop |
| `jax.jit` fails on a cache list | A `"Rep"` string sentinel isn't a valid traced pytree leaf | Replace with a zero-length real array (value-preserving) |
| DiT weight cast OOMs on-device | float32 + bfloat16 copies coexist during an on-device cast | Cast on the host (numpy) before `device_put` |
| i2v sequence-parallel chunking fails for some images | Image-derived resolution doesn't guarantee divisible token count | Grow width in 32px steps until divisible |
| Final decode OOMs only when i2v conditioning is also present | On-device concatenate of all chunks competes with other device-resident state | Move each chunk to host immediately, concatenate on host |
