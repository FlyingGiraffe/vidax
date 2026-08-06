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

**Weights are replicated, not Megatron-sharded, under sequence
parallelism**: since SP shards activations, not weights, every device
needs its own complete copy of every DiT weight. This is a real trade-off
(more per-device weight memory) made in exchange for cutting the
activation memory that was actually overflowing.

**A real bug found here:** `attend()`'s i2v CLIP image cross-attention
branch always dispatched through the mesh-based `dot_product_attention`
regardless of `sequence_parallel` — which would break inside `shard_map`
(the same "already inside a sharded body, but the dispatch heuristic can't
tell" issue as ordinary cross-attention, see §4 below). Fixed by routing it
through the same local, non-mesh-dispatched path as text cross-attention.
Verified numerically identical to the non-SP path on real TPU hardware for
both t2v and i2v (including this CLIP branch) using synthetic-dimension
models.

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
