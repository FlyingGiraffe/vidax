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
   needs; inside it, nothing does, so `vidax.models.wan.common.dit_layers
   .attend` and `vidax.models.cosmos2_5.dit_layers.cosmos_attend`
   (their shared output projection) and every DiT block's FFN down-
   projection now call `jax.lax.psum(x, 'tp')` by hand whenever
   `sequence_parallel` is set — a no-op when `'tp'` has size 1.
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

## 6. Cosmos-Predict2.5: bugs that only surfaced against real checkpoints

The Cosmos-Predict2.5 2B DiT/VAE/UniPC-scheduler/Reason1-text-encoder port
was initially built and verified with a mix of real-weight shape/forward
checks and synthetic conditioning (no local Reason1 checkpoint existed yet).
Once all three real checkpoints (DiT, VAE, and — from a separate
`nvidia/Cosmos-Reason1-7B` download — the text encoder) became available and
the actual `generate_cosmos2_5.py` script was run end-to-end against them,
six real bugs surfaced that no amount of architecture-level review or
synthetic-weight testing had caught. None of them were exotic — every one
is a good instance of a class of bug worth watching for generally.

### `apply_chat_template` returning a `BatchEncoding`, not a bare list

`Reason1Tokenizer.__call__` called
`tokenizer.apply_chat_template(conversation, tokenize=True,
add_generation_prompt=False)` expecting a plain `list[int]` back (matching
older `transformers` behavior/most examples in the wild). Against the
`transformers` version actually installed, this returns a `BatchEncoding`
(dict-like, `{"input_ids": [...], "attention_mask": [...]}`) instead —
`ids + [pad_id] * n` then fails with a `TypeError` (`BatchEncoding` doesn't
support `+` with a list). Fixed by unwrapping `ids["input_ids"]` when the
return value is dict-like. A synthetic-tokenizer test would never catch
this — it only shows up against a real `AutoTokenizer.from_pretrained(...)`.

### RoPE dtype: float32 cos/sin tables silently upcast q/k but not v

Reason1's RoPE (`_apply_rotary_pos_emb` in
`vidax.models.cosmos2_5.reason1`) computed `cos`/`sin` tables in
float32 (needed for numerical range/precision) and applied them via `x *
cos + rotate_half(x) * sin` *without* casting the result back to `x`'s
original dtype. Under bf16 (the real checkpoint's native dtype), this
silently upcast `q`/`k` to float32 after RoPE while `v` — never touched by
RoPE — stayed bf16. `jax.nn.dot_product_attention` hard-requires `q`/`k`/`v`
to share one dtype and raises (`"value dtype should be float32, but got
bfloat16"`) rather than promoting for you. Every earlier synthetic test used
float32 throughout, so this never triggered. Fixed by explicitly casting
back to the input's original dtype at the end of `_apply_rotary_pos_emb`
(the same convention `vidax.core.rope3d.apply_rope3d` and
`vidax.models.cosmos2_5.rope.apply_cosmos_rope3d` already followed
correctly — this was an inconsistency between two independently-written
RoPE implementations in the same repo, not a new idea to invent).

### `UniPCState` wasn't a registered JAX pytree

`vidax.schedulers.unipc.UniPCState` is a plain `@dataclasses.dataclass` —
which is *not* automatically a JAX pytree. Passing one as an argument to a
`jax.jit`-wrapped function (`single_step` in `generate_cosmos2_5.py`) fails
immediately: `"...was not marked as static using the static_argnums..."`.
Every earlier test of the scheduler called `.step()` eagerly, outside any
`jax.jit`, so this never came up until the real end-to-end script (which
jits the whole per-step DiT-forward-plus-scheduler-step for speed, like
every other example script in this repo) was actually run. Fixed with
`jax.tree_util.register_dataclass(UniPCState, data_fields=[...],
meta_fields=["this_order"])` — `this_order` specifically has to be a *meta*
(static, hashed-not-traced) field, not a data field, because it's consumed
by genuine Python-level branching inside the predictor/corrector math (see
the next bug for why that distinction matters).

### UniPC's `step_index` needs `static_argnums`, unlike Wan's Euler scheduler

Once `UniPCState` could cross the `jit` boundary, the next failure was a
`step_index` concretization error. Wan's `RectifiedFlowScheduler.step` only
ever *indexes* an array with `step_index` (`self.sigmas[step_index]`), which
works fine whether `step_index` is a concrete Python int or a traced value —
so the existing example scripts never needed to mark it static.
`FlowUniPCMultistepScheduler.step`, by contrast, has genuine Python-level
control flow keyed on `step_index`'s *value* (`if step_index > 0:` to decide
whether to run the corrector, `min(self.num_steps - step_index, ...)` for
the `lower_order_final` ramp) — tracing it as an abstract value hits
`jax.jit` head-on. Fixed by adding `step_index` to `static_argnums` in
`generate_cosmos2_5.py`'s `single_step`. Costs at most `num_steps`-many
retraces per run (one per distinct `step_index`), the same cost class the
per-value-not-per-shape reasoning elsewhere in this repo already accepts.

### A float32 conditioning mask silently upcast the whole DiT to float32

`generate_cosmos2_5.py` built its image2world/video2world conditioning mask
(`cond_mask`/`cond_mask_full`) as plain float32 (convenient for the
timestep-blending arithmetic elsewhere), then passed it straight into
`CosmosDiT`'s `condition_video_mask` input and into the sampling loop's
latent re-clamp arithmetic — both of which mix it with bf16 tensors
(`latents`, VAE-encoded conditioning frames). `jnp.concatenate` and ordinary
elementwise ops silently promote to the *widest* operand dtype, so the
DiT's entire input `x` (and therefore every downstream activation) became
float32 for the whole forward pass, and the sampling loop's `latents`
buffer's dtype flipped from bf16 to float32 after the very first
conditioning-frame re-clamp (which would additionally have broken
`single_step`'s `donate_argnums=(0,)`, which requires a donated buffer's
dtype to stay fixed across calls). The failure that actually surfaced first
was indirect and confusing: a cross-attention dtype mismatch between query
(derived from the now-float32 `x`) and key (derived from `context`, which
was never touched by this bug and stayed properly bf16) — nothing about the
error pointed at the mask. Fixed two ways: defensively inside
`CosmosDiT.__call__` (cast `padding_mask`/`condition_video_mask` to
`latents.dtype` before concatenating, protecting any future caller) and at
the source in the script (build `cond_mask_full` in the target compute
`dtype` directly, keeping a separate float32 `cond_frame_mask` only for the
timestep arithmetic that actually needs it).

### Conflating the VAE's compression with the VAE+patch combined compression

The image2world/video2world resolution-derivation code computed the
*latent tensor's* spatial size as `pixel_size // 16`, copying Wan2.2 TI2V's
own `dw, dh = pw * 16, ph * 16` divisibility-target constant verbatim. That
constant is correct *for Wan2.2*, whose own VAE already compresses spatially
16x (an 8x causal encoder plus an internal 2x pixel-patchify wrapper) before
the DiT's patch_size divides it further — but Cosmos-Predict2.5 reuses
**Wan2.1's** VAE, which only compresses 8x; the DiT's own `patch_size=(1,2,2)`
is a separate, later factor that must *not* be pre-baked into the latent
tensor's own shape (it's applied internally, inside `CosmosDiT.__call__`,
to the already-VAE-compressed latent). The bug was silent for pure
text2video (no crash — the script just self-consistently allocated,
sampled, and decoded a latent tensor at *half* the intended resolution,
producing a 64x64 video when 128x128 was requested, with nothing to flag
the discrepancy) and only turned into a hard crash for image2world, where
the independently-VAE-encoded conditioning frame (correctly at `pixel/8`)
no longer matched the wrongly-`pixel/16`-sized latent tensor it was being
inserted into. Fixed by using `pixel_size // 8` (the VAE's actual stride)
for the latent tensor's shape, and `pw * 8, ph * 8` (not `* 16`) for the
pixel-resolution divisibility target, so the resulting latent grid still
comes out evenly divisible by `patch_size` as intended. The lesson: a
silently-wrong-but-self-consistent resolution is much harder to catch than
a crash — this one shipped through an earlier real-checkpoint text2video
smoke test undetected because nothing in that test compared the *output*
shape against the *requested* one.

## 7. Cosmos-Predict2.5: "output is a grid of random colors" (real diffusion bugs)

Everything in section 6 was found by running the pipeline and checking for
crashes/NaNs/shapes — none of it touches whether the *generated video
actually looks right*, since none of those checks decode and look at a
frame. Once a full real-checkpoint run was visually inspected for the first
time, the output was a rigid, perfectly regular grid of small scrambled
color blobs — not noise, not a blurry-but-recognizable scene, a literal grid.
Diagnosing this took a different kind of verification than section 6's bugs:
line-by-line comparison against the actual reference PyTorch source (not
just the earlier architecture research notes), plus a series of ablations
run against the real checkpoint to narrow down which stage of the pipeline
was responsible, since none of the individual pieces (RoPE, attention
dispatch, checkpoint mapping) showed anything wrong under isolated unit
tests with synthetic inputs.

### Bug 1: `unpatchify`'s channel order isn't the inverse of `patchify`'s

The reference's `PatchEmbed` flattens each patch as `"b c (t r)(h m)(w n)
-> b t h w (c r m n)"` (channel outermost, then temporal-patch, height-
patch, width-patch) — this port replicated that correctly. But the
reference's `unpatchify` uses a *different, non-symmetric* order:
`"B T H W (p1 p2 t C) -> B C (T t)(H p1)(W p2)"` — height-patch, width-patch,
temporal-patch, **channel innermost**. An earlier version of this code
assumed unpatchify was simply patchify's inverse (same channel order) —
a reasonable-looking assumption that happens to be wrong for this specific
reference implementation, and was explicitly flagged as an unconfirmed risk
in `CosmosDiT`'s module docstring before it was checked against source.
Getting this wrong scrambles which of the final layer's 64 output values
maps to which (channel, row-in-patch, col-in-patch) position — every patch
still decodes to *something* locally smooth (each patch's own values are
self-consistent), but adjacent patches don't relate to each other in image
space at all, which is exactly what a literal grid of unrelated blobs looks
like. **Necessary but not sufficient** — fixing this alone made only a
modest per-pixel numerical difference and the output still looked like the
same kind of grid, which in hindsight should have been the tell that a
second, larger bug was still present (see Bug 3).

### Bug 2: `single_step` recompiled the entire 2B-parameter DiT every sampling step

`step_index` was passed as a `static_argnums` of the single `jax.jit`-wrapped
function that *also* contained both DiT forward passes (conditional +
unconditional) — necessary because UniPC's `step()` has genuine Python-level
branching on `step_index`'s value (unlike Wan's Euler scheduler, which only
ever indexes an array with it). Marking any argument of a jitted function
static forces a full retrace — and recompile — of the *entire* function
every time that argument's value changes, not just the small piece of code
that reads it. Since `step_index` changes every step, this meant `num_steps
* 2` full recompiles of a 28-block 2B-parameter model per run (confirmed:
30 low-resolution steps took 16+ minutes wall-clock before the fix, almost
entirely compile time). Fixed by splitting the per-step work: the DiT
forward pass (`compute_velocity`) is its own `jax.jit` with *no*
static/step-dependent argument at all (compiles exactly once, reused for
every step), while `scheduler.step(...)`'s cheap UniPC arithmetic runs
eagerly outside any `jit`, where a static `step_index` costs nothing.

### Bug 3 (the dominant one): missing EDM-style preconditioning wrapper

Cosmos-Predict2.5 is trained with `RectifiedFlowScaling` (`cosmos_predict2/
_src/imaginaire/modules/denoiser_scaling.py`), an EDM-style preconditioning
wrapper around the raw network, confirmed directly in the reference's
`denoise()` (`models/text2world_model.py`):

```python
t = sigma / (sigma + 1)                      # NOT sigma itself
c_skip, c_out, c_in = 1 - t, -t, 1 - t
c_noise = t * t_scaling_factor                # t_scaling_factor = 1.0 (confirmed default, unset for this checkpoint)
net_output = net(xt * c_in, timesteps=c_noise, ...)   # scaled input, remapped timestep
x0_pred = c_skip * xt + c_out * net_output            # NOT `xt - sigma * net_output`
```

This repo's implementation never applied any of it: the raw noisy latent
was fed to the DiT unscaled, and `sigma * num_train_timesteps` (Wan's own
convention, `[0, 1000]`) was used as the timestep instead of `c_noise`
(`t`, which is in `[0, 1)`) — roughly a **1000x scale error** on the
timestep the model was told it was at, at every single step of every
sampling run, plus a missing input rescale and a wrong raw-output-to-x0
combination formula. A sinusoidal timestep embedding is extremely sensitive
to the absolute scale of its input; being off by ~1000x means the model
received essentially arbitrary, out-of-distribution noise-level
conditioning throughout the entire trajectory, and could never correctly
gauge how much to denoise. This was flagged as an open question in the
architecture's original research notes ("confirm exact default
`t_scaling_factor` if bit-exactness matters") but was never actually
resolved before this port was first wired end-to-end — a reminder that a
documented uncertainty left unresolved is still a live bug, not a footnote.

Fixed by implementing the wrapper in `generate_cosmos2_5.py`'s
`compute_velocity` (sampling-loop orchestration, not DiT architecture — same
reasoning as why Wan's `y`/mask construction lives in its example scripts,
not its model file), applied per-*frame* (broadcasting `t`/`c_in` etc. over
`(B, T, 1, 1, 1)`) so image2world/video2world's differently-noised
conditioning frames still get their own correct preconditioning. Since
`FlowUniPCMultistepScheduler.step` internally assumes the simpler plain
flow-matching convention (`x0 = sample - sigma_t * model_output`) and wasn't
worth complicating just for this, the value passed to it as "model_output"
is instead back-solved algebraically so that formula reproduces the correct
EDM `x0_pred` exactly: `model_output := (xt + net_output) / (1 + sigma)`
(substitute into `sample - sigma_t * model_output` and simplify — it reduces
to precisely `c_skip * xt + c_out * net_output`).

**Effect confirmed real and large**: before this fix, output was a rigid,
perfectly regular grid at every resolution/step-count/guidance-scale tried.
After, the grid artifact is completely gone, replaced by a qualitatively
different, spatially organic (if, as of this writing, not yet fully
photorealistic) texture — confirming the fix's *direction* and *mechanism*
are correct. Full convergence to a clearly recognizable scene wasn't
achieved even at the model's native 704x1280 resolution and the reference's
own 35-step/`guide_scale=7` defaults in the runs done so far; whether that
needs further investigation (a remaining smaller bug), more steps, or is
simply a real limitation of testing with `bfloat16` weights and a handful of
frames is not yet resolved — see [`docs/models/cosmos2_5.md`](models/cosmos2_5.md)'s
status section for the current, honest state.

### A note on diagnostic method

Several plausible-looking hypotheses were tested and ruled out along the
way, each cheaply (via ablation against the real checkpoint) before
committing to a deeper investigation: VAE decoding pure random noise (rules
out the VAE decoder itself — its output looks like fine chaotic static, not
a regular grid); running with `--tensor_parallel_size 1` vs `4` (rules out
the Megatron-sharded flash-attention dispatch path); `--guide_scale 1.0`
(rules out classifier-free-guidance amplification of a miscalibrated
unconditional branch); and directly instrumenting a real block's attention
score matrix and AdaLN gate values with real weights (inconclusive on its
own, but useful groundwork). The bug was ultimately found not by any single
clever test but by re-reading the actual reference source end-to-end a
second time, line by line, rather than trusting an earlier research
summary's paraphrase of it — the summary wasn't wrong about the *existence*
of the preconditioning wrapper (it's described in section 4.1 of this
repo's original Cosmos architecture research), but the port's implementation
never actually followed through on it.

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
| Reason1 tokenizer crashes with a `TypeError` on `+` | `apply_chat_template` returns a `BatchEncoding`, not a bare list, in the installed `transformers` version | Unwrap `ids["input_ids"]` when the return value is dict-like |
| Reason1 cross-attention raises a q/k/v dtype mismatch under bf16 | RoPE's float32 cos/sin tables upcast q/k but the result was never cast back | Cast RoPE output back to the input's original dtype |
| `single_step` raises "not marked as static" the first time UniPC is jitted | `UniPCState` is a plain dataclass, not a registered JAX pytree | `jax.tree_util.register_dataclass(...)`, with `this_order` as a static meta field |
| UniPC raises a concretization error on `if step_index > 0` | Unlike Euler, UniPC's `step()` has real Python control flow keyed on `step_index`'s value | Add `step_index` to `static_argnums` |
| Cross-attention dtype mismatch that has nothing to do with cross-attention | A float32 conditioning mask upcasts the whole DiT input via `jnp.concatenate`'s dtype promotion | Cast masks to the compute dtype both defensively (in `CosmosDiT`) and at the source (in the script) |
| image2world output resolution silently half of what was requested (t2v version: no error at all) | Latent tensor shape computed as `pixel // 16` (Wan2.2's VAE+patch combined compression), but Cosmos reuses Wan2.1's 8x-only VAE | `pixel // 8` for the latent shape; `patch_size * 8` (not `* 16`) for the resolution divisibility target |
| Generated video is a rigid, perfectly regular grid of scrambled color blobs | `unpatchify`'s channel-flatten order assumed symmetric with `patchify`'s; reference uses a genuinely different order for the two | Reorder unpatchify's reshape/transpose to `(height-patch, width-patch, temporal-patch, channel)`, matching the reference exactly |
| 30 low-resolution sampling steps takes 16+ minutes wall-clock | `step_index` as `static_argnums` of a jit containing both full DiT forward passes forces a full recompile every step | Split into a step-independent jitted DiT forward + an eagerly-run (unjitted) UniPC scheduler step |
| Grid artifact persists after the unpatchify fix, at every resolution/step-count/guidance-scale | Missing EDM-style preconditioning wrapper (`c_skip`/`c_out`/`c_in`/`c_noise`) — DiT received an ~1000x-wrong-scale timestep and an unscaled input at every step | Implement the wrapper in the sampling loop; back-solve UniPC's expected "model_output" convention algebraically so the existing scheduler needs no changes |
