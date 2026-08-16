# Proposal: per-layer HBM/RAM weight streaming

Not yet implemented. A design for running a DiT larger than fits fully
HBM-resident at once, by keeping the full weight tree in host RAM and
streaming only one layer's worth into a small fixed-shape HBM buffer at a
time, swapped in a plain Python loop around a single reused `jax.jit`
compile. Written up so it can be implemented directly from this doc without
re-deriving the design.

## Motivation

This repo's 4-chip machine already has two real, measured cases where a
single model doesn't fit fully device-resident:

- Wan2.1 I2V-14B at native 720P, with the DiT weights correctly kept at
  float32 (see [`wan2_1_precision_debugging.md`](wan2_1_precision_debugging.md))
  — OOMs during the conditioning image's VAE encode, before generation
  even starts.
- Wan2.2 A14B (two 14B experts) — even with only one expert kept
  device-resident at a time (the existing whole-model swap in
  `generate_wan2_2_t2v_a14b.py`/`_i2v_a14b.py`, described below), that one
  ~7.5GB/device-sharded expert alone leaves little headroom at high
  resolution, and both experts together (~15GB/device) don't fit at all.

Tensor parallelism and sequence parallelism were both directly ruled out as
general fixes for cases like these: TP only shards weights as far as the
number of chips allows (already maxed at 4 on this machine), and SP shards
*activations*, not weights — trading TP width for SP width to try to free
weight memory does the opposite of what's needed when weight memory (not
activation memory) is the actual constraint. What's needed instead is
reducing how much weight memory is *resident at once*, not sharding it
differently across the same fixed chip budget.

## What already exists: whole-model swap (A14B)

`generate_wan2_2_t2v_a14b.py`/`generate_wan2_2_i2v_a14b.py` already do a
coarse version of this idea, worth understanding first since the per-layer
design below is a direct generalization of it. Both DiT experts' weights
are loaded and Megatron-sharded on the host (numpy pytrees, never
`device_put` yet). The sampling loop tracks which expert the current
timestep needs (`high_noise` above the boundary, `low_noise` below — the
reference's own crossover, happening at most once per run since timesteps
decrease monotonically) and only calls `jax.device_put(host_params,
dit_sharding_spec)` when the active expert actually changes — at most once
per run, not once per step. The single `jax.jit`-compiled per-step function
is called with whichever expert's device-resident params are currently
live; because both experts share identical shapes/dtypes/sharding, this
never triggers a recompile, only a rebind of which concrete buffer the
compiled program reads from.

This works well when swaps are rare (here, at most once per run) and each
swap can afford to be a full model's worth of host-to-device transfer. It
doesn't help the case where even *one* model's full weight tree doesn't fit
resident at all — that needs swapping happen far more often, at finer
granularity, which is what the per-layer version below is for.

## Proposed design: per-layer streaming

**Key enabling fact**: every Wan/Cosmos DiT block shares identical
parameter shapes across all `num_layers` blocks (this is architecturally
guaranteed — it's what lets `nn.scan`-style layer stacking work at all).
That means a single `block_forward(params, x, ...)` function, JIT-compiled
*once* against one layer's shape/dtype/sharding signature, is reusable
verbatim for every layer — the compiled program never needs to know which
layer's weights it's currently being handed, only that they match the
signature it was compiled for.

1. **Host-resident full weight tree.** Load and shard-plan (via
   `shard_wan_params`) the *entire* param pytree, but keep it as numpy
   arrays on the host — never `device_put` the whole thing at once. This is
   already how `cast_numpy_tree_to_dtype`/checkpoint loading works before
   the existing single `device_put(dit_params, sharding)` call in every
   example script; the change is to stop calling that single big
   `device_put` and instead keep the host pytree around for per-layer use.

2. **One small fixed-shape HBM buffer.** Allocate a `device_put`-able
   per-layer param pytree slot once, sized/sharded/dtype-matched to exactly
   one block's params (using the same `shard_wan_params`-derived
   `NamedSharding` a single layer would get). This is the *only* thing that
   needs to be HBM-resident for weights, at ~`total_weight_memory /
   num_layers` per device instead of the full total.

3. **One JIT-compiled `block_forward`.** `jax.jit(block_forward,
   donate_argnums=(0,))` (or similar), signature `(layer_params, x, ...) ->
   x`, compiled once against the fixed per-layer shape. `donate_argnums` on
   the params buffer lets each layer's `device_put` reuse the same
   physical HBM region as the previous layer's, rather than allocating a
   fresh one each time — this is what keeps the *resident* footprint at one
   layer's worth throughout, not accumulating across layers.

4. **Plain Python loop over layers**, not a `jax.jit`-traced loop — for the
   same reason `docs/hardware_and_sharding.md` §5 already establishes for
   the diffusion sampling loop and VAE decode's per-chunk loop: `jax.jit`
   unrolling a Python loop into one HLO program can need every iteration's
   intermediates to coexist in that program's memory footprint, defeating
   the entire point of streaming. Each iteration:
   ```python
   for layer_idx in range(num_layers):
       layer_params = jax.device_put(host_params_per_layer[layer_idx], layer_sharding)
       x = block_forward(layer_params, x, ...)
   ```
   Since `block_forward`'s shape/dtype/sharding signature is identical
   every call, this compiles exactly once total, not once per layer.

## Real cost: bandwidth, not compute

TPU MXU accumulates matmuls in float32 internally regardless of input
dtype (see `hardware_and_sharding.md`'s flash-attention section for the
same fact applied to a different question) — the cost of this scheme isn't
numerical, it's the host-to-device transfer time competing with compute
time for the same wall-clock budget. Rough scale: a 14B-class model's full
weight tree is on the order of ~14GB/device even Megatron-sharded across 4
chips at bf16 (more at fp32, see the precision doc); a single per-step
transfer of that magnitude is well under a second at typical TPU host-to-
device bandwidth, against a per-step compute cost on the order of ~10-30s
measured elsewhere in this repo's benchmarks — i.e. streaming the *entire*
weight tree once per diffusion step (not once per layer) would likely be
close to free, hidden behind compute.

**Per-layer granularity is a real open risk, not a settled win.** Streaming
once per *layer* (40 times per step for Wan's 14B DiT) means 40 separate
host-to-device transfers per step instead of 1, each individually small —
whether JAX/XLA's async dispatch actually overlaps a given layer's
transfer with the *previous* layer's compute (the only way per-layer
streaming avoids becoming purely additive latency) depends on scheduling
details this proposal doesn't resolve on paper. This needs empirical
validation before relying on it, not just the bandwidth-vs-compute
estimate above — see Verification below.

## Where this doesn't help

- Doesn't reduce *activation* memory (the per-token modulation/attention
  memory that sequence parallelism targets) — orthogonal to this proposal,
  composes with it independently exactly as TP and SP already compose with
  each other (see `hardware_and_sharding.md`'s "Combining with Megatron
  TP" section).
- Doesn't help if a *single layer's* params don't fit even alone — not a
  realistic concern for any model currently in this repo (a single Wan/
  Cosmos DiT block is a small fraction of total weight memory), but worth
  stating as a hard limit of the approach.
- Adds real implementation complexity (a second host-resident weight-
  loading path, a new per-layer sharding spec, careful `donate_argnums`
  bookkeeping) — only worth doing once a concrete case actually needs it
  (Wan2.1 I2V-14B native-720P and/or A14B are the two current candidates;
  neither is blocking today's default-resolution usage).

## Verification plan (before relying on this for real runs)

1. Implement `block_forward`/per-layer streaming for one model (Wan2.1's
   `WanDiTBlock` is the simplest target — identical shapes across layers,
   no MoE-expert complexity).
2. Correctness: compare output against the existing whole-tree-resident
   path on a config that fits both ways (e.g. Wan2.1 1.3B at 480P) —
   `jnp.allclose` within the same bf16-associativity tolerance
   `hardware_and_sharding.md` already uses for its own multi-device
   regression checks. A streaming bug should produce wrong numbers, not a
   crash, so this check is required, not optional.
3. Measure actual per-layer transfer/compute overlap on real hardware
   (wall-clock per step, streaming vs. the existing whole-tree-resident
   path, at a config that fits both) to confirm the "likely free" estimate
   above rather than assuming it.
4. Only then apply it to a case that currently doesn't fit at all (Wan2.1
   I2V-14B native-720P, or A14B without the tensor+sequence-parallel
   combination already documented in `hardware_and_sharding.md`), and
   report the real achieved resolution/frame-count honestly either way,
   matching this repo's existing documentation practice.
