# Per-layer weight offloading

Implemented (`--offload_dit_weights` on `generate_wan2_1_t2v.py`/
`generate_wan2_1_i2v.py`), for Wan2.1's `WanDiT` only. The technique itself
isn't novel — it's the same idea as DeepSpeed's ZeRO-Offload/ZeRO-Infinity
(offload parameter/optimizer state to host memory, stream it back to the
accelerator on demand) and, closer to this exact use case, HuggingFace
`diffusers`' `enable_sequential_cpu_offload()` (streams one submodule's
weights onto the accelerator at a time for large diffusion models). Here:
the full DiT weight tree stays in host RAM, and one layer's worth is
offloaded into a small fixed-shape HBM buffer at a time, swapped in a plain
Python loop around a single reused `jax.jit` compile. See "Implementation
status and measured results" below for what changed from the original
design during implementation, and the real (not estimated) cost.

## Implementation status and measured results

Used to fix two real, measured OOMs: Wan2.1 14B T2V and I2V-720P at native
720P resolution, both under the correct `--dit_dtype float32` default (see
[`lessons/wan2_1_precision_debugging.md`](lessons/wan2_1_precision_debugging.md)).
Both now produce coherent, correct output at native 720P — see
[`docs/benchmarking.md`](benchmarking.md) for the measured rows.

**What the OOMs actually were, once measured (not what was assumed
up-front)**: neither config was actually blocked by the DiT's own per-step
compute needing more HBM than fits — a fully-resident fp32 DiT tree, TP-4
sharded, comfortably fits *while the sampling loop runs* at native 720P.
The real problem in both cases was the DiT's weights staying HBM-resident
for the *entire script*, competing with an unrelated phase's own activation
memory for the same fixed budget: VAE decode's activations right after the
T2V sampling loop ends, and the conditioning image's VAE encode right
*before* the I2V sampling loop starts. Per-layer offloading fixes both as a
direct consequence of never letting the full DiT tree be HBM-resident
outside the brief window each individual block needs it — confirmed via
direct probing with the unmodified (pre-offloading) scripts before
implementing anything (see `docs/benchmarking.md`'s former `†` footnote,
now resolved).

**A real bug caught during implementation, not by inspection**: the
standalone `WanDiTBlock(...)` construction used for the per-layer
`block_forward` call must be passed `mesh=mesh` explicitly — easy to miss
since `WanDiT.setup()`'s own block construction always has `mesh` in scope
implicitly. Without it, `vidax.core.attention.dot_product_attention`'s
multi-device flash-attention dispatch (`_flash_attention_tpu_sharded`,
which needs `mesh` to `shard_map` the Pallas kernel) silently falls back to
`jax.nn.dot_product_attention`'s O(S²)-materializing path — fine at small
token counts, but a **~433GB** HLO temporary at native 720P's ~75,600-token
self-attention (confirmed by hitting exactly that `RESOURCE_EXHAUSTED`
error before catching it). This is the single most important thing to get
right if re-implementing this pattern for another model.

**Correctness, verified three ways**: (1) on CPU with a small dummy model,
`pre_process` -> per-layer `WanDiTBlock.apply` loop -> `post_process`
reproduces `WanDiT.__call__`'s output exactly (`jnp.allclose`, float32
noise-level only). (2) On real TPU hardware with the real 1.3B checkpoint,
sequential *eager* (non-`jax.jit`) per-layer application reproduces the
fully-fused single-`jax.jit` reference **bit-for-bit** (`max diff = 0.0`) —
proof the split itself introduces no logic error. (3) Once each block is
compiled as its *own* separate `jax.jit` program (what offloading actually
does, needed so `jax.device_put`-ing between layers isn't traced/unrolled
into one HLO program), a small but real numerical divergence appears versus
the fused path (~1-3% of output magnitude, single forward pass) — not from
the offloading logic, but from XLA choosing different fusion/precision
decisions for many small isolated per-block programs versus one large fused
one. Decoded video output stayed visually and statistically coherent (same
content, matching frame mean/std) in every comparison run at both 1.3B/480P
and native 14B/720P; this divergence is not corruption, but it does mean
offloaded output is not bit-identical to non-offloaded output the way,
e.g., re-running the same non-offloaded script twice would be.

**Real cost, measured**: per-layer offloading's promised "likely close to
free" bandwidth math (see "Real cost" below) did **not** hold up in
practice. First measured with a single `--offload_chunk_size 1` run on real
4-chip hardware, 14B T2V at native 720P: **130.0s/step offloaded vs.
26.1s/step non-offloaded at 480P** (not a perfectly matched comparison —
720P has more tokens than 480P too — but the earlier direct probe already
confirmed 720P's *non-offloaded* sampling loop itself completes fine,
meaning the per-layer transfer/compute overlap this design depends on is
not actually happening well on this hardware/JAX version), at **15.2GB**
peak HBM/chip (well under budget, plenty of headroom to spare) — the memory
goal was achieved cleanly; the throughput cost was not, contradicting the
optimistic estimate below. Treat `--offload_dit_weights` as a
correctness/memory-fit tool for configs that don't fit any other way, not a
free option to reach for by default. (This single-run measurement was later
superseded by the full 5-run `--offload_chunk_size 20` numbers now in
`docs/benchmarking.md` — see "Chunk size is flexible" below for both.)

**Chunk size is flexible, not fixed at one block**: `--offload_chunk_size N`
groups `N` consecutive blocks into one offloaded HBM buffer / one `jax.jit`
compile instead of always 1 (must divide `num_layers`; see "Design" below
for the mechanics). Swept on Wan2.1 14B T2V at native 720P
(`benchmarks/sweep_offload_chunks.py`, `--num_runs 1 --num_steps 5` per
chunk size — see [`docs/benchmarking.md`](benchmarking.md#weight-offloading-chunk-size-sweep)
for the full table and methodology note):

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 | 141.7 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model) | 111.3 | 26.1 |

Larger chunks do help — ~21% faster from chunk size 1 to 40 — but only
modestly, and even the largest chunk (the entire 40-layer DiT re-transferred
fresh every step, as one chunk) stays far short of the non-offloaded
~26-30s/step baseline. So grouping blocks isn't the main lever on the
overhead identified above; two more likely (unconfirmed, not chased further
this round) culprits: a chunk's weights get a fresh `jax.device_put` every
single step regardless of chunk size, wastefully re-transferring even a
`--offload_chunk_size 40` chunk that never actually changes between steps;
and splitting the forward pass into three separately-compiled `jax.jit`
programs (`pre_process` / chunk loop / `post_process`) loses end-to-end
operator fusion across those boundaries no matter how big each chunk is,
unlike the non-offloaded path's single fused `single_step`.

**`--offload_chunk_size 20` confirmed with the full standard methodology**
(`--num_runs 5`, full step count, for both T2V and I2V — the two
`docs/benchmarking.md` native-720P rows now use this, replacing the earlier
`--offload_chunk_size 1` single-run numbers, specifically so those rows are
directly comparable to every other model's 5-run measurement in that table):

| Model | `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| --- | ---: | ---: | ---: |
| T2V | 1 | 130.0 | 15.2 |
| T2V | 20 | 123.0 | 23.0 |
| I2V | 1 | 137.9 | 19.1 |
| I2V | 20 | 127.2 | 32.7 |

T2V's confirmed 123.0s/step matches the quick 5-step sweep's 123.7s/step
estimate closely — the cheap sweep methodology is a reasonable way to screen
chunk sizes before paying for a full run. Both models get a modest, real
win from chunk size 20 (~5.4% for T2V, ~7.8% for I2V) — but at a cost that
lands very differently: T2V's 23.0GB still leaves comfortable headroom,
while **I2V's 32.7GB is close to this chip's real ceiling** (I2V already
starts from a higher baseline at chunk size 1 too — 19.1GB vs T2V's 15.2GB
— from carrying CLIP and the conditioning image's VAE-encode residency on
top of the DiT). `--offload_chunk_size 20` is what `docs/benchmarking.md`'s
two native-720P rows now use, given that confirmed win; fall back to
`--offload_chunk_size 1` (still the safer default for a first attempt on
unfamiliar hardware, or when combining `--offload_dit_weights` with
anything else that also needs HBM headroom) if 20 doesn't fit.

## Motivation

This repo's 4-chip machine already has two real, measured cases where a
single model doesn't fit fully device-resident:

- Wan2.1 I2V-14B at native 720P, with the DiT weights correctly kept at
  float32 (see [`lessons/wan2_1_precision_debugging.md`](lessons/wan2_1_precision_debugging.md))
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

## Design: chunked (default per-layer) offloading

**Key enabling fact**: every Wan/Cosmos DiT block shares identical
parameter shapes across all `num_layers` blocks (this is architecturally
guaranteed — it's what lets `nn.scan`-style layer stacking work at all).
That means a single `chunk_forward(chunk_params, x, ...)` function,
JIT-compiled *once* against one chunk's shape/dtype/sharding signature, is
reusable verbatim for every chunk of `--offload_chunk_size` blocks — the
compiled program never needs to know which chunk's weights it's currently
being handed, only that they match the signature it was compiled for.

1. **Host-resident full weight tree.** Load and shard-plan (via
   `shard_wan_params`) the *entire* param pytree, but keep it as numpy
   arrays on the host — never `device_put` the whole thing at once. This is
   already how `cast_numpy_tree_to_dtype`/checkpoint loading works before
   the existing single `device_put(dit_params, sharding)` call in every
   example script; the change is to stop calling that single big
   `device_put` and instead keep the host pytree around for per-chunk use.

2. **One small fixed-shape HBM buffer.** Allocate a `device_put`-able
   chunk-sized param pytree slot once — a Python `list` of
   `--offload_chunk_size` per-layer param dicts (a valid pytree, so
   `jax.jit`/`jax.device_put` handle it like any other), each sharded with
   the same `shard_wan_params`-derived `NamedSharding` a single layer would
   get (every block's sharding is identical, so one layer's spec, repeated
   `chunk_size` times, is exactly the chunk's spec). This is the *only*
   thing that needs to be HBM-resident for weights, at ~`total_weight_memory
   * chunk_size / num_layers` per device instead of the full total.

3. **One JIT-compiled `chunk_forward`.** `jax.jit(chunk_forward,
   donate_argnums=(0,))`, signature `(chunk_params, x, ...) -> x`, its body
   a plain Python `for` loop over the chunk's `chunk_size` blocks —
   deliberately *traced* into one HLO program here (unlike the outer loop
   below), since fusing/scheduling across a chunk's few layers together is
   exactly the point of grouping them. Compiled once against the fixed
   chunk shape. `donate_argnums` on the params buffer lets each chunk's
   `device_put` reuse the same physical HBM region as the previous chunk's,
   rather than allocating a fresh one each time — this is what keeps the
   *resident* footprint at one chunk's worth throughout, not accumulating
   across chunks.

4. **Plain Python loop over chunks**, not a `jax.jit`-traced loop — for the
   same reason `docs/hardware_and_sharding.md` §5 already establishes for
   the diffusion sampling loop and VAE decode's per-chunk loop: `jax.jit`
   unrolling a Python loop into one HLO program can need every iteration's
   intermediates to coexist in that program's memory footprint, defeating
   the entire point of offloading. Each iteration:
   ```python
   for chunk_idx in range(num_layers // chunk_size):
       chunk_params = jax.device_put(host_params_per_chunk[chunk_idx], chunk_sharding)
       x = chunk_forward(chunk_params, x, ...)
   ```
   Since `chunk_forward`'s shape/dtype/sharding signature is identical
   every call, this compiles exactly once total, not once per chunk.
   `--offload_chunk_size 1` (the default) is the special case this whole
   design was originally written for — see "Chunk size is flexible" above
   for what raising it actually buys (modest throughput, real extra HBM).

## Real cost: bandwidth, not compute (estimate — see measured results above)

*This section is the original, pre-implementation estimate. It turned out
to be too optimistic — see "Implementation status and measured results"
above for what was actually measured (130.0s/step at native 720P, not
"close to free"). Left as-is below since the reasoning about* why *it could
have gone either way is still correct — it's the conclusion that didn't
hold.*

TPU MXU accumulates matmuls in float32 internally regardless of input
dtype (see `hardware_and_sharding.md`'s flash-attention section for the
same fact applied to a different question) — the cost of this scheme isn't
numerical, it's the host-to-device transfer time competing with compute
time for the same wall-clock budget. Rough scale: a 14B-class model's full
weight tree is on the order of ~14GB/device even Megatron-sharded across 4
chips at bf16 (more at fp32, see the precision doc); a single per-step
transfer of that magnitude is well under a second at typical TPU host-to-
device bandwidth, against a per-step compute cost on the order of ~10-30s
measured elsewhere in this repo's benchmarks — i.e. offloading the *entire*
weight tree once per diffusion step (not once per layer) would likely be
close to free, hidden behind compute.

**Per-layer granularity is a real open risk, not a settled win.** Offloading
once per *layer* (40 times per step for Wan's 14B DiT) means 40 separate
host-to-device transfers per step instead of 1, each individually small —
whether JAX/XLA's async dispatch actually overlaps a given layer's
transfer with the *previous* layer's compute (the only way per-layer
offloading avoids becoming purely additive latency) depends on scheduling
details this proposal doesn't resolve on paper. **This risk was real**: the
measured 130.0s/step at native 720P (vs. this repo's other DiTs' per-step
costs, all under 40s even at comparable or larger token counts — see
`docs/benchmarking.md`) indicates the 40 per-layer transfers are landing
mostly serialized with compute, not overlapped. Worth investigating further
(e.g. explicit prefetch of layer N+1 while layer N computes) before using
this for latency-sensitive serving rather than one-off correctness/memory
fixes, but that investigation wasn't done here.

## Where this doesn't help

- Doesn't reduce *activation* memory (the per-token modulation/attention
  memory that sequence parallelism targets) — orthogonal to this technique,
  composes with it independently exactly as TP and SP already compose with
  each other (see `hardware_and_sharding.md`'s "Combining with Megatron
  TP" section).
- Doesn't help if a *single layer's* params don't fit even alone — not a
  realistic concern for any model currently in this repo (a single Wan/
  Cosmos DiT block is a small fraction of total weight memory), but worth
  stating as a hard limit of the approach.
- Adds real implementation complexity (a second host-resident weight-
  loading path, a new per-layer sharding spec, careful `donate_argnums`
  bookkeeping) — and, per the measured results above, a real throughput
  cost too. Only worth opting into (`--offload_dit_weights`) for a config
  that doesn't fit any other way (native 720P for Wan2.1 14B T2V/I2V, both
  now fixed by it), not as a general-purpose default.

## Verification: done, results above

1. ~~Implement `block_forward`/per-layer offloading for one model~~ — done
   for Wan2.1's `WanDiT`/`WanDiTBlock`.
2. ~~Correctness~~ — done, three ways (CPU exact match, TPU bit-exact eager
   split, TPU offloaded-vs-fused divergence characterized as JIT-boundary
   noise, not a logic bug). See "Implementation status and measured
   results" above.
3. ~~Measure actual per-layer transfer/compute overlap~~ — done: it does
   *not* overlap well on this hardware/JAX version (130.0s/step at native
   720P). The "likely free" estimate did not hold.
4. ~~Apply it to a case that currently doesn't fit at all~~ — done: Wan2.1
   14B T2V and I2V-720P at native 720P, both now produce coherent output
   (see `docs/benchmarking.md`'s rows), comfortably under budget at either
   chunk size measured (15.2-32.7GB/chip depending on model/chunk size, see
   below — this chip's real ceiling was never hit).
5. ~~Generalize beyond one block per offloaded unit~~ — done: `--offload_
   chunk_size` groups any divisor of `num_layers` blocks per unit, verified
   correct at chunk sizes 1/2/4/8 (CPU exact match against the fused
   reference) and swept for throughput/memory at 1/2/4/8/20/40 on real 14B
   T2V/720P hardware — see "Chunk size is flexible" above.
6. ~~Confirm the sweep-chosen chunk size with the full standard
   methodology~~ — done: `--offload_chunk_size 20` re-measured with
   `--num_runs 5`/full step count for both T2V and I2V, matching the
   sweep's estimate closely for T2V and giving I2V its own (previously
   unmeasured) chunk-20 numbers — see "Chunk size is flexible" above.
   `docs/benchmarking.md`'s two native-720P rows now report these numbers.

**Not done, left for future work**: applying this to Wan2.2 A14B (its
weights aren't downloaded on this machine yet — out of scope for this
round); investigating why even a whole-model single chunk still falls well
short of non-offloaded throughput (the two candidate causes named above —
redundant per-step re-transfer of unchanged weights, and fusion lost across
the three-way `jax.jit` split — neither confirmed nor fixed this round);
finding a chunk size for I2V with more HBM headroom than chunk size 20's
32.7GB while keeping some of its throughput win, if I2V is ever combined
with something else that also needs HBM.
