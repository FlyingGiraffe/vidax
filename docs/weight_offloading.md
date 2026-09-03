# Per-layer weight offloading

Implemented for every model in this repo with a DiT that doesn't fit fully
device-resident at some resolution: Wan2.1's `WanDiT` (`--offload_dit_weights`
on `generate_wan2_1_t2v.py`/`generate_wan2_1_i2v.py`), Wan2.2 A14B's `WanDiT`
(`generate_wan2_2_t2v_a14b.py`/`generate_wan2_2_i2v_a14b.py`), and
Cosmos-Predict2.5's `CosmosDiT` (`generate_cosmos2_5.py`). The technique isn't
novel — it's the same idea as DeepSpeed's ZeRO-Offload/ZeRO-Infinity (offload
parameter/optimizer state to host memory, stream it back to the accelerator
on demand) and, closer to this exact use case, HuggingFace `diffusers`'
`enable_sequential_cpu_offload()`. Here: the full DiT weight tree stays in
host RAM, and one layer's (or one `--offload_chunk_size`-block group's)
worth is streamed into a small fixed-shape HBM buffer at a time, swapped in a
plain Python loop around a single reused `jax.jit` compile.

## Design

**Key enabling fact**: every DiT block in this repo shares identical
parameter shapes across all `num_layers` blocks (architecturally guaranteed —
it's what lets `nn.scan`-style layer stacking work at all). So a single
`chunk_forward(chunk_params, x, ...)` function, JIT-compiled *once* against
one chunk's shape/dtype/sharding signature, is reusable verbatim for every
chunk — the compiled program never needs to know which chunk's weights it's
currently handed, only that they match the signature it was compiled for.

1. **Host-resident full weight tree.** Load and shard-plan (via
   `shard_wan_params`) the entire param pytree, but keep it as numpy arrays
   on the host — skip the single big `device_put(dit_params, sharding)` call
   every non-offloaded script makes.
2. **One small fixed-shape HBM buffer.** A `device_put`-able chunk-sized
   param pytree slot — a Python `list` of `--offload_chunk_size` per-layer
   param dicts, each sharded with the same `NamedSharding` a single layer
   would get (every block's sharding is identical, so one layer's spec,
   repeated `chunk_size` times, is the chunk's spec). This is the only
   weight-memory that needs to be HBM-resident, at `~total_weight_memory *
   chunk_size / num_layers` per device instead of the full total.
3. **One JIT-compiled `chunk_forward`**, `jax.jit(chunk_forward,
   donate_argnums=(0,))`, body a plain Python `for` loop over the chunk's
   blocks — deliberately traced into one HLO program (fusing/scheduling
   across a chunk's few layers together is the point of grouping them).
   `donate_argnums` on the params buffer lets each chunk's `device_put` reuse
   the previous chunk's physical HBM region, keeping the resident footprint
   at one chunk's worth throughout, not accumulating across chunks.
4. **Plain Python loop over chunks**, not a `jax.jit`-traced one — same
   reasoning as the sampling loop and VAE decode's per-chunk loop elsewhere
   in this repo (see [`docs/hardware_and_sharding.md`](hardware_and_sharding.md)):
   unrolling into one HLO program would need every iteration's intermediates
   to coexist, defeating the point of offloading:
   ```python
   for chunk_idx in range(num_layers // chunk_size):
       chunk_params = jax.device_put(host_params_per_chunk[chunk_idx], chunk_sharding)
       x = chunk_forward(chunk_params, x, ...)
   ```
   `chunk_forward`'s signature is identical every call, so this compiles
   exactly once total, not once per chunk.

The forward pass is split into `pre_process` (patchify + time/text
embedding, and, under `sequence_parallel`, the token-sequence chunk) /
per-block loop / `post_process` (head projection + unpatchify) — each model's
DiT (`WanDiT`, `CosmosDiT`) is built with Flax `setup()` rather than
`@nn.compact` specifically so these three pieces can be called independently
outside a single `__call__`. Every one of these three-way splits was verified
bit-exact against the original fused `__call__` before being used for
anything (small dummy config, `jnp.allclose`/`max diff = 0.0`).

## Why it isn't free: bandwidth vs. compute

TPU's MXU accumulates matmuls in float32 internally regardless of input
dtype, so this scheme's cost isn't numerical — it's host-to-device transfer
time competing with compute time for the same wall-clock budget. On paper,
offloading the *entire* weight tree once per diffusion step (not once per
layer) looked like it should be close to free: a 14B-class model's weight
tree is on the order of ~14GB/device even Megatron-sharded across 4 chips at
bf16, and a transfer that size is well under a second at typical TPU
host-to-device bandwidth, against a ~10-30s per-step compute cost measured
elsewhere in this repo.

**That didn't hold at per-layer granularity.** Offloading once per layer (40
times per step for a 40-layer DiT) means 40 separate host-to-device
transfers per step instead of one — whether JAX/XLA's async dispatch overlaps
a given layer's transfer with the *previous* layer's compute is what
determines whether this stays close to free or becomes additive latency.
Measured on Wan2.1 14B T2V at native 720P, `--offload_chunk_size 1`:
**141.7s/step offloaded vs. 26.1s/step non-offloaded at 480P** (not a
perfectly matched comparison — 720P has more tokens too — but a direct probe
already confirmed 720P's *non-offloaded* sampling loop completes fine, so the
gap is real) at 15.2GB peak HBM/chip (well under budget). The per-layer
transfers are landing mostly serialized with compute on this hardware/JAX
version, not overlapped. Two likely (unconfirmed, not chased further)
contributors: a chunk's weights get a fresh `device_put` every step
regardless of chunk size, even when the chunk's contents never change
between steps; and splitting the forward pass into three separately-compiled
`jax.jit` programs loses end-to-end operator fusion across those boundaries,
unlike the non-offloaded path's single fused `single_step`.

**Treat `--offload_dit_weights` as a correctness/memory-fit tool for configs
that don't fit any other way, not a free option to reach for by default.**

**Chunk size trades HBM for throughput.** `--offload_chunk_size N` groups `N`
consecutive blocks into one offloaded HBM buffer / one `jax.jit` compile
(must divide `num_layers`). Swept on Wan2.1 14B T2V at native 720P
(`benchmarks/sweep_offload_chunks.py`, `--num_runs 1 --num_steps 5` per size
— cheap screening before a full run):

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 | 141.7 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model) | 111.3 | 26.1 |

Larger chunks help, but only modestly (~21% faster from 1 to 40), and even
the largest chunk (the entire DiT re-transferred fresh every step, as one
chunk) stays far short of the non-offloaded ~26-30s/step baseline — grouping
blocks isn't the main lever on the overhead above. `--offload_chunk_size 20`
was then confirmed with the full 5-run methodology and is what
`docs/benchmarking.md`'s native-720P Wan2.1 rows use (123.0s/step T2V,
127.2s/step I2V, both re-measured at 5 runs/full step count, matching the
5-step sweep's estimate closely for T2V). `--offload_chunk_size 1` remains
the safer default for a first attempt on unfamiliar hardware, or when
combining `--offload_dit_weights` with anything else that needs HBM headroom.

## Correctness

Verified three ways, on Wan2.1 (the first model this was built for) and
reused as the standard for every later model: (1) on CPU with a small dummy
model, `pre_process` -> per-layer block loop -> `post_process` reproduces
the fused `__call__`'s output exactly (`jnp.allclose`/`max diff = 0.0`).
(2) On real TPU hardware, sequential *eager* (non-`jax.jit`) per-layer
application reproduces the fully-fused single-`jax.jit` reference
bit-for-bit — proof the split itself introduces no logic error.
(3) Once each block is compiled as its own separate `jax.jit` program (what
offloading actually does), a small but real numerical divergence appears
versus the fused path (~1-3% of output magnitude, single forward pass) — not
from the offloading logic, but from XLA choosing different fusion/precision
decisions for many small isolated programs versus one large fused one.
Decoded video stayed visually and statistically coherent (matching frame
mean/std) in every comparison; this divergence means offloaded output isn't
bit-identical to non-offloaded output, but it is not corruption.

One implementation pitfall worth flagging for anyone extending this pattern:
the standalone per-layer block construction used for `chunk_forward` must be
passed `mesh=mesh` explicitly — easy to miss since a DiT's own `setup()`
block construction always has `mesh` in scope implicitly. Without it,
`vidax.core.attention.dot_product_attention`'s multi-device flash-attention
dispatch silently falls back to `jax.nn.dot_product_attention`'s
O(S²)-materializing path — fine at small token counts, but a ~433GB HLO
temporary at native 720P's ~75,600-token self-attention (this is exactly the
failure that surfaced it, on Wan2.1).

## Where this doesn't help

- Doesn't reduce *activation* memory (the per-token modulation/attention
  memory that sequence parallelism targets) — orthogonal to this technique,
  composes with it independently (see `hardware_and_sharding.md`'s
  "Combining with Megatron TP" section).
- Doesn't help if a single layer's params don't fit even alone — not a
  realistic concern for any model currently in this repo.
- Adds real implementation complexity (a second host-resident weight-loading
  path, a per-layer sharding spec, `donate_argnums` bookkeeping) and a real
  throughput cost (above). Only worth opting into for a config that doesn't
  fit any other way, not as a general-purpose default.

## Wan2.1: fixing two real OOMs at native 720P

Used to fix two real, measured OOMs: Wan2.1 14B T2V and I2V-720P at native
720P, both under the correct `--dit_dtype float32` default (see
[`lessons/wan2_1_debugging.md`](lessons/wan2_1_debugging.md)).
**Neither OOM was actually the DiT's own per-step compute needing more HBM
than fits** — a fully-resident fp32 DiT tree, TP-4 sharded, comfortably fits
*while the sampling loop runs* at native 720P. The real problem in both cases
was the DiT's weights staying HBM-resident for the *entire script*,
competing with an unrelated phase's own activation memory for the same fixed
budget: VAE decode's activations right after the T2V sampling loop ends, and
the conditioning image's VAE encode right before the I2V sampling loop
starts. Per-layer offloading fixes both by never letting the full DiT tree be
HBM-resident outside the brief window each block needs it. Both now produce
coherent, correct output at native 720P — see
[`docs/benchmarking.md`](benchmarking.md) for the measured rows
(`--offload_chunk_size 20`: 123.0s/step T2V at 23.0GB, 127.2s/step I2V at
32.7GB).

## A14B (Wan2.2)

A14B needed more than a straight port of Wan2.1's implementation, because
Wan2.2's `WanDiT` computes AdaLN modulation **per token**, not per sample
(`vidax.models.wan.wan2_2.dit`'s module docstring) — its `e0` tensor is
`(B, seq_len, 6, dim)` instead of Wan2.1's `(B, 6, dim)`. At native 720P's
~75k patch tokens, `e0` alone is multiple GB of *activation* memory, which
offloading alone (`--sequence_parallel_size 1`) cannot shrink no matter the
chunk size: the non-offloaded baseline needs ~61.7GB/chip of HLO temporaries
and offloading alone needs ~56.6GB, both far over this chip's ~30.75GB
budget. So `--offload_dit_weights` was extended to compose with
`--sequence_parallel_size` for this model (never needed for Wan2.1, where
activation memory was never the binding constraint).

`WanDiT.pre_process` computes `e0`/`x` at *full* (unsharded) token length and
only chunks them across the sequence-parallel axis at the very end — inherited
behavior from the fused `__call__`, not something the offloading split
introduced. That means `e0` is still briefly full-size even under
`sequence_parallel`, so it only shrinks *most*, not all, of `pre_process`'s
peak memory. Measured progression at native 720P/81 frames, all with
`--offload_dit_weights --offload_chunk_size 1`:

| `--tensor_parallel_size` | `--sequence_parallel_size` | HLO temporaries required | Fits in ~30.75GB? |
| ---: | ---: | ---: | :---: |
| 4 | 1 | 56.6GB | No |
| 2 | 2 | 45.8GB | No |
| 1 | 4 | 40.4GB | No |

None of these fit at the reference's full 81 frames — `--tensor_parallel_size
2 --sequence_parallel_size 2` (or fewer frames, see the results table below)
is what the measured rows use.

Row-parallel layers under `sequence_parallel` (`attend`'s output projection,
both DiT blocks' FFN down-projection) sum their per-device partial outputs
via `vidax.models.wan.common.dit_layers.psum_row_parallel`, not a bare
`jax.lax.psum` — it corrects for `nn.Dense`'s replicated bias otherwise
getting summed once per `tp`-way device instead of once (a true no-op when
`sequence_parallel` is off or `tp` has size 1, so safe to call
unconditionally). See
[`docs/lessons/wan2_1_debugging.md`](lessons/wan2_1_debugging.md#row-parallel-psum-double-counted-nndenses-bias-under-sequence_parallel)
for the bug this fixes and how it was found (first surfaced while combining
offloading with sequence parallelism here, but the underlying bug was in
shared Wan2.1/Wan2.2 code, not specific to offloading).

### A14B results, all configs

All four A14B rows (I2V/T2V × 480P/720P) use `--tensor_parallel_size 2
--sequence_parallel_size 2 --offload_dit_weights`; only `--offload_chunk_size`
and frame count differ. At 480P, smaller token count leaves enough HBM margin
that the reference's full 81 frames fit, and a swept `--offload_chunk_size`
(every divisor of 40: `1`,`2`,`4`,`10` fit, `20` OOMs) picks a larger chunk
for a real throughput win. At native 720P, per-token activation memory
dominates enough that only `--offload_chunk_size 1` fits, and even then the
reference's 81 frames don't — binary search lands on the largest count that
does. (T2V's native-720P search additionally found the OOM boundary is *not*
monotonic in frame count on this hardware/JAX version: 81 frames needs
50.0GB, 61 needs 31.7GB, but 57 needs *more* than 61 at 36.0GB — likely XLA
choosing different fusion/tiling decisions at different shapes, not a smooth
trend. Don't assume linear interpolation between two working/OOMing frame
counts predicts what's in between.)

I2V's `in_dim=36` (channel-concatenated mask+VAE-latent conditioning) gives
it a larger per-token footprint than T2V's `in_dim=16`, which shows up as
higher peak HBM and slightly slower per-step at native 720P (where per-token
activation memory dominates) but barely matters at 480P (where chunk-size-
driven weight residency dominates instead):

| Task | Resolution | `--offload_chunk_size` | Frames | Compile (s) | Per-step (s) | Peak HBM/chip (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| I2V | 480P (544x720) | 10 | 81 (full reference) | 146.1 | 44.5 | 28.3 |
| I2V | Native 720P (832x1104) | 1 | 33 (reduced from 81) | 102.9 | 49.1 | 20.5 |
| T2V | 480P (480x832) | 10 | 81 (full reference) | 65.8 | 43.2 | 28.4 |
| T2V | Native 720P (720x1280) | 1 | 33 (reduced from 81) | 33.7 | 46.4 | 18.1 |

Full 81-frame native 720P remains out of reach on this 4-chip machine for
both tasks, even with offloading and sequence parallelism combined — it would
need chunking `pre_process` itself across the token axis, not just the block
loop, a bigger change than implemented so far. See `docs/benchmarking.md`'s
`§` footnote and `docs/models/wan2_2.md`'s A14B sections for the CLI
reference.

## Cosmos-Predict2.5

`CosmosDiT` was refactored to the same `setup()`/`pre_process`/`post_process`
split as `WanDiT` (verified bit-exact against the original `@nn.compact`
`__call__`), and `--offload_dit_weights`/`--offload_chunk_size` added to
`generate_cosmos2_5.py`. Unlike A14B, this is a single DiT (no MoE
expert-switching to compose with), so the implementation is closer to
Wan2.1's original: host-resident weight tree, per-chunk `device_put` inside
the sampling loop, no separate "did the expert change" bookkeeping.

Used to fix the same class of problem as Wan2.1's native-720P rows: the 14B
checkpoint's reference default (704x1280, 93 frames) previously didn't fit
fully device-resident on this 4-chip machine at any
`--tensor_parallel_size`/`--sequence_parallel_size` split, and was reduced to
45 frames as a result. With `--offload_dit_weights --offload_chunk_size 1`
(`--tensor_parallel_size 4`, the model's own default sharding), the full
reference 93 frames fits, confirmed with the full 5-run benchmark
methodology: 48.5s compile, 4479.6s generation, 128.0s/step, **14.7GB**
peak HBM/chip — comfortable headroom, well under this chip's ~30.75GB
budget (unlike A14B's native-720P rows above, which sit close to the
ceiling even with offloading). See `docs/benchmarking.md`'s Cosmos-Predict2.5
14B row.
