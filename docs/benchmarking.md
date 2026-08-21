# Benchmarking

Inference-latency and memory benchmarks for every implemented model, on
Google Cloud TPU. **Empty cells mean that combination hasn't been measured
yet** — not that it's unsupported; see each model's own
[`docs/models/`](models/) guide for implementation/verification status.

Reproduce any row with `benchmarks/run_*.py` (see
[`benchmarks/common.py`](../benchmarks/common.py) for the shared harness and
[`benchmarks/run_all.py`](../benchmarks/run_all.py) to run every
checkpoint-available combination in one pass, in the same order as this
doc's table). Checkpoints are assumed to live under `./checkpoints/` (every
example script's own default); point elsewhere with `--checkpoint_dir` or
the `VIDAX_CHECKPOINT_DIR` environment variable without touching any script.
Measured with `jax==0.11.0` on `TPU v4` (4 chips) — a different JAX/libtpu
version or chip generation can shift these numbers meaningfully; each
`benchmarks/results/*.json` file records the exact `jax_version`/
`device_kind`/`device_count` used for that row.

Every row is the average of `--num_runs` independent end-to-end runs
(default 5, each with a freshly cleared JAX compilation cache, so every run
is a genuine cold start, not just the first) — `benchmarks/results/*.json`
keeps every individual run's raw metrics alongside the average. Each run's
output video is saved to `out/<slug>/<slug>_<run>.mp4` (e.g.
`out/cosmos_3_nano_t2v/cosmos_3_nano_t2v_1.mp4`). This includes both
native-720P Wan2.1 rows (`--offload_dit_weights --offload_chunk_size 20`,
see the `‡` footnote below) — each run there takes well over an hour, so
the full 5-run average took a long time to collect (multiple sequential
hours per model), but was worth it for a fair, consistent comparison against
every other row in this table (the chunk-size sweep below, which only needs
relative not absolute numbers, still uses a cheaper 1-run/5-step
methodology, noted separately there).

## Metrics

JAX is a trace-and-compile framework: the first call at a given
(shape, dtype, sharding) signature pays a one-time XLA compilation cost,
and every call after that runs the cached executable. That compile cost is
real (tens of seconds to minutes for these model sizes) but amortizes away
over a long-running server serving many requests at the same
resolution/step-count — so it's misleading to fold it into a single
"generation time" number the way a PyTorch-eager benchmark would. This repo
therefore reports it as its own column, separate from steady-state
generation time, rather than one blended end-to-end latency:

- **Compile time** — wall-clock for the first `single_step`/decode call at
  a given configuration (resolution, frame count, dtype, mesh shape), i.e.
  the one-time cost paid before any output is produced.
- **Generation time** — wall-clock for the sampling loop + VAE decode on
  every call *after* compilation, i.e. what a warm server actually pays per
  request.
- **Per-step time** — generation time divided by `--num_steps`. The most
  useful number for comparing across configs that differ only in step
  count, and for projecting cost at a step count you haven't measured.
- **Peak HBM / chip** — the highest per-device memory watermark during
  generation. Reported because it's the binding constraint for which
  `--tensor_parallel_size`/`--sequence_parallel_size` combination a given model
  and resolution actually needs (see
  [`docs/hardware_and_sharding.md`](hardware_and_sharding.md)) — not just a
  secondary detail, since several models in this repo (A14B in particular)
  are memory-bound before they're compute-bound on smaller chip counts.

All numbers are for a single generation request (batch size 1, no
concurrent requests), classifier-free guidance on (2x batch internally).
The **Hardware** column names the TPU generation and chip count. The
**I/O dtype** column is the compute dtype for activations/latents/VAE/text
encoder (`--dtype`); **Weight dtype** is specifically the DiT's own weight
dtype (`--dit_dtype` where a model exposes that flag separately, `--dtype`
otherwise — see the note below the table for why Wan2.1 needs the split).

## Results

Model order matches the root [`README.md`](../README.md#model-support--parity-matrix)'s
model-support table. Tasks get separate rows by default (T2V/I2V typically
differ in resolution/steps/shift, so one shared row would misrepresent one
of them) — merge tasks into a single "T2V/I2V/..." row only when their
configs are genuinely identical, as documented per-row when that's the case
(e.g. Cosmos-Predict2.5's T2V/I2V/V2V share the same resolution/frames/steps
and only differ in which conditioning-mask/timestep values are passed in).

| Model | Variant | Task | Hardware | Resolution | Frames | Steps | I/O dtype | Weight dtype | Compile (s) | Generation (s) | Per-step (s) | Peak HBM/chip (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cosmos3 | Nano (16B) | T2V | v4-8 | 704x1280 | 93 | 35 | bf16 | bf16 | 64.2 | 249.9 | 7.1 | 29.5 |
| Cosmos3 | Edge (4B) | T2V | v4-8 | 480x832 | 121 | 35 | bf16 | bf16 | 64.5 | 82.9 | 2.4 | 17.0 |
| Cosmos-Predict2.5 | 14B | T2V | v4-8 | 704x1280 | 45\* | 35 | bf16 | bf16 | 92.1 | 1259.9 | 36.0 | 22.1 |
| Cosmos-Predict2.5 | 2B | T2V | v4-8 | 704x1280 | 93 | 35 | bf16 | bf16 | 112.9 | 1357.3 | 38.8 | 16.0 |
| Wan2.2 | A14B | T2V | v4-8 | 720x1280 | 81 | 50 | | | | | | |
| Wan2.2 | A14B§ | I2V | v4-8 | 544x720\*\* | 81 | 40 | bf16 | fp32 | 146.1 | 1780.0 | 44.5 | 28.3 |
| Wan2.2 | A14B§ | I2V | v4-8 | 832x1104\*\* | 33 | 40 | bf16 | fp32 | 102.9 | 1962.5 | 49.1 | 20.5 |
| Wan2.2 | 5B¶ | T2V | v4-8 | 704x1280 | 121 | 50 | bf16 | fp32 | 87.3 | 525.9 | | 18.3 |
| Wan2.2 | 5B¶ | I2V | v4-8 | 704x1280 | 121 | 40 | bf16 | fp32 | 145.8 | 482.7 | | 18.3 |
| Wan2.1 | 14B‡ | T2V | v4-8 | 720x1280 | 81 | 50 | bf16 | fp32 | 108.2 | 6150.5 | 123.0 | 23.0 |
| Wan2.1 | 14B | T2V | v4-8 | 480x832 | 81 | 50 | bf16 | bf16 | 142.5 | 1306.8 | 26.1 | 17.2 |
| Wan2.1 | 14B (720P)‡ | I2V | v4-8 | 832x1104\*\* | 81 | 40 | bf16 | fp32 | 131.3 | 5090.0 | 127.2 | 32.7 |
| Wan2.1 | 14B (480P) | I2V | v4-8 | 544x720\*\* | 81 | 40 | bf16 | bf16 | 150.3 | 1125.3 | 28.1 | 22.1 |
| Wan2.1 | 1.3B | T2V | v4-8 | 480x832 | 81 | 50 | bf16 | bf16 | 85.4 | 348.3 | 7.0 | 10.2 |

Resolution/frame/step columns are each model's reference default (see its
own guide's CLI reference) — not necessarily the resolution that fits this
repo's current hardware today (A14B in particular; see
[`docs/models/wan2_2.md`](models/wan2_2.md)'s A14B sections). Every row uses
the same standardized prompt/conditioning-image/conditioning-video across
every model — see [`benchmarks/common.py`](../benchmarks/common.py) — so
results are comparable across model families, not just across variants of
one model.

\* Cosmos-Predict2.5 14B's reference default (93 frames) didn't fit this
machine's 4 chips at any `--tensor_parallel_size`/`--sequence_parallel_size`
split (`tp=4,sp=1` needed ~22.6G/chip with 18.5G free; `tp=2,sp=2` needed
~13.0G/chip with only 9.1G free — weight-sharding four ways beats splitting
across tensor+sequence parallelism here, since 14B's weights dominate over
activations at this frame count) — reduced to 45 frames (`tp=4`), the
largest that fits; see [`benchmarks/run_cosmos2_5.py`](../benchmarks/run_cosmos2_5.py).
Both Cosmos-Predict2.5 rows measure T2V only (I2V/V2V share the same DiT
compute cost, only conditioning-mask/timestep values differ — see
[`docs/models/cosmos2_5.md`](models/cosmos2_5.md)) — the "T2V/I2V/V2V" task label
reflects the checkpoint's supported tasks, not that all three were
separately measured.

Both Cosmos3 rows are verified correct — see
[`docs/models/cosmos3.md`](models/cosmos3.md#status) for implementation
notes. Edge's row measures T2V at its own real recipe (480x832, 121 frames,
35 steps, non-Karras `shift=10.0` — I2V uses a different recipe, 20 steps
and `shift=12.0`) — a different resolution/frame count than Nano's row, so
the two aren't directly comparable; Edge's lower generation time mainly
reflects its smaller model size, not fewer steps (both use 35 here). Both
rows use a JSON-structured version of the standardized prompt (see
[`docs/models/cosmos3.md#prompting`](models/cosmos3.md#prompting)) rather
than `benchmarks/common.py`'s plain-text default — Cosmos3 is documented to
need this format for good quality, especially Edge.

‡ Both native-720P rows require `--offload_dit_weights` (`--offload_chunk_size
20`, the empirically-chosen value for these rows — see
[`docs/weight_offloading.md`](weight_offloading.md#chunk-size-sweep)):
on this 4-chip machine, under the correct `--dit_dtype float32` default, a
fully-resident DiT otherwise leaves no HBM headroom for an unrelated
phase's own activation memory at this token count — VAE decode right after
T2V's sampling loop ends, or the conditioning image's VAE encode right
before I2V's sampling loop starts (confirmed via direct probing: the DiT's
own per-step compute already fits fine fully resident at native 720P; these
OOMs happen outside the sampling loop entirely). `--offload_dit_weights`
keeps the DiT's per-block weights host-resident, offloading `--offload_
chunk_size` blocks' worth into HBM at a time, which avoids ever having the
full tree resident outside the brief window each chunk needs it. This isn't
free even at the faster chunk size 20 used here: measured 123.0s/step for
T2V vs. 26.1s/step at 480P (more tokens too — see the "why 480P is faster"
reasoning in `docs/weight_offloading.md` — but a real per-layer-transfer/
compute-overlap cost remains on top of that). Peak HBM/chip differs
notably between the two rows at this chunk size — 23.0GB for T2V, comfortably
under budget, vs. **32.7GB for I2V**, close to this chip's real ceiling (the
extra CLIP/portrait-resolution/VAE-encode residency I2V carries on top of
the DiT leaves much less headroom at the same chunk size) — worth keeping in
mind before pushing I2V's chunk size any higher, or combining with anything
else that needs HBM. See
[`docs/lessons/wan2_1_precision_debugging.md`](lessons/wan2_1_precision_debugging.md)
for the original OOM this fixes.

§ Both A14B I2V rows need `--offload_dit_weights` **and**
`--sequence_parallel_size 2` (unlike Wan2.1, where offloading alone was
enough) — A14B computes AdaLN modulation per *token* rather than per
*sample* (see [`vidax.models.wan.wan2_2.dit`](../src/vidax/models/wan/wan2_2/dit.py)'s
module docstring), so at native resolutions its own activation memory, not
just DiT weight residency, is a real constraint that offloading alone can't
address; `--sequence_parallel_size` shards that per-token activation memory
across chips the same way it does for Wan2.1's sequence length. Composing
the two exposed a real, separate bug along the way — `nn.Dense`'s bias
getting double-counted under `sequence_parallel`'s row-parallel `psum`
reduction, invisible with near-zero random-init weights but a large,
compounding error with real trained (non-zero) biases — found and fixed in
[`vidax.models.wan.common.dit_layers.psum_row_parallel`](../src/vidax/models/wan/common/dit_layers.py)
(affected Wan2.1 too, also fixed there); see
[`docs/weight_offloading.md`](weight_offloading.md#a14b-wan22) for the full
investigation. `--offload_chunk_size` differs sharply between the two rows
because the two constraints (weight residency vs. per-token activation
memory) trade off differently at each token count: at 480P's smaller token
count there's HBM headroom to spare, so `--offload_chunk_size 10` (the
largest divisor of 40 that still fits — `20` OOMs, needing 5.8GB against
3.7GB free) both fits and measurably speeds up the per-chunk transfer/
compute pattern (44.5s/step here vs. what chunk size 1 would cost, mirroring
Wan2.1's identical "bigger chunks help, up to the memory ceiling" finding).
At native 720P, per-token activation memory alone already consumes most of
the budget, leaving no room to also hold more than `--offload_chunk_size 1`
resident, and even then the reference's full 81 frames don't fit — 33
frames is the largest that does (binary-searched: 41 frames needs ~18.1GB
against ~17.6GB free). Despite 720P's smaller `--offload_chunk_size`, its
**lower** peak HBM (20.5GB vs. 480P's 28.3GB) confirms chunk size, not
resolution, is the dominant HBM cost here — 480P's `--offload_chunk_size 10`
keeps 10 blocks' worth of fp32 weights resident at once (~10/40 of a full
~29GB/chip expert), while 720P's `--offload_chunk_size 1` keeps only 1/40
resident, more than offsetting 720P's larger per-token activation cost. Full
81-frame native 720P A14B I2V remains out of reach on this 4-chip machine
even with offloading and sequence parallelism combined; see
[`docs/weight_offloading.md`](weight_offloading.md#a14b-wan22) for why (it
would need chunking `WanDiT.pre_process` itself across the token axis, not
just the block loop — a bigger change than implemented so far).

¶ Both 5B rows use `--tensor_parallel_size 4 --sequence_parallel_size 1`
(all 4 chips shard weights, none go to sequence-parallel) despite TI2V-5B
having this repo's largest Wan token count (704x1280, 121 frames) — the
opposite tradeoff from A14B's rows above. `--sequence_parallel_size 4
--tensor_parallel_size 1` was tried first (5B's weights are small enough
that sharding activations instead of weights looked appealing), but at
`--dit_dtype float32` the ~5B DiT (~20GB) ends up fully unsharded/replicated
per chip under `tp=1`, consuming nearly the entire ~30.75GB HBM budget
before T5 prompt encoding even runs (`RESOURCE_EXHAUSTED` allocating a mere
64MB, with only 8.87MB free). `--tensor_parallel_size 2
--sequence_parallel_size 2` gets past T5 encoding but still OOMs inside the
DiT sampling step itself (39.4GB of HLO temporaries needed vs. 30.75GB
available) — at this frame count, per-token activation memory dominates
enough that even 2-way weight sharding isn't sufficient headroom. Only
`tp=4/sp=1` (full weight sharding, no sequence parallel) fits end-to-end at
the reference's full 121 frames/704x1280 — confirms 5B's real constraint is
DiT weight residency, not sequence length, unlike A14B's per-token-activation
-dominated rows above. No `--offload_dit_weights` needed at this size/config.

\*\* I2V output resolution is derived from the standardized conditioning
image's (`examples/assets/cat.jpg`, a 832x1104 portrait photo) aspect ratio
and `--max_area`, not a fixed `--height`/`--width` (see `compute_latent_grid`
in [`generate_wan2_1_i2v.py`](../examples/generate_wan2_1_i2v.py)/
[`generate_wan2_2_i2v_a14b.py`](../examples/generate_wan2_2_i2v_a14b.py)) —
so these rows' resolutions are portrait (taller than wide), unlike every
other row's fixed landscape resolution.

Wan2.1's I2V-14B ships as two separate checkpoints tuned for different
resolution ranges (`Wan2.1-I2V-14B-480P`/`720P`, identical architecture,
different weights) — both rows are the same model at its own real recipe
(`--shift` auto-selects `3.0` for 480P, `5.0` for 720P — see
[`docs/models/wan2_1.md#i2v-14b`](models/wan2_1.md#i2v-14b-generate_wan2_1_i2vpy)),
not a resolution choice made for this benchmark.

All five Wan2.1 rows' Weight dtype is `fp32`, reflecting the current
`--dit_dtype float32` default: Wan2.1's reference implementation keeps its
residual stream in float32 for virtually the whole network even under bf16
autocast, and rounding the DiT's checkpoint weights (natively float32 on
disk) down to bf16 at load — this repo's old default — causes severe,
visually obvious output corruption at large token counts (native 720P/81
frames; smaller configs happen not to accumulate enough error to see it).
`--dit_dtype` is decoupled from `--dtype` (still `bfloat16` for T5/VAE/CLIP)
— see
[`docs/lessons/wan2_1_precision_debugging.md`](lessons/wan2_1_precision_debugging.md)
for the full investigation and
[`docs/models/wan2_1.md`](models/wan2_1.md#precision-fp32-dit-weights) for
the model-doc summary. `--dit_dtype bfloat16` remains available as an
explicit opt-in for memory-constrained runs at smaller/safer token counts,
but isn't what any row above measures.

## Weight-offloading chunk-size sweep

`--offload_dit_weights` (see the `‡` footnote above and
[`docs/weight_offloading.md`](weight_offloading.md)) offloads
`--offload_chunk_size` consecutive DiT blocks per HBM buffer at a time,
default 1 (a fresh host-to-device transfer and `jax.jit` dispatch per
block, every block, every step). Grouping more blocks per chunk trades some
of the memory this technique frees back for fewer, larger transfers and
more within-chunk operator fusion — first swept cheaply with
[`benchmarks/sweep_offload_chunks.py`](../benchmarks/sweep_offload_chunks.py)
(reuses `generate_wan2_1_t2v.py`'s real `main(args)`, same as every
`run_*.py` script) on Wan2.1 14B T2V at native 720P, `--num_runs 1` and
`--num_steps 5` per chunk size instead of this doc's usual 5 runs / full
step count — only steady-state per-step time and peak HBM were of interest
for the initial sweep (final-output coherence was already verified
separately at `--offload_chunk_size 1`, see `docs/weight_offloading.md`):

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 | 141.7 | 15.2 |
| 2 | 136.3 | 15.2 |
| 4 | 133.8 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model) | 111.3 | 26.1 |

Larger chunks help, but only modestly (~21% faster from 1 to 40 -- the
whole 40-layer DiT offloaded as a single chunk, i.e. the entire tree
re-transferred fresh every step), while peak HBM grows roughly
proportionally with chunk size once it's no longer small relative to the
non-weight baseline (jumping from ~15GB to 23GB going from 8 to 20, since
20 blocks is already half the model). Even `--offload_chunk_size 40` stays
far short of this table's non-offloaded ~26-30s/step baseline (see the 480P
rows above) -- grouping more blocks per transfer isn't the main lever here.
Two likely reasons, neither investigated further in this round: (1) a chunk
never needs re-transferring if its weights never change between steps, but
this implementation's `jax.device_put` runs fresh every step regardless of
chunk size, wastefully re-paying the transfer even at `--offload_chunk_size
40` where the "chunk" is just the whole static DiT; (2) splitting the
forward pass into three separately-`jax.jit`-compiled pieces
(`pre_process`/chunk loop/`post_process`, needed so the outer chunk loop
stays untraced -- see `docs/weight_offloading.md`) loses end-to-end operator
fusion across those boundaries regardless of chunk granularity, unlike the
non-offloaded path's single fused `single_step`.

**`--offload_chunk_size 20` was then confirmed with the full standard
methodology** (`--num_runs 5`, full step count) for both models, to give a
fair apples-to-apples comparison against every other row in this table
(not just T2V's quick sweep) -- these are exactly the numbers in the two
native-720P rows above:

| Model | `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| --- | ---: | ---: | ---: |
| T2V | 1 | 130.0 | 15.2 |
| T2V | 20 | 123.0 | 23.0 |
| I2V | 1 | 137.9 | 19.1 |
| I2V | 20 | 127.2 | 32.7 |

T2V's full-run result (123.0s/step) matches the quick sweep's 5-step
estimate (123.7s/step) closely, validating that shorter/`--num_runs 1`
sweeps are a reasonable way to screen chunk sizes before committing to a
full 5-run measurement. Both models get a modest, consistent win from
`--offload_chunk_size 20` over the default (~5.4% for T2V, ~7.8% for I2V)
at a real HBM cost -- but that cost lands very differently: T2V's 23.0GB
still leaves comfortable headroom, while I2V's 32.7GB is close to this
chip's real ceiling (I2V's extra CLIP/portrait-resolution/VAE-encode
residency leaves much less room to begin with, see the `‡` footnote). This
doc's two native-720P rows use `--offload_chunk_size 20` given that
confirmed, modest-but-real win; `--offload_chunk_size 1` remains the safer
choice if you're combining `--offload_dit_weights` with anything else that
also needs HBM headroom, especially for I2V.

---

See [`docs/models/wan2_1.md`](models/wan2_1.md)/
[`docs/models/wan2_2.md`](models/wan2_2.md)/
[`docs/models/cosmos2_5.md`](models/cosmos2_5.md)/
[`docs/models/cosmos3.md`](models/cosmos3.md) for per-model implementation
and verification status, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
sharding/parallelism engineering behind the Peak HBM/chip numbers above.
