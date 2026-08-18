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
| Cosmos3 | Nano (16B) | T2V | v4-8 | 1280x704 | 93 | 35 | bf16 | bf16 | 64.2 | 249.9 | 7.1 | 29.5 |
| Cosmos3 | Edge (4B) | T2V | v4-8 | 832x480 | 121 | 35 | bf16 | bf16 | 64.5 | 82.9 | 2.4 | 17.0 |
| Cosmos-Predict2.5 | 14B | T2V | v4-8 | 1280x704 | 45\* | 35 | bf16 | bf16 | 92.1 | 1259.9 | 36.0 | 22.1 |
| Cosmos-Predict2.5 | 2B | T2V | v4-8 | 1280x704 | 93 | 35 | bf16 | bf16 | 112.9 | 1357.3 | 38.8 | 16.0 |
| Wan2.2 | A14B | T2V | v4-8 | 720x1280 | 81 | 50 | | | | | | |
| Wan2.2 | A14B | I2V | v4-8 | 720x1280 | 81 | 40 | | | | | | |
| Wan2.2 | 5B (TI2V) | T2V | v4-8 | 704x1280 | 121 | 50 | | | | | | |
| Wan2.2 | 5B (TI2V) | I2V | v4-8 | 704x1280 | 121 | 40 | | | | | | |
| Wan2.1 | 14B‡ | T2V | v4-8 | 720x1280 | 81 | 50 | bf16 | fp32 | 108.2 | 6150.5 | 123.0 | 23.0 |
| Wan2.1 | 14B | T2V | v4-8 | 480x832 | 81 | 50 | bf16 | fp32 | 142.5 | 1306.8 | 26.1 | 17.2 |
| Wan2.1 | 14B (720P)‡ | I2V | v4-8 | 832x1104\*\* | 81 | 40 | bf16 | fp32 | 131.3 | 5090.0 | 127.2 | 32.7 |
| Wan2.1 | 14B (480P) | I2V | v4-8 | 544x720\*\* | 81 | 40 | bf16 | fp32 | 150.3 | 1125.3 | 28.1 | 22.1 |
| Wan2.1 | 1.3B | T2V | v4-8 | 480x832 | 81 | 50 | bf16 | fp32 | 85.4 | 348.3 | 7.0 | 10.2 |

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

\*\* Wan2.1 I2V's output resolution is derived from the standardized
conditioning image's (`examples/assets/cat.jpg`, a 832x1104 portrait photo)
aspect ratio and `--max_area`, not a fixed `--height`/`--width` (see
`compute_latent_grid` in
[`generate_wan2_1_i2v.py`](../examples/generate_wan2_1_i2v.py)) — so these
two rows' resolutions are portrait (taller than wide), unlike every other
row's fixed landscape resolution.

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
