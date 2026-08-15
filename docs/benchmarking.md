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
`out/cosmos_3_nano_t2v/cosmos_3_nano_t2v_1.mp4`).

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
concurrent requests), classifier-free guidance on (2x batch internally),
`bfloat16` compute. The **Hardware** column names the TPU generation and
chip count.

## Results

Model order matches the root [`README.md`](../README.md#model-support--parity-matrix)'s
model-support table. Tasks get separate rows by default (T2V/I2V typically
differ in resolution/steps/shift, so one shared row would misrepresent one
of them) — merge tasks into a single "T2V/I2V/..." row only when their
configs are genuinely identical, as documented per-row when that's the case
(e.g. Cosmos-Predict2.5's T2V/I2V/V2V share the same resolution/frames/steps
and only differ in which conditioning-mask/timestep values are passed in).

| Model | Variant | Task | Hardware | Resolution | Frames | Steps | Compile (s) | Generation (s) | Per-step (s) | Peak HBM/chip (GB) |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cosmos3 | Nano (16B) | T2V/I2V | v4-8 | 1280x704 | 93 | 35 | 64.2 | 249.9 | 7.1 | 29.5 |
| Cosmos3 | Edge (4B) | T2V/I2V | v4-8 | 832x480 | 121 | 35 | 64.5 | 82.9 | 2.4 | 17.0 |
| Cosmos-Predict2.5 | 14B | T2V/I2V/V2V | v4-8 | 1280x704 | 45\* | 35 | 92.1 | 1259.9 | 36.0 | 22.1 |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | v4-8 | 1280x704 | 93 | 35 | 112.9 | 1357.3 | 38.8 | 16.0 |
| Wan2.2 | A14B | T2V | v4-8 | 720x1280 | 81 | 50 | | | | |
| Wan2.2 | A14B | I2V | v4-8 | 720x1280 | 81 | 40 | | | | |
| Wan2.2 | 5B (TI2V) | T2V | v4-8 | 704x1280 | 121 | 50 | | | | |
| Wan2.2 | 5B (TI2V) | I2V | v4-8 | 704x1280 | 121 | 40 | | | | |
| Wan2.1 | 14B | T2V | v4-8 | 480x832 | 81 | 50 | | | | |
| Wan2.1 | 14B (480P) | I2V | v4-8 | 720x1280 | 81 | 40 | | | | |
| Wan2.1 | 1.3B | T2V | v4-8 | 480x832 | 81 | 50 | | | | |

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

---

See [`docs/models/wan2_1.md`](models/wan2_1.md)/
[`docs/models/wan2_2.md`](models/wan2_2.md)/
[`docs/models/cosmos2_5.md`](models/cosmos2_5.md)/
[`docs/models/cosmos3.md`](models/cosmos3.md) for per-model implementation
and verification status, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
sharding/parallelism engineering behind the Peak HBM/chip numbers above.
