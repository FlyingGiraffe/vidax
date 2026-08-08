# Benchmarking

Inference-latency and memory benchmarks for every implemented model, on
Google Cloud TPU. **Empty cells mean that combination hasn't been measured
yet** — not that it's unsupported; see each model's own
[`docs/models/`](models/) guide for implementation/verification status.

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
  `--tensor_parallel_size`/`--sequence_parallel` combination a given model
  and resolution actually needs (see
  [`docs/hardware_and_sharding.md`](hardware_and_sharding.md)) — not just a
  secondary detail, since several models in this repo (A14B in particular)
  are memory-bound before they're compute-bound on smaller chip counts.

All numbers are for a single generation request (batch size 1, no
concurrent requests), classifier-free guidance on (2x batch internally),
`bfloat16` compute. The **Hardware** column names the TPU generation and
chip count.

## Results

| Model | Variant | Task | Hardware | Resolution | Frames | Steps | Compile (s) | Generation (s) | Per-step (s) | Peak HBM/chip (GB) |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Wan2.1 | 1.3B | T2V | v4-8 | 480x832 | 81 | 50 | | | | |
| Wan2.1 | 14B | T2V | v4-8 | 480x832 | 81 | 50 | | | | |
| Wan2.1 | 14B | I2V | v4-8 | 720x1280 | 81 | 40 | | | | |
| Wan2.2 | 5B (TI2V) | T2V | v4-8 | 704x1280 | 121 | 50 | | | | |
| Wan2.2 | 5B (TI2V) | I2V | v4-8 | 704x1280 | 121 | 40 | | | | |
| Wan2.2 | A14B | T2V | v4-8 | 720x1280 | 81 | 50 | | | | |
| Wan2.2 | A14B | I2V | v4-8 | 720x1280 | 81 | 40 | | | | |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | v4-8 | | | | | | | |
| Cosmos 3 | Nano (16B) | T2V/I2V | v4-8 | | | | | | | |
| Cosmos 3 | Edge (4B) | T2V/I2V | v4-8 | | | | | | | |

Resolution/frame/step columns are each model's reference default (see its
own guide's CLI reference) — not necessarily the resolution that fits this
repo's current hardware today (A14B in particular; see
[`docs/models/wan2_2.md`](models/wan2_2.md)'s A14B sections).

---

See [`docs/models/wan2_1.md`](models/wan2_1.md)/
[`docs/models/wan2_2.md`](models/wan2_2.md)/
[`docs/models/cosmos.md`](models/cosmos.md)/
[`docs/models/cosmos3.md`](models/cosmos3.md) for per-model implementation
and verification status, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
sharding/parallelism engineering behind the Peak HBM/chip numbers above.
