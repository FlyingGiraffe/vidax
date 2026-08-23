# Benchmarking

Inference-latency and memory benchmarks for every implemented model, on
Google Cloud TPU. **Empty cells mean that combination hasn't been measured
yet** — not that it's unsupported; see each model's own
[`docs/models/`](models/) guide for implementation/verification status.

Reproduce any row with `benchmarks/run_*.py` (see
[`benchmarks/common.py`](../benchmarks/common.py) for the shared harness and
[`benchmarks/run_all.py`](../benchmarks/run_all.py) to run every
checkpoint-available combination in one pass). Checkpoints default to
`./checkpoints/`; point elsewhere with `--checkpoint_dir` or
`VIDAX_CHECKPOINT_DIR`. Measured with `jax==0.11.0` on `TPU v4` (4 chips) —
a different JAX/libtpu version or chip generation can shift these numbers;
each `benchmarks/results/*.json` records the exact `jax_version`/
`device_kind`/`device_count`.

Every row is the average of `--num_runs` independent end-to-end runs
(default 5, cold-started with a cleared JAX compilation cache each time) —
`benchmarks/results/*.json` keeps every individual run's raw metrics.

## Metrics

- **Compile time** — one-time XLA compilation cost for a given (shape,
  dtype, sharding) signature, paid before any output. Reported separately
  from generation time since it amortizes away over many requests.
- **Generation time** — sampling loop + VAE decode, per call, after
  compilation — what a warm server actually pays per request.
- **Per-step time** — generation time / `--num_steps`; useful for comparing
  across step counts.
- **Peak HBM / chip** — highest per-device memory watermark. The binding
  constraint for which `TP/SP` split a model/resolution needs — see
  [`docs/hardware_and_sharding.md`](hardware_and_sharding.md).

All numbers are batch size 1, CFG on (2x batch internally). **TP/SP** is
`--tensor_parallel_size`/`--sequence_parallel_size`. **I/O dtype** is
`--dtype` (activations/VAE/text encoder); **Weight dtype** is the DiT's own
weight dtype (`--dit_dtype` where a model exposes it separately). **Offloading**
is `--offload_dit_weights`'s `--offload_chunk_size` where used (`chunk N`),
`-` otherwise — see [`docs/weight_offloading.md`](weight_offloading.md).

## Results

Model order matches the root [`README.md`](../README.md#-model-support).
Tasks get separate rows by default; merged into one "T2V/I2V/..." row only
when configs are genuinely identical (documented per-row).

| Model | Variant | Task | Hardware | TP/SP | Resolution | Frames | Steps | I/O dtype | Weight dtype | Offloading | Compile (s) | Generation (s) | Per-step (s) | Peak HBM/chip (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cosmos3 | Nano (16B) | T2V | v4-8 | 4/- | 704x1280 | 93 | 35 | bf16 | bf16 | - | 64.2 | 249.9 | 7.1 | 29.5 |
| Cosmos3 | Edge (4B) | T2V | v4-8 | 4/- | 480x832 | 121 | 35 | bf16 | bf16 | - | 64.5 | 82.9 | 2.4 | 17.0 |
| Cosmos-Predict2.5 | 14B | T2V | v4-8 | 4/1 | 704x1280 | 93 | 35 | bf16 | bf16 | chunk 1 | 48.5 | 4479.6 | 128.0 | 14.7 |
| Cosmos-Predict2.5 | 2B | T2V | v4-8 | 4/1 | 704x1280 | 93 | 35 | bf16 | bf16 | - | 112.9 | 1357.3 | 38.8 | 16.0 |
| Wan2.2 | A14B | T2V | v4-8 | 2/2 | 480x832 | 81 | 50 | bf16 | fp32 | chunk 10 | 65.8 | 2159.1 | 43.2 | 28.4 |
| Wan2.2 | A14B | T2V | v4-8 | 2/2 | 720x1280 | 33 | 50 | bf16 | fp32 | chunk 1 | 33.7 | 2321.9 | 46.4 | 18.1 |
| Wan2.2 | A14B | I2V | v4-8 | 2/2 | 544x720\* | 81 | 40 | bf16 | fp32 | chunk 10 | 146.1 | 1780.0 | 44.5 | 28.3 |
| Wan2.2 | A14B | I2V | v4-8 | 2/2 | 832x1104\* | 33 | 40 | bf16 | fp32 | chunk 1 | 102.9 | 1962.5 | 49.1 | 20.5 |
| Wan2.2 | 5B | T2V | v4-8 | 4/1 | 704x1280 | 121 | 50 | bf16 | fp32 | - | 87.3 | 525.9 | 10.5 | 18.3 |
| Wan2.2 | 5B | I2V | v4-8 | 4/1 | 704x1280 | 121 | 40 | bf16 | fp32 | - | 145.8 | 482.7 | 12.1 | 18.3 |
| Wan2.1 | 14B | T2V | v4-8 | 4/1 | 720x1280 | 81 | 50 | bf16 | fp32 | chunk 20 | 108.2 | 6150.5 | 123.0 | 23.0 |
| Wan2.1 | 14B | T2V | v4-8 | 4/1 | 480x832 | 81 | 50 | bf16 | bf16 | - | 142.5 | 1306.8 | 26.1 | 17.2 |
| Wan2.1 | 14B (720P) | I2V | v4-8 | 4/1 | 832x1104\* | 81 | 40 | bf16 | fp32 | chunk 20 | 131.3 | 5090.0 | 127.2 | 32.7 |
| Wan2.1 | 14B (480P) | I2V | v4-8 | 4/1 | 544x720\* | 81 | 40 | bf16 | bf16 | - | 150.3 | 1125.3 | 28.1 | 22.1 |
| Wan2.1 | 1.3B | T2V | v4-8 | 4/1 | 480x832 | 81 | 50 | bf16 | bf16 | - | 85.4 | 348.3 | 7.0 | 10.2 |
| LTX-Video (0.9.8) | 13B (dev) | T2V | v4-8 | 4/- | 1216x704 | 121 | 30 | bf16 | bf16 | - | 134.7 | 156.0 | 5.2 | 15.3 |
| LTX-Video (0.9.8) | 13B (distilled) | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16 | - | 136.4 | 104.2 | 13.0 | 15.3 |
| LTX-Video (0.9.8) | 2B (distilled) | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16 | - | 83.5 | 47.3 | 5.9 | 8.8 |


Resolution/frame/step columns are each model's reference default — not
necessarily what fits this hardware today (see each model's own
`docs/models/` guide). Every row uses the same standardized prompt/image/
video (see [`benchmarks/common.py`](../benchmarks/common.py)), so results
are comparable across model families.

## Why some rows need offloading and/or sequence parallelism

The full reasoning, investigation, and every config's numbers live in
[`docs/weight_offloading.md`](weight_offloading.md) — this is a summary of
*which* rows need what and why, in one place instead of scattered footnotes:

| Rows | Need | Why |
| --- | --- | --- |
| Wan2.1 native-720P (T2V, I2V) | offloading only | A fully-resident fp32 DiT leaves no HBM headroom for an unrelated phase (VAE decode/encode) at this token count — the DiT's own compute fits fine; offloading avoids ever holding the full tree resident outside the block it's needed for. |
| A14B (all 4 rows) | offloading **+** SP | A14B's AdaLN modulation is per-*token*, not per-sample, so activation memory (not just weight residency) is the constraint at native resolutions — offloading alone can't shrink that, sequence parallelism does. |
| Cosmos-Predict2.5 14B | offloading only | Same class of problem as Wan2.1's rows — the reference's full 93-frame default doesn't fit fully resident at any TP/SP split. |
| Wan2.2 5B | neither | Weight-sharding alone (`tp=4`) is enough — the opposite tradeoff from A14B: DiT weight residency dominates here, not per-token activation memory. |
| LTX-Video (all 3 T2V rows) | TP only | Even the 2B checkpoint's own weights fit replicated on a single chip, but the reference's full `704x1216`/121-frame token count's self-attention activations don't (confirmed OOM at `tp=1`) — `tp=4` shards both and fits every variant at the same reference resolution, no offloading or sequence parallelism needed. |

`--offload_chunk_size` varies row to row because it trades resident-weight
headroom for transfer/compute overlap — larger where there's HBM to spare
(A14B's 480P rows, `chunk 10`), forced down to `1` where activation memory
already consumes most of the budget (native-720P rows). See
[`docs/weight_offloading.md`](weight_offloading.md) for the full chunk-size
sweep and every model's numbers, and for a real correctness bug found while
combining offloading with sequence parallelism (`nn.Dense` bias
double-counted under row-parallel `psum` — fixed in
[`psum_row_parallel`](../src/vidax/models/wan/common/dit_layers.py)).

Wan2.1's I2V-14B ships as two checkpoints tuned for different resolution
ranges (`480P`/`720P`, same architecture, different weights, different
`--shift`) — both rows are that model at its own real recipe, not a
resolution choice made for this benchmark.

## Notes

\* I2V output resolution is derived from the standardized conditioning
image's aspect ratio + `--max_area`, not a fixed `--height`/`--width` — so
these rows are portrait, unlike every other row's landscape resolution.

All Wan2.1 rows use `fp32` DiT weights (`--dit_dtype float32`): the
reference keeps its residual stream in float32 even under bf16 autocast, and
rounding weights to bf16 (this repo's old default) causes visible corruption
at large token counts. See
[`docs/lessons/wan2_1_precision_debugging.md`](lessons/wan2_1_precision_debugging.md).

## Weight-offloading chunk-size sweep

Chunk sizes are cheap to screen with
[`benchmarks/sweep_offload_chunks.py`](../benchmarks/sweep_offload_chunks.py)
(`--num_runs 1 --num_steps 5`) before a full 5-run measurement — swept this
way on Wan2.1 14B T2V at native 720P:

| `--offload_chunk_size` | Per-step (s) | Peak HBM/chip (GB) |
| ---: | ---: | ---: |
| 1 | 141.7 | 15.2 |
| 2 | 136.3 | 15.2 |
| 4 | 133.8 | 15.2 |
| 8 | 131.3 | 15.3 |
| 20 | 123.7 | 23.0 |
| 40 (whole model) | 111.3 | 26.1 |

Larger chunks help modestly (~21% faster from 1 to 40); the 5-step estimate
at 20 (123.7s/step) matched the full 5-run confirmation (123.0s/step, this
doc's Wan2.1 T2V row) closely, validating the cheap-sweep methodology. See
[`docs/weight_offloading.md`](weight_offloading.md) for why larger chunks
don't help more, and every other model's chunk-size tradeoff.

---

See [`docs/models/wan2_1.md`](models/wan2_1.md)/
[`docs/models/wan2_2.md`](models/wan2_2.md)/
[`docs/models/cosmos2_5.md`](models/cosmos2_5.md)/
[`docs/models/cosmos3.md`](models/cosmos3.md) for per-model implementation
and verification status, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
sharding/parallelism engineering behind the Peak HBM/chip numbers above.
