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
weight dtype (`--dit_dtype` where a model exposes it separately) — the
*dominant* dtype where a model's checkpoint mixes precisions (see LTX-2.5's
own footnote below: its DiT is bf16 except a small set of AdaLN tables the
checkpoint itself ships in float32). **Offloading**
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
| LTX-2.5 | 22B (dev), conv VAE | T2V | v4-8 | 4/- | 1216x704 | 121 | 30 | bf16 | bf16\*\* | chunk 8 | 87.7 | 217.9 | 7.3 | 16.7 |
| LTX-2.5 | 22B (distilled), conv VAE | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16\*\* | chunk 8 | 87.9 | 37.3 | 4.7 | 15.3 |
| LTX-2.5 | 22B (dev), diffusion VAE | T2V | v4-8 | 4/- | 1216x704 | 121 | 30 | bf16 | bf16\*\* | chunk 8 | 479.5 | 2859.6 | 95.3\*\*\* | 16.1 |
| LTX-2.5 | 22B (distilled), diffusion VAE | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16\*\* | chunk 8 | 478.2 | 2680.8 | 335.1\*\*\* | 14.8 |
| LTX-Video (0.9.8) | 13B (dev) | T2V | v4-8 | 4/- | 1216x704 | 121 | 30 | bf16 | bf16 | - | 134.7 | 156.0 | 5.2 | 15.3 |
| LTX-Video (0.9.8) | 13B (distilled) | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16 | - | 136.4 | 104.2 | 13.0 | 15.3 |
| LTX-Video (0.9.8) | 2B (distilled) | T2V | v4-8 | 4/- | 1216x704 | 121 | 8 | bf16 | bf16 | - | 83.5 | 47.3 | 5.9 | 8.8 |
| HunyuanVideo-1.5 | 8.3B (720p) | T2V | v4-8 | 4/- | 1280x720 | 121 | 30 | bf16 | bf16 | - | 410.8 | 6629.8 | 221.0 | 30.3 |
| HunyuanVideo-1.5 | 8.3B (720p) | I2V | v4-8 | 4/- | 832x1104\* | 121 | 30 | bf16 | bf16 | - | 416.7 | 6575.6 | 219.2 | 32.0 |
| HunyuanVideo-1.5 | 8.3B (480p) | T2V | v4-8 | 4/- | 832x480 | 121 | 30 | bf16 | bf16 | - | 362.3 | 3583.8 | 119.5 | 29.1 |
| HunyuanVideo-1.5 | 8.3B (480p) | I2V | v4-8 | 4/- | 544x720\* | 121 | 30 | bf16 | bf16 | - | 362.4 | 3386.4 | 112.9 | 31.4 |
| CogVideoX1.5 | 5B | T2V | v4-8 | 1/4 | 1360x768 | 81 | 50 | bf16 | bf16 | - | 306.6 | 2639.8 | 52.8 | 31.5 |
| CogVideoX1.5 | 5B | I2V | v4-8 | 1/4 | 1360x768 | 81 | 50 | bf16 | bf16 | - | 304.2 | 2639.8 | 52.8 | 31.5 |
| CogVideoX | 5B | T2V | v4-8 | 4/1 | 720x480 | 49 | 50 | bf16 | bf16 | - | 105.8 | 470.6 | 9.4 | 23.2 |
| CogVideoX | 5B | I2V | v4-8 | 4/1 | 720x480 | 49 | 50 | bf16 | bf16 | - | 106.1 | 470.6 | 9.4 | 23.3 |
| CogVideoX | 2B | T2V | v4-8 | 2/1 | 720x480 | 49 | 50 | bf16 | bf16† | - | 48.4 | 211.5 | 4.2 | 17.2 |


Resolution/frame/step columns are each model's reference default — not
necessarily what fits this hardware today (see each model's own
`docs/models/` guide). Every row uses the same standardized prompt/image/
video (see [`benchmarks/common.py`](../benchmarks/common.py)), so results
are comparable across model families.

The CogVideoX VAE decode runs eagerly (not `jax.jit`-wrapped — the
tiled/chunked loop unrolled by jit OOMs), so its cost is folded into
`generation_s`, not `compile_s` (same treatment as Wan's VAE). CogVideoX-2b
runs at `tp=2`/`dp=2` (its 30 attention heads aren't divisible by 4); the
5b / 5b-i2v rows are `tp=4`.

The **CogVideoX-1.5 rows run at their native 1360×768** (~45k visual tokens
after `patch_size_t=2`) via DeepSpeed-Ulysses sequence parallelism
(`--sequence_parallel_size 4`, `tp=1` — the two are mutually exclusive for
CogVideoX). Under plain Megatron TP that DiT-step graph never finished XLA
compilation on a v4-8 (killed after 30+ min) and the per-block activations
didn't fit a chip anyway; SP shrinks each device's block-loop sequence to
~11k tokens, compiling in ~5 min and fitting the full 81-frame clip — at
**31.5 GB/chip peak, right at the v4's HBM ceiling** (so this leaves no room
for a larger frame count or batch without also offloading). Per-step is
~7.5× the 720×480 number, roughly tracking the 3× token count plus the
all-to-all/all-gather traffic 42 blocks × 5 collectives adds. See
`docs/hardware_and_sharding.md` §3 and
`docs/lessons/cogvideox_debugging.md`.

† CogVideoX-2b's checkpoint ships as **float16** (all others bf16); it's
cast to bf16 here to keep the Weight-dtype column comparable, at a small
precision cost (fp16 has 2 more mantissa bits).

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
| LTX-2.5 (both T2V rows) | TP **+** offloading (for a different reason than every other offloaded row) | `tp=4` is required just for the 22B DiT's/12B Gemma-4's own bf16 weights to fit at all (unlike LTX-Video, where `tp=1`'s weights fit and only activations forced `tp=4`). Offloading (`--offload_chunk_size 8`, the largest divisor of 48 that still fits) is needed too, but *not* because DiT weight residency is the bottleneck (it isn't — ~6.6GB/chip at tp=4, comfortable) — the fused 48-block forward pass's own per-block activations were measured not to be freed across blocks (temp memory scaled ~linearly with block count), and offloading's side effect of splitting the trace into per-chunk `jax.jit` calls fixes that regardless of whether weight streaming itself is needed. See `docs/lessons/ltx2_5_debugging.md` for the full investigation — this combination is what got the reference's own `704x1216`/121-frame default working after it previously OOM'd even at `tp=4` alone. |
| HunyuanVideo-1.5 (all rows) | TP only (no offloading yet) | The 8.3B DiT's own bf16 weights (~16.6GB) don't fit replicated alongside the other components (Qwen2.5-VL ~14GB, VAE ~2.5GB, byT5 ~0.5GB, all simply replicated across the same mesh) on one TPU v4 chip — `tp=4` shards the DiT's Q/K/V/output/FFN Dense layers, leaving ~29GB/chip peak (see the table above), comfortable at 480p. VAE decode also needed spatial tiling (`--vae_tile_latent_size`, independent of TP) to fit the reference's real 121-frame default — see `docs/lessons/hunyuan_video_1_5_debugging.md`. |
| CogVideoX-1.5 (T2V, I2V) | SP only | The 5B DiT's bf16 weights fit replicated per chip, but at native 1360×768 the ~45k-visual-token joint attention's per-block activations don't — and the non-SP graph over that sequence never finished compiling. `--sequence_parallel_size 4` (DeepSpeed-Ulysses over the visual tokens; mutually exclusive with `--tensor_parallel_size` here) shrinks each device's block-loop sequence to ~11k tokens. Fits at 31.5GB/chip — at the ceiling, so a larger frame count would also need offloading. The 1.0 rows (2b / 5b / 5b-i2v) need neither: they fit natively at 720×480 with plain `tp`. |

`--offload_chunk_size` varies row to row because it trades resident-weight
headroom for transfer/compute overlap — larger where there's HBM to spare
(A14B's 480P rows, `chunk 10`), forced down to `1` where activation memory
already consumes most of the budget (native-720P rows). See
[`docs/weight_offloading.md`](weight_offloading.md) for the full chunk-size
sweep and every model's numbers.

Wan2.1's I2V-14B ships as two checkpoints tuned for different resolution
ranges (`480P`/`720P`, same architecture, different weights, different
`--shift`) — both rows are that model at its own real recipe, not a
resolution choice made for this benchmark.

## Notes

\* I2V output resolution is derived from the standardized conditioning
image's aspect ratio + `--max_area`, not a fixed `--height`/`--width` — so
these rows are portrait, unlike every other row's landscape resolution.

\*\* LTX-2.5's DiT weights are almost entirely bf16 (4059 of 4349 tensors
in the real checkpoint), **except** every `scale_shift_table`/
`prompt_scale_shift_table` (290 tensors — the AdaLN modulation tables,
every one of these two names at every block plus the top-level table),
which the checkpoint itself ships in float32 and this port preserves at
float32 regardless of `--dit_dtype` — downcasting them was a real,
measurable quality bug (see `docs/lessons/ltx2_5_debugging.md`). This
table's "Weight dtype" column reports the dominant/bulk dtype (bf16) per
its own convention (see the "Metrics" section above); it isn't literally
100% bf16 for this model. Everything else (VAE, Gemma-4 text encoder, the
embeddings connector) is fully bf16, matching their checkpoints exactly —
confirmed by auditing every tensor's actual stored dtype, not assumed.

\*\*\* The diffusion-VAE rows' "Per-step (s)" is not a fair per-DiT-step
cost — it's `Generation (s) / Steps`, and generation time here is
dominated by the VAE decode itself (a one-time cost per generation, not
per DiT step), not the DiT sampling loop. Compare "Generation (s)"
directly between VAE variants instead — the DiT sampling cost itself is
essentially unchanged from the conv-VAE rows (same DiT, same steps); the
diffusion VAE decode step alone (single full-volume NA tile, no tiling
yet — see `docs/models/ltx2_5.md`) accounts for the bulk of the
difference, dominated by its own compile time (`~478s`) more than actual
decode compute. See `docs/lessons/ltx2_5_debugging.md`'s "Diffusion
(NATTEN) VAE decoder: the full compile-time and memory story" for why.

All Wan2.1 rows use `fp32` DiT weights (`--dit_dtype float32`): the
reference keeps its residual stream in float32 even under bf16 autocast, and
rounding weights to bf16 causes visible corruption at large token counts. See
[`docs/lessons/wan2_1_debugging.md`](lessons/wan2_1_debugging.md).

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

See [`docs/models/cosmos3.md`](models/cosmos3.md)/
[`docs/models/cosmos2_5.md`](models/cosmos2_5.md)/
[`docs/models/wan2_2.md`](models/wan2_2.md)/
[`docs/models/wan2_1.md`](models/wan2_1.md)/
[`docs/models/ltx2_5.md`](models/ltx2_5.md)/
[`docs/models/ltx_video.md`](models/ltx_video.md)/
[`docs/models/hunyuan_video_1_5.md`](models/hunyuan_video_1_5.md)/
[`docs/models/cogvideox.md`](models/cogvideox.md) for per-model
implementation and verification status, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
sharding/parallelism engineering behind the Peak HBM/chip numbers above.
