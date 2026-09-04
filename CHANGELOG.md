# Changelog

All notable changes to `vidax` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0:
a minor bump for new models / capabilities / breaking API changes, a patch bump
for fixes and docs). See [`docs/releasing.md`](docs/releasing.md) for the
release process.

## [Unreleased]

_Small, non-code changes (benchmark-number refreshes, README link fixes, etc.)
land here between releases and ship with the next tagged version._

## [0.1.0] - UNRELEASED

First public release.

### Added

- **JAX/Flax inference engine** for video Diffusion Transformers on Google
  Cloud TPU (v4/v5e/v6e), with explicit PyTree model architectures and no
  framework overhead.
- **PyTorch → JAX weight translator** (`vidax.translator`): loads PyTorch
  `.safetensors`/`.pth` checkpoints straight into Flax pytrees, with automatic
  key mapping and layout transposition, verified against every supported model
  by exact 1:1 parameter-tree matches.
- **Parallelism**: Megatron-style tensor parallelism and DeepSpeed-Ulysses
  sequence parallelism, composable via a 3-axis `(dp, tp, sp)` device mesh
  (`vidax.core.build_tpu_mesh`).
- **Per-layer weight offloading** (`--offload_dit_weights`): host-resident DiT
  weights streamed into HBM one chunk at a time, extending every model's
  resolution/frame-count reach on a given chip count.
- **Attention kernels** (`vidax.core`): a real Pallas/Mosaic TPU flash-attention
  kernel, plus `scan`/`vmap` windowed neighborhood attention where no native
  TPU kernel exists.
- **Schedulers** (`vidax.schedulers`): deterministic and ancestral (SDE) Euler
  flow-matching samplers and a from-scratch UniPC multistep
  predictor-corrector, covering every supported model's native schedule.
- **Model coverage** — standalone inference scripts under `examples/`, usage
  guides under `docs/models/`, and measured latency/memory in
  [`docs/benchmarking.md`](docs/benchmarking.md):
  - Cosmos3 — Nano (16B), Edge (4B) — T2V/I2V
  - Cosmos-Predict2.5 — 2B, 14B — T2V/I2V/V2V
  - Wan2.2 — TI2V-5B, A14B (T2V & I2V) — T2V/I2V
  - Wan2.1 — 1.3B, 14B (T2V), 14B I2V (480P/720P)
  - LTX-2.5 — 22B dev/distilled — T2V/I2V (conv + diffusion VAE decoders)
  - LTX-Video 0.9.8 — 2B/13B distilled, 13B dev — T2V/I2V
  - HunyuanVideo-1.5 — 8.3B — T2V/I2V (480p & 720p)
  - HunyuanVideo 1.0 — 13B — T2V/I2V
  - CogVideoX / CogVideoX1.5 — 2B, 5B, 5B-I2V, 1.5-5B, 1.5-5B-I2V
- **Library API**: `import vidax` exposes `__version__` and lazy re-exports of
  the common entry points; `vidax.core` / `vidax.schedulers` /
  `vidax.translator` are usable standalone. See
  [`docs/library_usage.md`](docs/library_usage.md) for the guide and
  [`docs/api/`](docs/api/index.md) for the per-function reference.

### Licensing

- Apache-2.0 overall, with a `NOTICE` file attributing every upstream project
  and its source-code license. The HunyuanVideo / HunyuanVideo-1.5 / LTX-2.5
  re-implementations are additionally subject to their upstream community
  licenses (Tencent Hunyuan Community License, LTX-2.x Community License) — see
  `NOTICE` and the `LICENSE` file in each affected directory.

[Unreleased]: https://github.com/FlyingGiraffe/vidax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FlyingGiraffe/vidax/releases/tag/v0.1.0
