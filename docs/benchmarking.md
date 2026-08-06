# Benchmarking

Placeholder — this doc will track performance comparisons between vidax's
JAX/TPU implementations and the reference PyTorch/GPU implementations
(throughput, latency, memory footprint, cost-per-video), once benchmarking
harnesses exist.

## Planned coverage

- Wan2.1 T2V 1.3B (JAX/TPU vs. PyTorch/GPU)
- Wan2.1 I2V 14B
- Wan2.2 TI2V 5B (t2v and i2v)
- Per-device and multi-device (tensor-parallel / sequence-parallel) scaling

Nothing here yet — see [`docs/models/wan.md`](models/wan.md) for current
functional status per model, and
[`docs/hardware_and_sharding.md`](hardware_and_sharding.md) for the
TPU/JAX-specific engineering this repo already relies on.
