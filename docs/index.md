# `vidax` Documentation

Start at the [README](../README.md) for the project overview and the model
support table. This page maps the rest of the docs.

## Guides

| Doc | What it covers |
| --- | --- |
| [`directory_layout.md`](directory_layout.md) | Full source tree, and why the package is split the way it is |
| [`library_usage.md`](library_usage.md) | Using `vidax` as a library — flash attention, schedulers, the translator, and a full model, from `import vidax` (task-oriented guide) |
| [`api/`](api/index.md) | **API reference** — every public function/class in `vidax.core`, `vidax.schedulers`, `vidax.translator`: signatures and how to call them |
| [`hardware_and_sharding.md`](hardware_and_sharding.md) | TPU meshes, tensor/sequence parallelism, JIT-safety, dtype policy |
| [`weight_offloading.md`](weight_offloading.md) | Per-layer DiT weight offloading (host RAM → HBM), chunk-size tradeoffs |
| [`benchmarking.md`](benchmarking.md) | Measured latency/memory for every supported model/config, and how to reproduce |
| [`releasing.md`](releasing.md) | Versioning scheme and the release/publish process (maintainers) |

## Per-model usage guides — [`models/`](models/)

CLI reference, checkpoint sources, and architecture notes, one per family:
[cogvideox](models/cogvideox.md) ·
[cosmos2_5](models/cosmos2_5.md) ·
[cosmos3](models/cosmos3.md) ·
[hunyuan_video](models/hunyuan_video.md) ·
[hunyuan_video1_5](models/hunyuan_video1_5.md) ·
[ltx2_5](models/ltx2_5.md) ·
[ltx_video](models/ltx_video.md) ·
[wan2_1](models/wan2_1.md) ·
[wan2_2](models/wan2_2.md)

## Porting postmortems — [`lessons/`](lessons/)

War stories and regression-prevention notes from each port, referenced from
code comments where relevant:
[cogvideox](lessons/cogvideox_debugging.md) ·
[cosmos2_5](lessons/cosmos2_5_debugging.md) ·
[cosmos3](lessons/cosmos3_debugging.md) ·
[hunyuan_video](lessons/hunyuan_video_debugging.md) ·
[hunyuan_video1_5](lessons/hunyuan_video1_5_debugging.md) ·
[ltx2_5](lessons/ltx2_5_debugging.md) ·
[ltx_video](lessons/ltx_video_debugging.md) ·
[wan2_1](lessons/wan2_1_debugging.md)

## Contributing

See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
