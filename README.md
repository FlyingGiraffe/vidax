<p align="center">
  <img alt="vidax" src="assets/wordmark.svg" height="100">
</p>

<p align="center">
  <a href="https://flyinggiraffe.github.io/vidax-site/docs/intro">Documentation</a> |
  arXiv (coming soon) |
  <a href="https://flyinggiraffe.github.io/vidax-site/blog">Blog</a> |
  <a href="https://flyinggiraffe.github.io/vidax-site/benchmarks">Benchmark</a> |
  <a href="https://flyinggiraffe.github.io/vidax-site/gallery">Gallery</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <!-- Uncomment once the package is published to PyPI:
  <a href="https://pypi.org/project/vidax/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vidax.svg"></a>
  -->
</p>

**`vidax`** is a lightweight JAX/Flax inference engine and
PyTorch-to-JAX weight translator for modern Video Diffusion Transformers
(DiTs) and beyond. Built for **Google Cloud TPUs (v4, v5e, v6e)**, it
eliminates framework overhead with clean, explicit PyTree architectures and
native multi-chip parallelism (Megatron tensor parallelism and
DeepSpeed-Ulysses sequence parallelism) across architecturally distinct
model families.

<p align="center">
  <img alt="Cosmos3-Nano T2V sample generated with vidax" src="assets/demo.gif" width="100%">
</p>

## 🔑 Key Features

- **🚀 Native TPU performance:** `jax.sharding` device meshes, a real Pallas
  flash-attention kernel (not `jax.nn`'s materializing default), and a
  custom `scan`/`vmap`-based windowed neighborhood-attention kernel where no
  native TPU kernel exists.
- **🔄 Universal weight translator:** loads PyTorch `.safetensors`/`.pth`
  checkpoints straight into Flax pytrees — key mappings and layout
  transpositions handled automatically, verified against every model via
  exact 1:1 parameter-tree matches.
- **🧵 Two parallelism strategies:** Megatron-style tensor parallelism and
  DeepSpeed-Ulysses sequence parallelism, composable and picked per
  model/resolution depending on whether weight or activation memory is the
  bottleneck — see
  [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md).
- **💾 Per-layer weight offloading:** `--offload_dit_weights` keeps a DiT's
  weights host-resident and streams one `--offload_chunk_size`-block group
  into HBM at a time, extending every model's reach to resolutions/frame
  counts that don't fit fully device-resident on a given chip count — see
  [`docs/weight_offloading.md`](docs/weight_offloading.md).
- **🌊 Flow-matching sampling:** deterministic and ancestral (SDE) Euler,
  plus a from-scratch UniPC multistep predictor-corrector, covering every
  supported model's native schedule (linear, Karras-sigma, and shift-warped
  variants alike).
- **🖼️ Faithful conditioning:** each model's own image/video-conditioning
  mechanism ported exactly, not approximated — cross-attention, per-token/
  per-frame latent substitution, and conditioning-mask channels all show up
  where the reference actually uses them.
- **🧩 Broad, growing model coverage:** DiTs, dual-pathway
  Mixture-of-Transformers models, and beyond — every new architecture
  ported and verified to the same bar (exact checkpoint key/shape matches,
  bit-exact or real end-to-end checks against the reference).

## 🎲 Model Support

Rows are merged across tasks when one script/checkpoint handles all of them
(e.g. Wan2.2 TI2V-5B, Cosmos-Predict2.5); kept separate when the reference
ships them as genuinely distinct checkpoints/pipelines (e.g. Wan2.1's T2V vs.
I2V, Wan2.2's A14B).

| Model Family | Variant | Task | TPU test (v4/v5e/v6e) | Guide | Weights |
| --- | --- | --- | --- | --- | --- |
| Cosmos3 | Nano (16B) | T2V/I2V | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | 🤗[Link](https://huggingface.co/nvidia/Cosmos3-Nano) |
| Cosmos3 | Edge (4B) | T2V/I2V | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | 🤗[Link](https://huggingface.co/nvidia/Cosmos3-Edge) |
| Cosmos-Predict2.5 | 14B | T2V/I2V/V2V | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | 🤗[Link](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B) |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | 🤗[Link](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) |
| Wan2.2 | A14B | T2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2 | A14B | I2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2 | 5B | T2V/I2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.1 | 14B | T2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1 | 14B (720P) | I2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan2.1 | 14B (480P) | I2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1 | 1.3B | T2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| LTX-2.5 | 22B (dev) | T2V/I2V | ✅/❌/❌ | [ltx2_5.md](docs/models/ltx2_5.md) | 🤗[Link](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX-2.5 | 22B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx2_5.md](docs/models/ltx2_5.md) | 🤗[Link](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX-Video (0.9.8) | 13B (dev) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | 🤗[Link](https://huggingface.co/Lightricks/LTX-Video) |
| LTX-Video (0.9.8) | 13B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | 🤗[Link](https://huggingface.co/Lightricks/LTX-Video) |
| LTX-Video (0.9.8) | 2B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | 🤗[Link](https://huggingface.co/Lightricks/LTX-Video) |
| HunyuanVideo-1.5 | 8.3B | T2V/I2V | ✅/❌/❌ | [hunyuan_video1_5.md](docs/models/hunyuan_video1_5.md) | 🤗[Link](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| HunyuanVideo | 13B | T2V/I2V | ✅/❌/❌ | [hunyuan_video.md](docs/models/hunyuan_video.md) | 🤗[Link](https://huggingface.co/tencent/HunyuanVideo) |
| CogVideoX1.5 | 5B | I2V | ✅/❌/❌ | [cogvideox.md](docs/models/cogvideox.md) | 🤗[Link](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) |
| CogVideoX1.5 | 5B | T2V | ✅/❌/❌ | [cogvideox.md](docs/models/cogvideox.md) | 🤗[Link](https://huggingface.co/THUDM/CogVideoX1.5-5B) |
| CogVideoX | 5B | T2V | ✅/❌/❌ | [cogvideox.md](docs/models/cogvideox.md) | 🤗[Link](https://huggingface.co/THUDM/CogVideoX-5b) |
| CogVideoX | 5B | I2V | ✅/❌/❌ | [cogvideox.md](docs/models/cogvideox.md) | 🤗[Link](https://huggingface.co/THUDM/CogVideoX-5b-I2V) |
| CogVideoX | 2B | T2V | ✅/❌/❌ | [cogvideox.md](docs/models/cogvideox.md) | 🤗[Link](https://huggingface.co/THUDM/CogVideoX-2b) |

Per-model checkpoint sources, CLI flags, and architecture notes live in
each **Guide** link above. Measured latency/memory numbers for every row
above live in [`docs/benchmarking.md`](docs/benchmarking.md).

## 🚀 Quickstart

```bash
# Clone and install (editable). On a Cloud TPU VM add the "tpu" extra for the
# right jaxlib wheel: pip install -e ".[tpu]"
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e .

# Generate a video (Wan2.1 T2V, 1.3B)
python examples/generate_wan2_1_t2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

## 🐍 Library / Python API

Beyond the `examples/` scripts, `vidax` is usable as a library — reuse the
Pallas flash-attention kernel, the diffusion schedulers, the PyTorch→JAX
translator, or a model's DiT/VAE modules directly:

```python
from vidax.core import dot_product_attention, build_tpu_mesh
from vidax.schedulers import RectifiedFlowScheduler
from vidax.translator import load_torch_checkpoint_to_jax

params = load_torch_checkpoint_to_jax("model.safetensors", model_type="wan_dit")
out = dot_product_attention(q, k, v)          # real O(seq)-memory flash attn on TPU
```

See [`docs/library_usage.md`](docs/library_usage.md) for worked examples
(standalone attention, schedulers, checkpoint translation, and a full model),
and [`docs/api/`](docs/api/index.md) for the full per-function API reference.

## 🛠 Directory Layout

Standard Python `src`-layout: one subpackage per model family under
`models/`, one usage guide per family under `docs/models/`, one standalone
inference script per family/task under `examples/`. See
[`docs/directory_layout.md`](docs/directory_layout.md) for the full tree, and
[`docs/index.md`](docs/index.md) for the documentation map.

## 📚 References

| Model | Developer | Code | Report | Weights | License |
| --- | --- | --- | --- | --- | --- |
| Wan2.2 | Alibaba (Wan team) | [code](https://github.com/Wan-Video/Wan2.2) | [report](https://arxiv.org/abs/2503.20314) | [weights](https://huggingface.co/Wan-AI) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Wan2.1 | Alibaba (Wan team) | [code](https://github.com/Wan-Video/Wan2.1) | [report](https://arxiv.org/abs/2503.20314) | [weights](https://huggingface.co/Wan-AI) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Cosmos 3 | NVIDIA | [code](https://github.com/NVIDIA/cosmos) | [report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | [weights](https://huggingface.co/nvidia/Cosmos3-Nano) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) |
| Cosmos-Predict2.5 | NVIDIA | [code](https://github.com/nvidia-cosmos/cosmos-predict2.5) | [report](https://arxiv.org/abs/2511.00062) | [weights](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) |
| LTX-2.5 | Lightricks | [code](https://github.com/Lightricks/LTX-2) | [report](https://arxiv.org/abs/2601.03233) | [weights](https://huggingface.co/Lightricks/LTX-2.5) | [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md) |
| LTX-Video (0.9.8) | Lightricks | [code](https://github.com/Lightricks/LTX-Video) | [report](https://arxiv.org/abs/2501.00103) | [weights](https://huggingface.co/Lightricks/LTX-Video) | [LTX-Video Open Weights License](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt) |
| HunyuanVideo-1.5 | Tencent | [code](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | [report](https://arxiv.org/abs/2511.18870) | [weights](https://huggingface.co/tencent/HunyuanVideo-1.5) | [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE) |
| HunyuanVideo | Tencent | [code](https://github.com/Tencent-Hunyuan/HunyuanVideo) | [report](https://arxiv.org/abs/2412.03603) | [weights](https://huggingface.co/tencent/HunyuanVideo) | [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/LICENSE.txt) |
| CogVideoX / 1.5 | THUDM / ZhipuAI | [code](https://github.com/THUDM/CogVideo) | [report](https://arxiv.org/abs/2408.06072) | [weights](https://huggingface.co/THUDM) | [CogVideoX License](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE) |

**Parallelism techniques implemented in this repo:**
- Megatron-style tensor parallelism — Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053).
- DeepSpeed-Ulysses sequence parallelism — Jacobs et al., [*DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*](https://arxiv.org/abs/2309.14509).

See [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) for how
both are implemented here.

## 🙏 Acknowledgments

This project is supported by the Google [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) program.
