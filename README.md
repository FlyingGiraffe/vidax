# vidax 🎬⚡

**`vidax`** is a lightweight JAX/Flax inference engine and
PyTorch-to-JAX weight translator for modern Video Diffusion Transformers
(DiTs) and beyond. Built for **Google Cloud TPUs (v4, v5e, v6e)**, it
eliminates framework overhead with clean, explicit PyTree architectures and
native multi-chip parallelism (Megatron tensor parallelism and
DeepSpeed-Ulysses sequence parallelism) across five architecturally distinct
model families: **Wan 2.1/2.2**, **Cosmos-Predict2.5**, **Cosmos 3** (Nano
and Edge — an omnimodal Mixture-of-Transformers, not a DiT continuation of
Cosmos-Predict2.5), **LTX-Video (0.9.8)**, and **LTX-2.5**.

## 🔑 Key Features

- **🚀 Native TPU performance:** `jax.sharding` device meshes and a real
  Pallas flash-attention kernel, not `jax.nn`'s materializing default.
- **🔄 Universal weight translator:** loads PyTorch `.safetensors`/`.pth`
  checkpoints straight into Flax pytrees — key mappings and layout
  transpositions handled automatically, verified against every model via
  exact 1:1 parameter-tree matches.
- **🧵 Two parallelism strategies:** Megatron-style tensor parallelism and
  DeepSpeed-Ulysses sequence parallelism, picked per model/resolution
  depending on whether weight or activation memory is the bottleneck — see
  [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md).
- **💾 Per-layer weight offloading:** `--offload_dit_weights` keeps a DiT's
  weights host-resident and streams one `--offload_chunk_size`-block group
  into HBM at a time, extending every model's reach to resolutions/frame
  counts that don't fit fully device-resident on a given chip count — see
  [`docs/weight_offloading.md`](docs/weight_offloading.md).
- **🌊 Flow matching engine:** a Rectified Flow Euler sampler and a
  from-scratch UniPC multistep predictor-corrector port, covering every
  supported model's native scheduler (including Cosmos 3's Karras-sigma
  variant).
- **🖼️ Faithful image/video conditioning:** each model family's own
  conditioning mechanism ported exactly, not approximated — CLIP
  cross-attention, per-token/per-frame latent substitution, and
  conditioning-mask channels all show up where the reference actually uses
  them.
- **🧩 Broad, growing model coverage:** DiTs (Wan 2.1/2.2, Cosmos-Predict2.5)
  and beyond — Cosmos 3's omnimodal Mixture-of-Transformers, a genuinely
  different architecture, ported with the same care and verification bar.

## 🎲 Model Support

Rows are merged across tasks when one script/checkpoint handles all of them
(e.g. Wan2.2 TI2V-5B, Cosmos-Predict2.5); kept separate when the reference
ships them as genuinely distinct checkpoints/pipelines (e.g. Wan2.1's T2V vs.
I2V, Wan2.2's A14B).

| Model Family | Variant | Task | TPU test (v4/v5e/v6e) | Guide | Weights |
| --- | --- | --- | --- | --- | --- |
| Cosmos3 | Nano (16B) | T2V/I2V | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | [🤗](https://huggingface.co/nvidia/Cosmos3-Nano) |
| Cosmos3 | Edge (4B) | T2V/I2V | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | [🤗](https://huggingface.co/nvidia/Cosmos3-Edge) |
| Cosmos-Predict2.5 | 14B | T2V/I2V/V2V | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | [🤗](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B) |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | [🤗](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) |
| Wan2.2 | A14B | T2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2 | A14B | I2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2 | 5B | T2V/I2V | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.1 | 14B | T2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1 | 14B (720P) | I2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan2.1 | 14B (480P) | I2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1 | 1.3B | T2V | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | [🤗](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| LTX-2.5 | 22B (dev) | T2V/I2V | ✅/❌/❌ | [ltx2_5.md](docs/models/ltx2_5.md) | [🤗](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX-2.5 | 22B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx2_5.md](docs/models/ltx2_5.md) | [🤗](https://huggingface.co/Lightricks/LTX-2.5) |
| LTX-Video (0.9.8) | 13B (dev) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | [🤗](https://huggingface.co/Lightricks/LTX-Video) |
| LTX-Video (0.9.8) | 13B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | [🤗](https://huggingface.co/Lightricks/LTX-Video) |
| LTX-Video (0.9.8) | 2B (distilled) | T2V/I2V | ✅/❌/❌ | [ltx_video.md](docs/models/ltx_video.md) | [🤗](https://huggingface.co/Lightricks/LTX-Video) |
| HunyuanVideo1.5 | 8.3B | T2V/I2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| HunyuanVideo | 13B | T2V/I2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/tencent/HunyuanVideo) |
| CogVideoX1.5 | 5B | I2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) |
| CogVideoX1.5 | 5B | T2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/THUDM/CogVideoX1.5-5B) |
| CogVideoX | 5B | I2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/THUDM/CogVideoX-5b-I2V) |
| CogVideoX | 5B | T2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/THUDM/CogVideoX-5b) |
| CogVideoX | 2B | T2V | ❌/❌/❌ | To appear | [🤗](https://huggingface.co/THUDM/CogVideoX-2b) |

Per-model checkpoint sources, CLI flags, architecture notes, and
verification status live in each **Guide** link above. Measured
latency/memory numbers for every row above live in
[`docs/benchmarking.md`](docs/benchmarking.md).

## 🚀 Quickstart

```bash
# Clone and install (editable, with TPU / torch-checkpoint-loading / tokenizer extras)
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e ".[tpu,torch,text]"

# Generate a video (Wan2.1 T2V, 1.3B)
python examples/generate_wan2_1_t2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

## 🛠 Directory Layout

Standard Python `src`-layout: one subpackage per model family under
`models/`, one usage guide per family under `docs/models/`, one standalone
inference script per family/task under `examples/`. See
[`docs/directory_layout.md`](docs/directory_layout.md) for the full tree.

## 📚 References

**Wan** — developed by Alibaba's Wan team.
- Wan2.1: [code](https://github.com/Wan-Video/Wan2.1) | [report](https://arxiv.org/abs/2503.20314) | [weights](https://huggingface.co/Wan-AI)
- Wan2.2: [code](https://github.com/Wan-Video/Wan2.2) | [report](https://arxiv.org/abs/2503.20314) | [weights](https://huggingface.co/Wan-AI)

**Cosmos-Predict2.5** — developed by NVIDIA.
- [code](https://github.com/nvidia-cosmos/cosmos-predict2.5) | [report](https://arxiv.org/abs/2511.00062) | [weights](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)

**Cosmos 3** — developed by NVIDIA.
- [code](https://github.com/NVIDIA/cosmos) | [report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | [weights](https://huggingface.co/nvidia/Cosmos3-Nano)

**LTX-Video** — developed by Lightricks.
- LTX-Video (0.9.8): [code](https://github.com/Lightricks/LTX-Video) | [report](https://arxiv.org/abs/2501.00103) | [weights](https://huggingface.co/Lightricks/LTX-Video)

**LTX-2.5** — developed by Lightricks.
- [code](https://github.com/Lightricks/LTX-2) | [weights](https://huggingface.co/Lightricks/LTX-2.5)

**Parallelism techniques implemented in this repo:**
- Megatron-style tensor parallelism — Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053).
- DeepSpeed-Ulysses sequence parallelism — Jacobs et al., [*DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*](https://arxiv.org/abs/2309.14509).

See [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) for how
both are implemented here.

## 🙏 Acknowledgments

This project was supported by the [Google Cloud TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/).
