# vidax 🎬⚡

**`vidax`** is a lightweight JAX/Flax inference engine and
PyTorch-to-JAX weight translator for modern Video Diffusion Transformers
(DiTs) and beyond. Built for **Google Cloud TPUs (v4, v5e, v6e)**, it
eliminates framework overhead with clean, explicit PyTree architectures and
native multi-chip parallelism (Megatron tensor parallelism and
DeepSpeed-Ulysses sequence parallelism) for models like **Wan 2.1/2.2**,
**Cosmos-Predict2.5**, and **Cosmos 3** (Nano and Edge, T2V/I2V) — the last
of these architecturally unrelated to the first two (an omnimodal
Mixture-of-Transformers, not a DiT continuation).

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

| Model Family | Variant | Task | Implemented (Smoke test) | TPU test (v4/v5e/v6e) | Guide | Weights |
| --- | --- | --- | --- | --- | --- | --- |
| Cosmos3 | Nano (16B) | T2V/I2V | ✅ | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | 🤗[Huggingface](https://huggingface.co/nvidia/Cosmos3-Nano) |
| Cosmos3 | Edge (4B) | T2V/I2V | ✅ | ✅/❌/❌ | [cosmos3.md](docs/models/cosmos3.md) | 🤗[Huggingface](https://huggingface.co/nvidia/Cosmos3-Edge) |
| Cosmos-Predict2.5 | 14B | T2V/I2V/V2V | ✅ | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | 🤗[Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B) |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | ✅ | ✅/❌/❌ | [cosmos2_5.md](docs/models/cosmos2_5.md) | 🤗[Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) |
| Wan2.2 | A14B | T2V | ✅ | ❌/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Wan2.2 | A14B | I2V | ✅ | ❌/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2 | 5B | T2V/I2V | ✅ | ✅/❌/❌ | [wan2_2.md](docs/models/wan2_2.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| Wan2.1 | 14B | T2V | ✅ | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1 | 14B (720P) | I2V | ✅ | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |
| Wan2.1 | 14B (480P) | I2V | ❌ | ❌/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1 | 1.3B | T2V | ✅ | ✅/❌/❌ | [wan2_1.md](docs/models/wan2_1.md) | 🤗[Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| LTX-Video | 13B | T2V/I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/Lightricks/LTX-Video) |
| LTX-Video | 2B | T2V/I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/Lightricks/LTX-Video) |
| HunyuanVideo1.5 | 8.3B | T2V/I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/tencent/HunyuanVideo-1.5) |
| HunyuanVideo | 13B | T2V/I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/tencent/HunyuanVideo) |
| CogVideoX1.5 | 5B | I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) |
| CogVideoX1.5 | 5B | T2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/THUDM/CogVideoX1.5-5B) |
| CogVideoX | 5B | I2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/THUDM/CogVideoX-5b-I2V) |
| CogVideoX | 5B | T2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/THUDM/CogVideoX-5b) |
| CogVideoX | 2B | T2V | ❌ | ❌/❌/❌ | To appear | 🤗[Huggingface](https://huggingface.co/THUDM/CogVideoX-2b) |

Per-model checkpoint sources, CLI flags, architecture notes, and
verification status live in each **Guide** link above. Benchmarking numbers
will land in [`docs/benchmarking.md`](docs/benchmarking.md) once that
harness exists.

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

`vidax` follows the standard Python `src`-layout. `models/` holds one
subpackage per model family. `models/wan/` additionally splits into one
subpackage per released *architecture version* (`wan2_1/`, `wan2_2/`), plus a
`common/` package for building blocks shared across those versions, since
Wan2.1 and Wan2.2 are genuinely different architectures. `models/cosmos2_5/`
and `models/cosmos3/` are both flat instead (no version/common split): each
released size within a family (Cosmos-Predict2.5's 2B/14B, Cosmos 3's
Nano/Edge) is the *same* architecture at different widths/depths, not a
different version, so one `dit.py` plus a `configs.py` of named
hyperparameter presets covers every size in that family. (`models/cosmos2_5/`
and `models/cosmos3/` were originally nested under a shared `models/cosmos/`
package on the assumption Cosmos-Predict2.5 and Cosmos 3 would share
significant model code; in practice they turned out architecturally
unrelated, so they're now flat siblings under `models/`, matching `wan/`'s
own top-level positioning.) `translator/mappings/` mirrors this
layout one-for-one.

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # This file — concise landing page
├── docs/
│   ├── models/
│   │   ├── wan2_1.md           # Full CLI reference & usage for Wan2.1 scripts
│   │   ├── wan2_2.md           # Full CLI reference & usage for Wan2.2 scripts
│   │   ├── cosmos2_5.md         # Full CLI reference & usage for Cosmos-Predict2.5
│   │   └── cosmos3.md          # Full CLI reference & usage for Cosmos 3 (Nano/Edge)
│   ├── lessons/                 # Model-specific debugging postmortems & design proposals
│   ├── hardware_and_sharding.md # General sharding/JIT/dtype engineering notes
│   └── benchmarking.md         # JAX vs PyTorch performance (placeholder)
├── examples/
│   ├── generate_wan2_1_t2v.py     # Wan2.1 t2v, --model_size {1.3B,14B}
│   ├── generate_wan2_1_i2v.py     # Wan2.1 i2v (14B only)
│   ├── generate_wan2_2_ti2v.py    # Wan2.2 TI2V-5B, t2v + i2v
│   ├── generate_wan2_2_t2v_a14b.py # Wan2.2 A14B t2v (two-expert MoE)
│   ├── generate_wan2_2_i2v_a14b.py # Wan2.2 A14B i2v (two-expert MoE)
│   ├── generate_cosmos2_5.py      # Cosmos-Predict2.5, --model_size {2B,14B}, text2world/image2world/video2world
│   └── generate_cosmos3.py        # Cosmos3 Nano/Edge, --model_size {nano,edge}, text2video + image2video
└── src/
    └── vidax/                  # Core Python Package
        ├── __init__.py
        ├── core/                # XLA & Hardware Primitives (model-family-agnostic)
        │   ├── attention.py     # RMSNorm + dot-product/flash/sequence-parallel attention
        │   ├── rope3d.py        # 3D RoPE (T/H/W split) & sinusoidal time embedding
        │   └── sharding.py      # TPU Mesh & NamedSharding topology maps
        ├── models/
        │   ├── wan/
        │   │   ├── common/          # Building blocks shared by every Wan version
        │   │   │   ├── t5.py            # UMT5-XXL Text Encoder + tokenizer wrapper
        │   │   │   ├── vae_layers.py    # Shared causal-VAE primitives
        │   │   │   └── dit_layers.py    # Shared DiT primitives (attend(), WanHead, chunk_by_rank)
        │   │   ├── wan2_1/
        │   │   │   ├── configs.py       # Named hyperparameter presets: T2V_1_3B/T2V_14B/I2V_14B
        │   │   │   ├── dit.py           # Wan2.1 DiT, t2v + i2v, sequence-parallel-capable, config-driven size
        │   │   │   ├── vae.py           # Wan2.1 VAE, encoder + decoder, jit-per-chunk
        │   │   │   └── clip_vision.py   # CLIP ViT-H/14, for i2v image conditioning
        │   │   └── wan2_2/
        │   │       ├── configs.py       # Named hyperparameter presets: TI2V_5B/T2V_A14B/I2V_A14B
        │   │       ├── dit.py           # Wan2.2 DiT, per-token timestep, sequence-parallel-capable
        │   │       └── vae.py           # Wan2.2 VAE (AvgDown3D/DupUp3D/patchify), jit-per-chunk
        │   ├── cosmos2_5/
        │   │   ├── configs.py            # Named hyperparameter presets: BASE_2B_CONFIG/BASE_14B_CONFIG
        │   │   ├── reason1.py            # Qwen2.5-VL-7B text tower + embedding-extraction pipeline
        │   │   ├── rope.py               # Cosmos's 3D RoPE (rotate-half convention, NTK extrapolation)
        │   │   ├── dit_layers.py         # Shared DiT attention block (per-head QK-RMSNorm)
        │   │   └── dit.py                # Cosmos-Predict2.5 DiT (2B or 14B, config-driven), AdaLN-LoRA, per-frame timestep, TP/sequence-parallel-capable
        │   └── cosmos3/               # Cosmos 3 -- architecturally unrelated to cosmos2_5/ above
        │       ├── configs.py            # Named hyperparameter presets: NANO_CONFIG/EDGE_CONFIG
        │       ├── mrope.py              # Interleaved 3D mRoPE (distinct from both Wan's and Cosmos-Predict2.5's RoPE)
        │       ├── dit_layers.py         # Dual-pathway (causal "und" / full-attention "gen") attention + decoder layer, config-driven per-checkpoint toggles
        │       └── dit.py                # Cosmos3 DiT (Nano or Edge), no AdaLN, additive timestep injection, TP-capable
        ├── schedulers/
        │   ├── flow_match.py     # Euler / Rectified Flow Sampler (Wan)
        │   └── unipc.py           # Flow-matching UniPC multistep solver (Cosmos-Predict2.5, Cosmos 3)
        └── translator/            # PyTorch -> JAX Translation Bridge
            ├── converter.py       # Tensor layout conversion (host-side, numpy)
            └── mappings/          # Model-specific state-dict key mappings
                ├── __init__.py            # load_torch_checkpoint_to_jax dispatch table
                ├── common.py              # Mappers shared by every Wan version
                ├── wan2_1.py              # Wan2.1-specific VAE/CLIP key mappings
                ├── wan2_2.py              # Wan2.2-specific VAE key mappings (original repo layout)
                ├── wan2_2_diffusers.py    # Wan2.2 VAE key mappings, diffusers' AutoencoderKLWan layout (Cosmos3's VAE)
                ├── cosmos2_5.py           # Cosmos-Predict2.5 DiT key mappings
                ├── cosmos3.py             # Cosmos3 (Nano/Edge) DiT key mappings
                └── reason1.py             # Reason1 (Qwen2.5-VL-7B) text-encoder key mappings
```

Neither of vidax's own model implementations depend on `torch`/`transformers`
— they're used solely to deserialize checkpoints and tokenize text.
