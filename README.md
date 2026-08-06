# vidax 🎬⚡

**`vidax`** is a lightweight, hardware-agnostic JAX/Flax inference engine and
PyTorch-to-JAX weight translator for modern Video Diffusion Transformers
(DiTs). Built for **Google Cloud TPUs (v4, v5e, v6e)**, it eliminates
framework overhead with clean, explicit PyTree architectures and native
multi-chip parallelism (Megatron tensor parallelism and DeepSpeed-Ulysses
sequence parallelism) for models like **Wan 2.1/2.2**, with **NVIDIA
Cosmos** planned next.

## 🔑 Key Features

- **🚀 Native TPU Performance:** `jax.sharding` device meshes + a real Pallas
  TPU flash-attention kernel (not `jax.nn`'s materializing default).
- **🔄 Universal Weight Translator:** Converts PyTorch `.safetensors`/`.pth`
  checkpoints on the fly — key mappings and layout transpositions
  (`NCHW`/`NDHWC` → `NHWC`/`T_H_W_In_Out`) handled automatically, verified
  against every model via exact 1:1 parameter-tree matches.
- **🧵 Two parallelism strategies:** Megatron-style tensor parallelism and
  DeepSpeed-Ulysses sequence parallelism, since which one actually helps
  depends on whether *weight* or *per-token activation* memory is the
  bottleneck for a given model/resolution — see
  [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md).
- **⚡ 3D Causal VAE + 3D RoPE + UMT5-XXL text encoder:** full JAX ports,
  matching the reference's frame-chunked causal decoding and per-layer
  relative-position bias exactly.
- **🖼️ Image-to-Video:** both Wan2.1's (14B, CLIP cross-attention + extra
  latent channel) and Wan2.2's (5B, per-token-timestep latent substitution —
  a genuinely different mechanism) conditioning schemes.
- **🌊 Flow Matching Engine:** Rectified Flow Euler sampler.

See [`docs/models/wan.md`](docs/models/wan.md) for full usage and
[`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) for the
engineering reasoning (and debugging history) behind the sharding/JIT/dtype
choices above.

## Model Support & Parity Matrix

| Model Family | Variant | Task | Implemented | Tested (TPU) | Benchmarked (JAX vs PyTorch) | Status / Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Wan 2.1 | 1.3B | T2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint, output visually confirmed correct. |
| Wan 2.1 | 14B | I2V | ✅ | ❌ | ❌ | Architecture + converter mappings verified against synthetic weights only; real checkpoint not yet run. |
| Wan 2.2 | 5B | T2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint (4-way sequence parallelism). |
| Wan 2.2 | 5B | I2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint, incl. bundled example image. |
| Wan 2.2 | 14B (A14B) | T2V/I2V | ❌ | ❌ | ❌ | Not yet implemented (two-expert MoE variant). |
| Cosmos | 7B/14B | — | ❌ | ❌ | ❌ | Planned next model family. |

Full per-model details (checkpoint sources, CLI flags, verification
methodology) live in [`docs/models/wan.md`](docs/models/wan.md).
Benchmarking numbers will land in [`docs/benchmarking.md`](docs/benchmarking.md)
once that harness exists.

## 🛠 Project Architecture & Directory Layout

`vidax` strictly adheres to the standard Python `src`-layout. `models/wan/`
holds one subpackage per released architecture version (`wan2_1/`,
`wan2_2/`, ...), since each version's DiT/VAE differ enough to need their
own modules, plus a `common/` package for building blocks that are
byte-for-byte identical across versions (verified against the reference
PyTorch source, not assumed) — the UMT5-XXL text encoder, and the
causal-VAE / DiT-attention primitives every version's own `vae.py`/`dit.py`
wires together differently. `models/cosmos/` (planned) would follow the
same `common/` + per-version-subpackage shape. `translator/mappings/`
mirrors this split one-for-one.

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # This file — concise landing page
├── docs/
│   ├── models/
│   │   └── wan.md              # Full CLI reference & usage for all Wan scripts
│   ├── hardware_and_sharding.md # Sharding/JIT/dtype engineering notes + debugging history
│   └── benchmarking.md         # JAX vs PyTorch performance (placeholder)
├── examples/
│   ├── generate_wan2_1_t2v.py  # Wan2.1 t2v (1.3B)
│   ├── generate_wan2_1_i2v.py  # Wan2.1 i2v (14B only)
│   └── generate_wan2_2_ti2v.py # Wan2.2 TI2V-5B, t2v + i2v
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
        │   │   │   ├── dit.py           # Wan2.1 DiT, t2v + i2v, sequence-parallel-capable
        │   │   │   ├── vae.py           # Wan2.1 VAE, encoder + decoder, jit-per-chunk
        │   │   │   └── clip_vision.py   # CLIP ViT-H/14, for i2v image conditioning
        │   │   └── wan2_2/
        │   │       ├── dit.py           # Wan2.2 DiT, per-token timestep, sequence-parallel-capable
        │   │       └── vae.py           # Wan2.2 VAE (AvgDown3D/DupUp3D/patchify), jit-per-chunk
        │   └── cosmos/               # (planned)
        ├── schedulers/
        │   └── flow_match.py     # Euler / Rectified Flow Sampler
        └── translator/            # PyTorch -> JAX Translation Bridge
            ├── converter.py       # Tensor layout conversion (host-side, numpy)
            └── mappings/          # Model-specific state-dict key mappings
                ├── __init__.py       # load_torch_checkpoint_to_jax dispatch table
                ├── common.py         # Mappers shared by every Wan version
                ├── wan2_1.py         # Wan2.1-specific VAE/CLIP key mappings
                └── wan2_2.py         # Wan2.2-specific VAE key mappings
```

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

Checkpoints come from the official Wan repos on HuggingFace (e.g.
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)). Neither
of vidax's own model implementations depend on `torch`/`transformers` —
they're used solely to deserialize checkpoints and tokenize text.

**→ For every other model variant (Wan2.1 I2V 14B, Wan2.2 TI2V-5B t2v/i2v),
full CLI flag references, tensor/sequence-parallelism guidance, and
verification status, see [`docs/models/wan.md`](docs/models/wan.md).**

## 📚 Further Reading

- [`docs/models/wan.md`](docs/models/wan.md) — usage guide, CLI reference,
  and per-model verification status.
- [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) — TPU/JAX
  engineering conventions (tensor layouts, Megatron vs. sequence
  parallelism, flash attention, JIT-compilation safety) and the debugging
  history behind them (OOM investigations, a compile-time "hang", a couple
  of subtle correctness bugs and how they were found).
- [`docs/benchmarking.md`](docs/benchmarking.md) — JAX vs. PyTorch
  performance comparisons (placeholder, not yet populated).
