# vidax 🎬⚡

**`vidax`** is a lightweight, hardware-agnostic JAX/Flax inference engine and
PyTorch-to-JAX weight translator for modern Video Diffusion Transformers
(DiTs) and beyond. Built for **Google Cloud TPUs (v4, v5e, v6e)**, it
eliminates framework overhead with clean, explicit PyTree architectures and
native multi-chip parallelism (Megatron tensor parallelism and
DeepSpeed-Ulysses sequence parallelism) for models like **Wan 2.1/2.2**,
**Cosmos-Predict2.5**, and **Cosmos 3** (Nano, T2V/I2V) — the last of these
architecturally unrelated to the first two (an omnimodal
Mixture-of-Transformers, not a DiT continuation).

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
- **🖼️ Image/Video-to-Video:** Wan2.1's (14B, CLIP cross-attention + extra
  latent channel), Wan2.2's (5B, per-token-timestep latent substitution), and
  Cosmos-Predict2.5's (2B, per-frame timestep + conditioning-mask channel,
  supporting both single-frame image2world and multi-frame video2world) —
  three genuinely different conditioning mechanisms.
- **🌊 Flow Matching Engine:** Rectified Flow Euler sampler (Wan) and a
  from-scratch UniPC multistep predictor-corrector port (Cosmos-Predict2.5,
  Cosmos 3 — including Cosmos 3's Karras-sigma schedule variant).
- **🧩 Omnimodal Mixture-of-Transformers (Cosmos 3):** a dual-pathway
  (causal "understanding" + full-attention "generation") transformer with
  interleaved 3D mRoPE — architecturally unrelated to the DiT family above,
  ported against a fixed-shape `(B, seq_len, hidden)` packing instead of
  the reference's ragged multi-item batching.

See [`docs/models/wan.md`](docs/models/wan.md) / [`docs/models/cosmos.md`](docs/models/cosmos.md) /
[`docs/models/cosmos3.md`](docs/models/cosmos3.md) for full usage and
[`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) for the
engineering reasoning (and debugging history) behind the sharding/JIT/dtype
choices above.

## 🎲 Model Support

| Model Family | Variant | Task | Implemented | Tested (TPU) | Benchmarked (JAX vs PyTorch) | Status / Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Wan 2.1 | 1.3B | T2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint, output visually confirmed correct. |
| Wan 2.1 | 14B | I2V | ✅ | ❌ | ❌ | Architecture + converter mappings verified against synthetic weights only; real checkpoint not yet run. |
| Wan 2.2 | 5B | T2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint (4-way sequence parallelism). |
| Wan 2.2 | 5B | I2V | ✅ | ✅ | ❌ | Verified end-to-end on real checkpoint, incl. bundled example image. |
| Wan 2.2 | 14B (A14B) | T2V/I2V | ❌ | ❌ | ❌ | Not yet implemented (two-expert MoE variant). |
| Cosmos-Predict2.5 | 2B | T2V/I2V/V2V | ✅ | ✅ | ❌ | Verified end-to-end on real weights: coherent, prompt-matching output. Took 4 real bugs to get there (unpatchify channel order, per-step DiT recompilation, a missing DiT-internal `timestep_scale`, and — the dominant one, found last — an EDM preconditioning wrapper that didn't belong on this checkpoint's model class at all). See [`docs/models/cosmos.md`](docs/models/cosmos.md#status). |
| Cosmos 3 | Nano (16B) | T2V/I2V | ✅ | ✅ | ❌ | Verified end-to-end on real weights, coherent prompt-matching output on the *first* successful full run — the Cosmos-Predict2.5 lessons (verify pieces in isolation first; no EDM preconditioning wrapper) were applied proactively. Architecturally unrelated to the DiTs above (omnimodal Mixture-of-Transformers). Scoped to T2V/I2V only (no video2video/action/sound/Reasoner). See [`docs/models/cosmos3.md`](docs/models/cosmos3.md#status). |
| Cosmos 3 | Super (64B) / Edge (4B) | — | ❌ | ❌ | ❌ | Not yet implemented. |

Full per-model details (checkpoint sources, CLI flags, verification
methodology) live in [`docs/models/wan.md`](docs/models/wan.md),
[`docs/models/cosmos.md`](docs/models/cosmos.md), and
[`docs/models/cosmos3.md`](docs/models/cosmos3.md). Benchmarking numbers
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

`vidax` strictly adheres to the standard Python `src`-layout. `models/wan/`
and `models/cosmos/` each hold one subpackage per released architecture
version (`wan2_1/`, `wan2_2/`, `cosmos2_5/`, ...), since each version's
DiT/VAE differ enough to need their own modules, plus a `common/` package
per family for building blocks that are byte-for-byte identical across that
family's versions (verified against the reference PyTorch source, not
assumed) — e.g. Wan's UMT5-XXL text encoder and causal-VAE/DiT-attention
primitives, or Cosmos's Reason1 text encoder and DiT attention block.
Cosmos-Predict2.5 additionally reuses Wan2.1's VAE directly (no
`models/cosmos/*/vae.py` — see [`docs/models/cosmos.md`](docs/models/cosmos.md)).
`models/cosmos3/` is a separate top-level package, not nested under
`models/cosmos/` — Cosmos 3 is architecturally unrelated to
Cosmos-Predict2.5 (an omnimodal Mixture-of-Transformers, not a DiT
continuation) and reuses Wan2.2's VAE directly (see
[`docs/models/cosmos3.md`](docs/models/cosmos3.md)). `translator/mappings/`
mirrors this split one-for-one.

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # This file — concise landing page
├── docs/
│   ├── models/
│   │   ├── wan.md              # Full CLI reference & usage for all Wan scripts
│   │   ├── cosmos.md           # Full CLI reference & usage for Cosmos-Predict2.5
│   │   └── cosmos3.md          # Full CLI reference & usage for Cosmos 3 (Nano)
│   ├── hardware_and_sharding.md # Sharding/JIT/dtype engineering notes + debugging history
│   └── benchmarking.md         # JAX vs PyTorch performance (placeholder)
├── examples/
│   ├── generate_wan2_1_t2v.py     # Wan2.1 t2v (1.3B)
│   ├── generate_wan2_1_i2v.py     # Wan2.1 i2v (14B only)
│   ├── generate_wan2_2_ti2v.py    # Wan2.2 TI2V-5B, t2v + i2v
│   ├── generate_cosmos2_5.py      # Cosmos-Predict2.5 2B, text2world/image2world/video2world
│   └── generate_cosmos3_nano.py   # Cosmos3-Nano 16B, text2video + image2video
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
        │   ├── cosmos/
        │   │   ├── common/           # Building blocks shared by every Cosmos-Predict2.5 version
        │   │   │   ├── reason1.py        # Qwen2.5-VL-7B text tower + embedding-extraction pipeline
        │   │   │   ├── rope.py           # Cosmos's 3D RoPE (rotate-half convention, NTK extrapolation)
        │   │   │   └── dit_layers.py     # Shared DiT attention block (per-head QK-RMSNorm)
        │   │   └── cosmos2_5/
        │   │       └── dit.py            # Cosmos-Predict2.5 2B DiT, AdaLN-LoRA, per-frame timestep, TP/sequence-parallel-capable
        │   └── cosmos3/               # Cosmos 3 -- architecturally unrelated to models/cosmos/ above
        │       ├── common/
        │       │   ├── mrope.py          # Interleaved 3D mRoPE (distinct from both Wan's and Cosmos-Predict2.5's RoPE)
        │       │   └── dit_layers.py     # Dual-pathway (causal "und" / full-attention "gen") attention + decoder layer
        │       └── nano/
        │           └── dit.py            # Cosmos3-Nano 16B DiT, no AdaLN, additive timestep injection, TP-capable
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
                ├── wan2_2_diffusers.py    # Wan2.2 VAE key mappings, diffusers' AutoencoderKLWan layout (Cosmos3-Nano's VAE)
                ├── cosmos2_5.py           # Cosmos-Predict2.5 DiT key mappings
                ├── cosmos3.py             # Cosmos3-Nano DiT key mappings
                └── reason1.py             # Reason1 (Qwen2.5-VL-7B) text-encoder key mappings
```


Checkpoints come from the official Wan repos on HuggingFace (e.g.
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)). Neither
of vidax's own model implementations depend on `torch`/`transformers` —
they're used solely to deserialize checkpoints and tokenize text.

**→ For every other model variant (Wan2.1 I2V 14B, Wan2.2 TI2V-5B t2v/i2v,
Cosmos-Predict2.5 2B, Cosmos3-Nano 16B), full CLI flag references,
parallelism guidance, and verification status, see
[`docs/models/wan.md`](docs/models/wan.md),
[`docs/models/cosmos.md`](docs/models/cosmos.md), and
[`docs/models/cosmos3.md`](docs/models/cosmos3.md).**

## 📚 Further Reading

- [`docs/models/wan.md`](docs/models/wan.md) — Wan2.1/2.2 usage guide, CLI
  reference, and per-model verification status.
- [`docs/models/cosmos.md`](docs/models/cosmos.md) — Cosmos-Predict2.5 usage
  guide, CLI reference, architecture notes, and verification status.
- [`docs/models/cosmos3.md`](docs/models/cosmos3.md) — Cosmos 3 (Nano) usage
  guide, CLI reference, architecture notes (the dual-pathway
  Mixture-of-Transformers design, interleaved mRoPE, fixed-shape packed
  sequence), and scope (T2V/I2V only, deliberately).
- [`docs/hardware_and_sharding.md`](docs/hardware_and_sharding.md) — TPU/JAX
  engineering conventions (tensor layouts, Megatron vs. sequence
  parallelism, flash attention, JIT-compilation safety) and the debugging
  history behind them (OOM investigations, a compile-time "hang", a couple
  of subtle correctness bugs and how they were found).
- [`docs/benchmarking.md`](docs/benchmarking.md) — JAX vs. PyTorch
  performance comparisons (placeholder, not yet populated).
