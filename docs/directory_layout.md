# Directory Layout

`vidax` follows the standard Python `src`-layout. `models/` holds one
subpackage per model family. `models/wan/` additionally splits into one
subpackage per released *architecture version* (`wan2_1/`, `wan2_2/`), plus a
`common/` package for building blocks shared across those versions, since
Wan2.1 and Wan2.2 are genuinely different architectures. `models/cosmos2_5/`,
`models/cosmos3/`, `models/ltx_video/`, and `models/ltx2_5/` are each flat
instead (no version/`common` split): within each of these families, every
released size is the *same* architecture at different widths/depths, not a
different version, so one `dit.py` plus a `configs.py` of named
hyperparameter presets (or, for the two LTX families, checkpoint-embedded
metadata) covers every size. `models/cosmos2_5/` and `models/cosmos3/` were
originally nested under a shared `models/cosmos/` package on the assumption
they'd share significant model code; in practice they turned out
architecturally unrelated, so they're now flat siblings under `models/`,
matching `wan/`'s own top-level positioning — the same is true of
`models/ltx_video/` and `models/ltx2_5/`. `translator/mappings/` mirrors this
layout one-for-one.

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata
├── README.md                   # Concise landing page
├── docs/
│   ├── directory_layout.md     # This file
│   ├── models/
│   │   ├── wan2_1.md           # Full CLI reference & usage for Wan2.1 scripts
│   │   ├── wan2_2.md           # Full CLI reference & usage for Wan2.2 scripts
│   │   ├── cosmos2_5.md        # Full CLI reference & usage for Cosmos-Predict2.5
│   │   ├── cosmos3.md          # Full CLI reference & usage for Cosmos 3 (Nano/Edge)
│   │   ├── ltx_video.md        # Full CLI reference & usage for LTX-Video (0.9.8)
│   │   └── ltx2_5.md           # Full CLI reference & usage for LTX-2.5
│   ├── lessons/                # Model-specific debugging postmortems
│   ├── hardware_and_sharding.md # General sharding/JIT/dtype engineering notes
│   ├── weight_offloading.md    # Per-layer DiT weight offloading (host RAM -> HBM)
│   └── benchmarking.md         # Measured latency/memory for every model/config above
├── examples/
│   ├── generate_wan2_1_t2v.py      # Wan2.1 t2v, --model_size {1.3B,14B}
│   ├── generate_wan2_1_i2v.py      # Wan2.1 i2v (14B only)
│   ├── generate_wan2_2_ti2v.py     # Wan2.2 TI2V-5B, t2v + i2v
│   ├── generate_wan2_2_t2v_a14b.py # Wan2.2 A14B t2v (two-expert MoE)
│   ├── generate_wan2_2_i2v_a14b.py # Wan2.2 A14B i2v (two-expert MoE)
│   ├── generate_cosmos2_5.py       # Cosmos-Predict2.5, --model_size {2B,14B}, text2world/image2world/video2world
│   ├── generate_cosmos3.py         # Cosmos3 Nano/Edge, --model_size {nano,edge}, text2video + image2video
│   ├── generate_ltx_video.py       # LTX-Video 0.9.8, --model_size {2b-distilled,13b-dev,13b-distilled}, t2v + i2v
│   └── generate_ltx2_5.py          # LTX-2.5, both 22B checkpoints (dev/distilled), t2v + i2v
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
        │   │   ├── configs.py           # Named hyperparameter presets: BASE_2B_CONFIG/BASE_14B_CONFIG
        │   │   ├── reason1.py           # Qwen2.5-VL-7B text tower + embedding-extraction pipeline
        │   │   ├── rope.py              # Cosmos's 3D RoPE (rotate-half convention, NTK extrapolation)
        │   │   ├── dit_layers.py        # Shared DiT attention block (per-head QK-RMSNorm)
        │   │   └── dit.py               # Cosmos-Predict2.5 DiT (2B or 14B, config-driven), AdaLN-LoRA, per-frame timestep, TP/sequence-parallel-capable
        │   ├── cosmos3/              # Cosmos 3 -- architecturally unrelated to cosmos2_5/ above
        │   │   ├── configs.py           # Named hyperparameter presets: NANO_CONFIG/EDGE_CONFIG
        │   │   ├── mrope.py             # Interleaved 3D mRoPE (distinct from both Wan's and Cosmos-Predict2.5's RoPE)
        │   │   ├── dit_layers.py        # Dual-pathway (causal "und" / full-attention "gen") attention + decoder layer, config-driven per-checkpoint toggles
        │   │   └── dit.py               # Cosmos3 DiT (Nano or Edge), no AdaLN, additive timestep injection, TP-capable
        │   ├── ltx_video/            # LTX-Video 0.9.8 -- architecturally unrelated to Wan/Cosmos
        │   │   ├── configs.py           # Checkpoint-embedded-metadata loader, not hardcoded per-variant presets
        │   │   ├── dit.py               # LTX-Video DiT, full-tensor RoPE, TP-capable
        │   │   ├── vae.py               # Causal-conv VAE: PixelNorm, pixel-unshuffle patchify, noise-conditioned decoder
        │   │   ├── patchifier.py        # Latent<->pixel coordinate bookkeeping (patch_size=1 at the transformer level)
        │   │   ├── rope.py              # Full-tensor, fractional-pixel-coordinate RoPE
        │   │   └── t5.py                # Standard (non-UMT5) T5-XXL text encoder
        │   └── ltx2_5/               # LTX-2.5 -- shares the LTX VAE lineage, otherwise unrelated to ltx_video/
        │       ├── configs.py           # Checkpoint-embedded-metadata loader (DiT, connector, VAE, Gemma-4)
        │       ├── dit.py               # LTX-2.5 DiT: cross-attention AdaLN, per-head gated attention
        │       ├── connector.py         # 8-layer embeddings connector (Gemma-4 features -> DiT cross-attention space)
        │       ├── vae.py               # Conv-decoder VAE variant (PixelNorm, pixel-unshuffle, self-normalizing encoder)
        │       ├── diffusion_vae.py     # NATTEN-based transformer VAE decoder variant (--vae_variant diffusion)
        │       ├── patchifier.py        # Latent<->pixel coordinate bookkeeping, fps-aware temporal RoPE bounds
        │       ├── rope.py              # Split (rotate-half), per-head, float64-precision RoPE
        │       └── gemma4.py            # Gemma-4 12B text encoder + embedded-tokenizer extraction
        ├── schedulers/
        │   ├── flow_match.py             # Euler / Rectified Flow Sampler (Wan)
        │   ├── unipc.py                  # Flow-matching UniPC multistep solver (Cosmos-Predict2.5, Cosmos 3)
        │   ├── ltx_rectified_flow.py     # Rectified Flow sampler, LinearQuadratic/Constant sigma schedules (LTX-Video)
        │   └── ltx2_5_ancestral_euler.py # Ancestral (SDE) / plain Euler sampler (LTX-2.5)
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
                ├── reason1.py             # Reason1 (Qwen2.5-VL-7B) text-encoder key mappings
                ├── ltx_video.py           # LTX-Video DiT/VAE/T5 key mappings
                └── ltx2_5.py              # LTX-2.5 DiT/connector/VAE/Gemma-4 key mappings
```

Neither of vidax's own model implementations depend on `torch`/`transformers`
— they're used solely to deserialize checkpoints and tokenize text.
