# Directory Layout

`vidax` follows the standard Python `src`-layout. `models/` holds one
subpackage per model family. `models/wan/` additionally splits into one
subpackage per released *architecture version* (`wan2_1/`, `wan2_2/`), plus a
`common/` package for building blocks shared across those versions, since
Wan2.1 and Wan2.2 are genuinely different architectures. `models/
hunyuan_video/` follows the same version-split pattern: `common/` holds the
dual-stream/single-stream MMDiT blocks and RoPE shared between HunyuanVideo
1.0 and HunyuanVideo-1.5 (same block class shapes, confirmed by reading both
reference repos), `hunyuan_video_1_5/` holds that version's own DiT
assembly/config/VAE/text-and-vision-encoder wiring, `hunyuan_video_1_0/` (T2V
only — I2V is a separate, un-cloned upstream repo) holds HunyuanVideo 1.0's
own DiT/VAE/Llama-text/CLIP-L-text/config, fully built and checked against
real downloaded checkpoints (see `docs/models/hunyuan_video_1_0.md`).
`models/cosmos2_5/`,
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
`models/ltx_video/` and `models/ltx2_5/`. `models/cogvideo/` is flat too:
CogVideoX 1.0 and 1.5 share a single transformer class in the diffusers
reference (1.5 only toggles `patch_size_t` and the RoPE grid type), so one
config-driven `dit.py` + a `configs.py` of named presets covers all five
released checkpoints (2b / 5b / 5b-I2V / 1.5-5B / 1.5-5B-I2V).
`translator/mappings/` mirrors this layout one-for-one.

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
│   │   ├── ltx2_5.md           # Full CLI reference & usage for LTX-2.5
│   │   ├── hunyuan_video_1_5.md # Full CLI reference & usage for HunyuanVideo-1.5
│   │   ├── hunyuan_video.md    # HunyuanVideo 1.0 (T2V) -- DiT/translator only, partial port
│   │   └── cogvideox.md        # Full CLI reference & usage for CogVideoX (all 5 variants)
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
│   ├── generate_ltx2_5.py          # LTX-2.5, both 22B checkpoints (dev/distilled), t2v + i2v
│   ├── generate_hunyuan_video_1_5.py # HunyuanVideo-1.5, --resolution {480p,720p}, t2v + i2v
│   ├── generate_hunyuan_video.py   # HunyuanVideo (1.0), 720p-native single checkpoint, t2v only
│   └── generate_cogvideox.py       # CogVideoX, --variant {2b,5b,5b-i2v,1.5-5b,1.5-5b-i2v}, t2v + i2v
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
        │   ├── ltx2_5/               # LTX-2.5 -- shares the LTX VAE lineage, otherwise unrelated to ltx_video/
        │       ├── configs.py           # Checkpoint-embedded-metadata loader (DiT, connector, VAE, Gemma-4)
        │       ├── dit.py               # LTX-2.5 DiT: cross-attention AdaLN, per-head gated attention
        │       ├── connector.py         # 8-layer embeddings connector (Gemma-4 features -> DiT cross-attention space)
        │       ├── vae.py               # Conv-decoder VAE variant (PixelNorm, pixel-unshuffle, self-normalizing encoder)
        │       ├── diffusion_vae.py     # NATTEN-based transformer VAE decoder variant (--vae_variant diffusion)
        │       ├── patchifier.py        # Latent<->pixel coordinate bookkeeping, fps-aware temporal RoPE bounds
        │       ├── rope.py              # Split (rotate-half), per-head, float64-precision RoPE
        │       └── gemma4.py            # Gemma-4 12B text encoder + embedded-tokenizer extraction
        │   └── hunyuan_video/        # HunyuanVideo 1.0 / 1.5 -- shares only the dual/single-stream MMDiT block shapes
        │       ├── common/              # Shared between HunyuanVideo versions
        │       │   ├── dit_layers.py        # MMDoubleStreamBlock/MMSingleStreamBlock/SingleTokenRefiner, segment-masked flash attention
        │       │   └── rope.py              # 3D axial RoPE (interleaved-pair -- reuses vidax.core.rope3d.apply_rope3d)
        │       └── hunyuan_video_1_5/
        │           ├── configs.py           # config.json loaders + DiT/VAE kwargs builders
        │           ├── dit.py               # HunyuanVideo15DiT: T2V+I2V unified, token-concat text/glyph/vision conditioning
        │           ├── vae.py               # Causal 3D-conv VAE, channel-last, pixel-(un)shuffle Down/Upsample
        │           ├── qwen_text.py         # Qwen2.5-VL-7B MLLM wrapper (reuses cosmos2_5.reason1.Qwen2TextModel)
        │           ├── byt5.py              # byT5 glyph encoder (reuses ltx_video.t5.T5Encoder) + ByT5Mapper
        │           └── siglip.py            # SigLIP vision encoder (I2V conditioning)
        │       └── hunyuan_video_1_0/     # HunyuanVideo 1.0, T2V only (I2V is a separate, un-cloned upstream repo)
        │           ├── configs.py           # Named DiT presets + vae/config.json loader/kwargs builder
        │           ├── dit.py               # HunyuanVideo1DiT -- single-LLM-encoder text conditioning, no I2V channel-concat
        │           ├── vae.py               # AutoencoderKLCausal3D port: GroupNorm, plain strided Down/Upsample (different family from 1.5's)
        │           ├── llama_text.py        # Llama3-8B decoder tower (extracted xtuner/llava-llama-3-8b-v1_1-transformers)
        │           └── clip_text.py         # CLIP-L pooled text encoder (openai/clip-vit-large-patch14)
        │   └── cogvideo/            # CogVideoX 1.0 / 1.5 -- one diffusers transformer class covers both (flat)
        │       ├── configs.py           # Named presets for all 5 checkpoints (2b/5b/5b-I2V/1.5-5B/1.5-5B-I2V)
        │       ├── dit.py               # CogVideoXDiT: joint [text;visual] self-attn, LayerNormZero, partial-RoPE, patch_size_t (1.5)
        │       ├── vae.py               # Causal 3D-conv VAE, channels-last, temporal-chunked encode/decode w/ conv cache
        │       ├── rope.py              # 3D RoPE (linspace + "slice" grid) + 3D sincos pos-embed (2b/I2V non-rotary path)
        │       └── t5.py                # Re-exports ltx_video.t5.T5Encoder + tokenizer (t5-v1.1-xxl, seq_len 226)
        ├── schedulers/
        │   ├── flow_match.py             # Euler / Rectified Flow Sampler (Wan)
        │   ├── unipc.py                  # Flow-matching UniPC multistep solver (Cosmos-Predict2.5, Cosmos 3)
        │   ├── ltx_rectified_flow.py     # Rectified Flow sampler, LinearQuadratic/Constant sigma schedules (LTX-Video)
        │   ├── ltx2_5_ancestral_euler.py # Ancestral (SDE) / plain Euler sampler (LTX-2.5)
        │   └── cogvideox.py             # CogVideoX DDIM + DPM (v-prediction, zero-terminal-SNR, SD3 SNR shift, trailing)
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
                ├── ltx2_5.py              # LTX-2.5 DiT/connector/VAE/Gemma-4 key mappings
                ├── hunyuan_video_1_5.py   # HunyuanVideo-1.5 DiT/VAE/byT5/SigLIP key mappings (Qwen2.5-VL reuses reason1.py)
                ├── hunyuan_video_1_0.py     # HunyuanVideo 1.0 DiT key mapping (structure only, not yet checked against a real checkpoint)
                └── cogvideox.py           # CogVideoX DiT + VAE key mappings (T5 reuses ltx_video.py's map_ltx_video_t5_keys)
```

Neither of vidax's own model implementations depend on `torch`/`transformers`
— they're used solely to deserialize checkpoints and tokenize text.
