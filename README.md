# vidax 🎬⚡

**`vidax`** is a lightweight, hardware-agnostic JAX/Flax inference engine and PyTorch-to-JAX weight translator designed for modern Video Diffusion Transformers (DiTs). 

Built specifically for **Google Cloud TPUs (v4, v5e, v6e)**, `vidax` eliminates framework overhead, providing clean, explicit PyTree architectures, zero-recompilation JIT compilation, and native multi-chip sequence parallelism for state-of-the-art video models like **Wan 2.1/2.2** and **NVIDIA Cosmos**.


## 🔑 Key Features

- **🚀 Native TPU Performance:** Built on `jax.sharding` device meshes and `jax.nn.dot_product_attention` for TPU v4/v5e/v6e.
- **🔄 Universal Weight Translator:** Convert PyTorch `.safetensors` checkpoints on the fly—handling key mappings and array transpositions (`NCHW` / `NDHWC` $\rightarrow$ `NHWC` / `T_H_W_In_Out`) automatically.
- **🧩 Zero-Bloat PyTrees:** Explicit, functional JAX/Flax architecture implementations without monolithic enterprise abstractions.
- **⚡ 3D Causal VAE:** Channels-last video VAE decoder, matching the reference's frame-chunked causal decoding exactly.
- **📐 3D Spatial-Temporal RoPE:** Vectorized 3D Rotary Position Embeddings matching Wan2.1's T/H/W frequency split.
- **📝 UMT5-XXL Text Encoder:** Full JAX port of Wan2.1's text conditioning encoder, including its per-layer relative-position bias.
- **🖼️ I2V Image Conditioning:** CLIP ViT-H/14 vision tower + VAE encoder + `WanI2VCrossAttention`, for image-to-video generation (14B model only).
- **🌊 Flow Matching Engine:** Rectified Flow Euler sampler.


## 🛠 Project Architecture & Directory Layout

`vidax` strictly adheres to the standard Python `src`-layout to ensure robust packaging and isolated imports:

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata (Flit/Hatch)
├── README.md                  # Project context & documentation
├── examples/
│   ├── generate_wan2_1.py     # Standalone TPU inference script (t2v)
│   └── generate_wan2_1_i2v.py # Standalone TPU inference script (i2v, 14B only)
└── src/
    └── vidax/                  # Core Python Package
        ├── __init__.py
        ├── core/              # XLA & Hardware Primitives
        │   ├── __init__.py
        │   ├── attention.py   # RMSNorm + dot-product attention wrapper
        │   ├── rope3d.py      # 3D RoPE (T/H/W split) & sinusoidal time embedding
        │   └── sharding.py    # TPU Mesh & NamedSharding topology maps
        ├── models/            # Native JAX Model Implementations
        │   ├── __init__.py
        │   └── wan/
        │       ├── __init__.py
        │       ├── dit.py          # Wan2.1 DiT Backbone (Flax Linen), t2v + i2v
        │       ├── vae.py          # Wan2.1 3D Causal Video VAE, encoder + decoder
        │       ├── t5.py           # UMT5-XXL Text Encoder + tokenizer wrapper
        │       └── clip_vision.py  # CLIP ViT-H/14 vision tower, for i2v image conditioning
        ├── schedulers/        # Diffusion & Flow Matching Samplers
        │   ├── __init__.py
        │   └── flow_match.py  # Euler / Rectified Flow Sampler
        └── translator/        # PyTorch -> JAX Translation Bridge
            ├── __init__.py
            ├── converter.py   # Tensor layout conversion (Transposition logic)
            └── mappings.py    # Model-specific state-dict key mappings
```


## ⚡ Technical Conventions for Agents & Developers

When generating or modifying code in this repository, strictly observe the following technical standards:

### 1. Tensor Memory Layouts
- **JAX / XLA prefers Channels-Last (`NHWC` or `NDHWC`):**
  - **PyTorch Conv3D:** `[Batch, Channels, Time, Height, Width]` $\rightarrow$ **JAX Conv3D:** `[Batch, Time, Height, Width, Channels]`
  - **PyTorch Linear:** `[Out_Features, In_Features]` $\rightarrow$ **JAX Linear:** `[In_Features, Out_Features]`
- The `vidax.translator.converter` module handles these layout conversions dynamically during weight ingestion.

### 2. Sharding & TPU Topology Rules
- `vidax.core.sharding.build_tpu_mesh` builds a 2D `(dp, tp)` device mesh: `dp` (data-parallel) shards the batch, `tp` (tensor-parallel) shards attention heads and FFN channels within each DiT/T5 layer, Megatron-1D-style.
- `shard_wan_params` assigns the actual `NamedSharding`s: attention Q/K/V and the FFN up-projection are column-parallel (shard their output), attention-output and the FFN down-projection are row-parallel (shard their input — GSPMD auto-inserts the resulting all-reduce). Everything else (norms, embeddings, modulation) stays replicated, since it's small and elementwise ops against it are free regardless of the other operand's sharding.
- This exists because full-resolution DiT self-attention runs over tens of thousands of patches; without tensor parallelism the O(S² × num_heads) attention matrix alone can exceed a single TPU v4 chip's HBM. `tensor_parallel_size` must divide both `num_devices` and `num_heads` (12 for the 1.3B DiT, 64 for the T5 encoder, 40 for the 14B DiT).

### 3. Flash Attention (TPU)
- `vidax.core.attention.dot_product_attention` dispatches to a real Pallas
  flash-attention kernel (`jax.experimental.pallas.ops.tpu.flash_attention`)
  on TPU whenever no `bias`/`mask` is given -- i.e. for DiT's self- and
  cross-attention. **This matters**: `jax.nn.dot_product_attention`'s
  default `"xla"` implementation, and the `jax.nn` API in general, has no
  automatic TPU flash-attention path -- it always fully materializes the
  `(B, num_heads, S_q, S_k)` logits matrix, which for DiT self-attention
  over tens of thousands of video patches is the dominant memory cost (well
  beyond what tensor parallelism alone divides down to a usable size).
  T5's self-attention keeps its relative-position `bias` and stays on the
  materializing path, since its sequence length is small and fixed (512)
  and doesn't need flash attention's O(S) memory anyway.
- Mosaic (Pallas TPU) kernels are opaque custom calls that **GSPMD cannot
  auto-partition** -- running one on a sharded array of any kind (tensor-
  parallel *or* plain data-parallel-batched) raises `"Mosaic kernels cannot
  be automatically partitioned"` unless it's explicitly wrapped in
  `jax.experimental.shard_map.shard_map`. `dot_product_attention` does this
  itself given a `mesh` argument (threaded through `WanDiT`/`WanDiTBlock` as
  a `mesh` field) -- every caller running on more than one device must pass
  one, or the call falls back to the (correct, just slower/O(S²)) XLA path.

### 4. JIT Compilation Safety
- Keep sequence lengths, frame counts, and spatial dimensions static or explicitly padded.
- Avoid dynamic Python branching on array values inside functions decorated with `@jax.jit`.
- **Don't `jax.jit` a Python loop over many repeated forward passes** (diffusion sampling steps, VAE decode's per-latent-frame chunks). `jax.jit` traces the loop by fully unrolling it into *one* HLO program, so every iteration's intermediate buffers can end up needing to coexist in that program's memory footprint instead of being freed between iterations -- this is what caused VAE decode (which internally decodes ~20 latent-frame chunks) to OOM even though DiT sampling had already succeeded. Instead, jit only the per-iteration function (its shape/dtype signature is identical every call, so this never recompiles) and call it from a plain Python loop: `examples/generate_wan2_1.py`'s sampling loop does exactly this (with `donate_argnums` on the latents carry, so each step reuses the previous step's buffer in place); `WanVAEDecoder`'s per-chunk decode currently runs fully eagerly for the same reason (correct and simple, at some dispatch-overhead cost -- sharding it and/or jitting per-chunk would recover speed, see the note in its module docstring).


## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/FlyingGiraffe/vidax.git](https://github.com/FlyingGiraffe/vidax.git)
cd vidax

# Install package in editable mode with TPU and Dev extras
pip install -e ".[tpu,dev]"
```

### 2. Running Unit Tests

Verify local installation and TPU/XLA primitives:

```bash
pytest tests/ -v
```

### 3. End-to-End Video Generation
Run video generation directly on your Google Cloud TPU VM:

```bash
python examples/generate_wan2_1.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

Checkpoints (DiT `.safetensors`, VAE `.pth`, T5 `.pth` + its `google/umt5-xxl`
tokenizer folder) can be downloaded from the
[official Wan2.1-T2V-1.3B repo](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
Loading `.pth` checkpoints requires the `torch` extra, and the T5 tokenizer
requires the `text` extra: `pip install -e ".[torch,text]"`. Neither of
vidax's own model implementations depend on torch or transformers — they're
used solely to deserialize checkpoints and tokenize text.
By default `--tokenizer_path` is inferred as `<t5_checkpoint_dir>/google/umt5-xxl`,
matching the official repo's layout; pass it explicitly if yours differs.

`--shift` (noise-schedule shift, default 5.0) and classifier-free guidance
(`--guide_scale`, default 5.0, and `--negative_prompt`, defaulting to the
reference's quality-negative-prompt) match the reference pipeline's own
defaults and aren't optional extras — the reference always runs with both,
and skipping CFG in particular produces washed-out, low-contrast output
(the model's raw conditional prediction on its own regresses hard toward an
"average video"). See `RectifiedFlowScheduler` and `single_step` in
`generate_wan2_1.py` for why.

### 4. Tensor Parallelism, Batching, and dtype

```bash
python examples/generate_wan2_1.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --num_steps 50 --height 480 --width 832 --num_frames 81 \
  --output_path "out/output.mp4"
```

- **`--tensor_parallel_size`** (default 1): shards each model's attention
  heads / FFN channels across this many devices (see §2 above). On a TPU
  v4-8, `--tensor_parallel_size 4` (4-way tensor-parallel × 2-way
  data-parallel) is a reasonable starting point for the 1.3B model at full
  1280×720 resolution; raise it (up to 8, or however many devices you have)
  if you still hit HBM OOM. The reference PyTorch pipeline only ever runs a
  single video per `generate()` call, sharded across GPUs with `xDiT`
  context-parallelism instead — vidax uses tensor parallelism instead,
  since it's the more direct fix for the actual bottleneck (the attention
  matrix), and doesn't require reference support for it to exist here.
- **`--prompt`** accepts one or more values. A single prompt is broadcast to
  every data-parallel replica (`num_devices // tensor_parallel_size`) with
  independent noise, giving you that many samples of one prompt "for free".
  Exactly `dp_size` prompts gives one video per prompt. The underlying
  `WanDiT`/`T5Encoder` architecture supports arbitrary batch sizes (as does
  the PyTorch reference's model code), it's just that the reference
  pipeline's `generate()` never uses it — vidax's example script does.
- **`--dtype`** (`float32` | `float16` | `bfloat16`, default `bfloat16`):
  matches the reference's default (`bfloat16` for T5/DiT, `float32` for the
  VAE in the PyTorch config) while making the choice explicit and uniform
  across all three models. `float16` is exposed for completeness but TPU's
  XLA backend does not implement `float16` matmuls — it will fail at
  runtime on TPU (this is a TPU/XLA hardware limitation, not a vidax one).

### 5. Image-to-Video (I2V)

I2V only ships as a **14B** model (`Wan2.1-I2V-14B-480P`/`720P` on
[HuggingFace](https://huggingface.co/Wan-AI) — there is no 1.3B I2V
checkpoint), and additionally needs a **CLIP vision encoder** checkpoint
(`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, bundled in the
same repo) to extract image features from the conditioning frame.

```bash
python examples/generate_wan2_1_i2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --image_path "./checkpoints/Wan2.1-I2V-14B-480P/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 8 \
  --output_path "out/output_i2v.mp4"
```

Everything about tensor parallelism, dtype, CFG, and the noise-schedule
shift from §§2–4 applies unchanged (i2v's own defaults: 40 sampling steps
and shift 5.0, per the reference — pass `--shift 3.0` for 480p output,
matching the reference's own recommendation there). A few things unique to
i2v:

- **`--image_path`** is required; output resolution is derived from it, not
  set directly: `--max_area` (default 720×1280) bounds the output's pixel
  count, and `compute_latent_grid` picks the largest (height, width) at that
  budget that preserves the input image's aspect ratio and aligns to both
  the VAE's spatial stride and the DiT's patch size — matching
  `WanI2V.generate`'s own resolution selection exactly.
- Conditioning is built from the image two ways: `vidax.models.wan.vae.WanVAEEncoder`
  encodes it (as a single real frame followed by zero frames, matching the
  reference's `y` construction) into the DiT's mask+latent conditioning
  channels, and `vidax.models.wan.clip_vision.ClipVisionTransformer` extracts
  CLIP features the DiT cross-attends onto through a second, image-only K/V
  projection (`WanDiT`'s `model_type="i2v"` path). Since the 14B DiT itself
  is much larger than the 1.3B t2v model, `--tensor_parallel_size 8` (full
  8-way, on a v4-8) is a more typical starting point here than the 4 used
  for 1.3B t2v above.
- This hasn't been verified against the real I2V-14B/CLIP checkpoints yet
  (they're much larger downloads than what t2v needed) — the architecture,
  converter mappings, and full pipeline wiring are all verified against
  synthetic PyTorch-shaped weights (exact 1:1 parameter-tree matches, same
  as every other component in this repo), but not against real weights the
  way the t2v path has been. Treat it as needing the same verification pass
  t2v already went through.