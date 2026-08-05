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
- **🎞️ Wan2.2 TI2V-5B:** A second, genuinely different DiT (per-token timestep embedding, no CLIP branch) and VAE (`AvgDown3D`/`DupUp3D` shortcuts, 2x2 pixel-patchify wrapping) for Wan2.2's 5B model, alongside Wan2.1 -- see `src/vidax/models/wan/`'s `common/`+per-version-subpackage layout below.
- **🧵 Sequence Parallelism:** DeepSpeed-Ulysses-style token-sequence sharding (`WanDiT(sequence_parallel=True)`) for both Wan2.1 and Wan2.2's DiTs, as an alternative to Megatron tensor parallelism for models/resolutions where the per-token activation memory (not weight memory) is the actual bottleneck -- see `vidax.models.wan.wan2_2.dit`'s module docstring for the mechanism and why it was needed.
- **🌊 Flow Matching Engine:** Rectified Flow Euler sampler.


## 🛠 Project Architecture & Directory Layout

`vidax` strictly adheres to the standard Python `src`-layout to ensure robust packaging and isolated imports:

`models/wan/` holds one subpackage per released architecture version
(`wan2_1/`, `wan2_2/`, ...), since each version's DiT/VAE differ enough to
need their own modules, plus a `common/` package for building blocks that
are byte-for-byte identical across versions (verified against the reference
PyTorch source, not assumed) — the UMT5-XXL text encoder, and the causal-VAE
/ DiT-attention primitives every version's own `vae.py`/`dit.py` wires
together differently. `models/cosmos/` (planned) would follow the same
`common/` + per-version-subpackage shape. `translator/mappings/` mirrors
this split one-for-one, since each new model version otherwise means adding
to one increasingly-unrelated file.

```text
vidax/                          # Repository Root
├── pyproject.toml              # Build & dependency metadata (Flit/Hatch)
├── README.md                  # Project context & documentation
├── examples/
│   ├── generate_wan2_1_t2v.py # Standalone TPU inference script (Wan2.1 t2v)
│   ├── generate_wan2_1_i2v.py # Standalone TPU inference script (Wan2.1 i2v, 14B only)
│   └── generate_wan2_2_ti2v.py # Standalone TPU inference script (Wan2.2 TI2V-5B, t2v path)
└── src/
    └── vidax/                  # Core Python Package
        ├── __init__.py
        ├── core/              # XLA & Hardware Primitives (model-family-agnostic)
        │   ├── __init__.py
        │   ├── attention.py   # RMSNorm + dot-product attention wrapper
        │   ├── rope3d.py      # 3D RoPE (T/H/W split) & sinusoidal time embedding
        │   └── sharding.py    # TPU Mesh & NamedSharding topology maps
        ├── models/            # Native JAX Model Implementations
        │   ├── __init__.py
        │   └── wan/
        │       ├── __init__.py
        │       ├── common/         # Building blocks shared by every Wan version
        │       │   ├── __init__.py
        │       │   ├── t5.py           # UMT5-XXL Text Encoder + tokenizer wrapper
        │       │   ├── vae_layers.py   # Shared causal-VAE primitives (ResidualBlock, AttentionBlock, Resample, ...)
        │       │   └── dit_layers.py   # Shared DiT primitives (attend(), WanHead)
        │       ├── wan2_1/
        │       │   ├── __init__.py
        │       │   ├── dit.py          # Wan2.1 DiT Backbone (Flax Linen), t2v + i2v
        │       │   ├── vae.py          # Wan2.1 3D Causal Video VAE, encoder + decoder
        │       │   └── clip_vision.py  # CLIP ViT-H/14 vision tower, for i2v image conditioning
        │       └── wan2_2/
        │           ├── __init__.py
        │           ├── dit.py          # Wan2.2 DiT Backbone (Flax Linen), per-token timestep
        │           └── vae.py          # Wan2.2 3D Causal Video VAE (AvgDown3D/DupUp3D/patchify)
        ├── schedulers/        # Diffusion & Flow Matching Samplers
        │   ├── __init__.py
        │   └── flow_match.py  # Euler / Rectified Flow Sampler
        └── translator/        # PyTorch -> JAX Translation Bridge
            ├── __init__.py
            ├── converter.py   # Tensor layout conversion (Transposition logic)
            └── mappings/      # Model-specific state-dict key mappings
                ├── __init__.py  # load_torch_checkpoint_to_jax dispatch table
                ├── common.py    # Mappers shared by every Wan version (DiT, T5, VAE primitives)
                ├── wan2_1.py    # Wan2.1-specific VAE/CLIP key mappings
                └── wan2_2.py    # Wan2.2-specific VAE key mappings
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
- **Don't `jax.jit` a Python loop over many repeated forward passes** (diffusion sampling steps, VAE decode's per-latent-frame chunks). `jax.jit` traces the loop by fully unrolling it into *one* HLO program, so every iteration's intermediate buffers can end up needing to coexist in that program's memory footprint instead of being freed between iterations -- this is what caused VAE decode (which internally decodes ~20 latent-frame chunks) to OOM even though DiT sampling had already succeeded. Instead, jit only the per-iteration function (its shape/dtype signature is identical every call, so this never recompiles) and call it from a plain Python loop: `examples/generate_wan2_1_t2v.py`'s sampling loop does exactly this (with `donate_argnums` on the latents carry, so each step reuses the previous step's buffer in place); `WanVAEDecoder.decode_chunk` (both Wan versions) does the same for VAE decode -- calling `WanVAEDecoder.apply()` directly on a whole video instead (i.e. `__call__`'s own internal Python loop, kept only as a simple fully-eager convenience path) means every individual op inside the decoder triggers its own separate XLA compilation; tolerable at small scale, but this is exactly what made Wan2.2's much wider decoder appear to hang for 45+ minutes at its full resolution before being jit-per-chunk instead. See `WanVAEDecoder.decode_chunk`'s docstring (in either `wan2_1/vae.py` or `wan2_2/vae.py`) for the full reasoning. `WanVAEEncoder.encode_chunk` mirrors this on the encode side (used by I2V's conditioning-image path, which actually encodes a full, mostly-zero, num_frames-long video -- not just the one real frame -- so it hits the same ~20-chunk loop).


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
python examples/generate_wan2_1_t2v.py \
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
`generate_wan2_1_t2v.py` for why.

### 4. Tensor Parallelism, Batching, and dtype

```bash
python examples/generate_wan2_1_t2v.py \
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
- **`--sequence_parallel`** (off by default): shards the DiT's token
  sequence itself across `--tensor_parallel_size` devices (DeepSpeed-Ulysses)
  instead of Megatron tensor parallelism. Megatron TP already fits the 1.3B
  model fine at typical resolutions (the path this section's example
  already uses), so there's no reason to reach for this yet — it's there
  for the 14B models, which are expected to need it the way Wan2.2's 5B
  model did in practice (see `vidax.models.wan.wan2_2.dit`'s module
  docstring for the full mechanism: Megatron TP shards attention heads/FFN
  channels but keeps the *full* token sequence on every device, so it
  doesn't help when per-token activation memory, not weight memory, is the
  actual bottleneck). Verified numerically identical to the non-SP path on
  real TPU hardware (both t2v and i2v, including the CLIP image
  cross-attention branch) with synthetic-dimension models; not yet run
  against a real 14B checkpoint, since none has been downloaded here yet.

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
- Conditioning is built from the image two ways: `vidax.models.wan.wan2_1.vae.WanVAEEncoder`
  encodes it (as a single real frame followed by zero frames, matching the
  reference's `y` construction) into the DiT's mask+latent conditioning
  channels, and `vidax.models.wan.wan2_1.clip_vision.ClipVisionTransformer` extracts
  CLIP features the DiT cross-attends onto through a second, image-only K/V
  projection (`WanDiT`'s `model_type="i2v"` path). Since the 14B DiT itself
  is much larger than the 1.3B t2v model, `--tensor_parallel_size 8` (full
  8-way, on a v4-8) is a more typical starting point here than the 4 used
  for 1.3B t2v above.
- `--sequence_parallel` (§4) is exactly the flag to reach for once you're
  running this 14B model for real: at higher resolutions its self-attention
  activation memory is the more likely bottleneck than at 1.3B t2v scale,
  and `WanDiT`'s i2v-specific CLIP image cross-attention branch is verified
  to work correctly under it too (its `attend()` call needed a small fix to
  route through the same local, non-mesh-dispatched attention path as text
  cross-attention once inside `shard_map` -- see `attend`'s docstring).
- This hasn't been verified against the real I2V-14B/CLIP checkpoints yet
  (they're much larger downloads than what t2v needed) — the architecture,
  converter mappings, and full pipeline wiring are all verified against
  synthetic PyTorch-shaped weights (exact 1:1 parameter-tree matches, same
  as every other component in this repo), but not against real weights the
  way the t2v path has been. Treat it as needing the same verification pass
  t2v already went through.

### 6. Wan2.2 TI2V-5B (Text-to-Video)

Wan2.2's TI2V-5B is a single checkpoint that supports both text-to-video
*and* image-conditioned generation, but the two use the model differently:
image conditioning works by substituting the known conditioning frame's
latent back into `x` between sampling steps (driven by a per-token
timestep of 0 there), not by any extra model input the way Wan2.1's i2v
does. Only the text-to-video path is wired up in
`generate_wan2_2_ti2v.py` so far — see `vidax.models.wan.wan2_2.dit`'s
module docstring for the full explanation and what implementing image
conditioning here would additionally need.

```bash
python examples/generate_wan2_2_ti2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v.mp4"
```

A few things unique to Wan2.2:

- **`--dit_checkpoint_path`** points at the `.safetensors.index.json`
  manifest, not a single `.safetensors` file: the 5B (and 14B) DiT ships
  sharded across multiple files. `load_torch_checkpoint_to_jax` resolves
  and merges every shard the manifest references automatically; passing a
  single non-sharded `.safetensors` still works too, same as Wan2.1.
- **`--vae_checkpoint_path`** takes `Wan2.2_VAE.pth`, a different file (and
  architecture) from Wan2.1's `Wan2.1_VAE.pth` — `WanVAEDecoder`'s 48-channel
  latent space, 2x2 pixel-patchify wrapping, and 16x spatial / 4x temporal
  compression (`vae_stride = (4, 16, 16)`) are all specific to it, see
  `vidax.models.wan.wan2_2.vae`'s module docstring.
- `--t5_checkpoint_path` uses the exact same file format (and even the same
  filename, `models_t5_umt5-xxl-enc-bf16.pth`) as Wan2.1 — the text encoder
  is byte-identical across versions, see `map_wan_t5_keys`.
- Defaults differ to match the reference's own `wan_ti2v_5B` config: 50
  sampling steps, shift 5.0, guide_scale 5.0, 121 frames, 24 fps (vs.
  Wan2.1's 81 frames / 16 fps).
- **`--tensor_parallel_size` means something different here than in
  `generate_wan2_1_t2v.py`.** At TI2V-5B's only supported resolution
  (704x1280, 121 frames), the patch-token sequence is ~27k long, and
  Wan2.2's per-token AdaLN modulation tensors scale with that directly —
  Megatron-style tensor parallelism (sharding attention heads/FFN channels,
  what Wan2.1's script uses) keeps the *full* sequence on every device and
  doesn't shrink those, so it doesn't fit a 4-chip v4 slice's HBM even after
  quartering weight memory. `WanDiT(sequence_parallel=True)` instead shards
  the *token sequence itself* between blocks (DeepSpeed-Ulysses, matching
  the reference's own `wan/distributed/sequence_parallel.py` +
  `ulysses.py`) — see `vidax.models.wan.wan2_2.dit`'s module docstring for
  the full mechanism. `--tensor_parallel_size` sets the sequence-parallel
  size for the DiT and the ordinary Megatron tensor-parallel size for T5
  (whose sequence length, 512, was never the bottleneck) simultaneously,
  and must divide both num_heads (24 for the DiT, 64 for T5) and the DiT's
  patch token count (true by construction at the default resolution for
  1/2/4/5/8-way splits).
  DiT weights are correspondingly left **replicated** rather than
  Megatron-sharded (sequence parallelism shards activations, not weights),
  and the checkpoint is cast to the target dtype on the host *before*
  `device_put` rather than after — the DiT ships as raw float32 (5B params,
  20GB replicated per device), so casting after `device_put` would need
  both the float32 and bf16 copies to coexist on-device for the cast, which
  was the difference between OOMing and not.
- **VAE decode uses `WanVAEDecoder.decode_chunk`, not a plain
  `vae_model.apply(vae_params, latents)` call.** Calling the decoder
  eagerly, one latent frame at a time (as `generate_wan2_1_t2v.py` does),
  means every individual op inside Wan2.2's much deeper, 1024-channel-wide
  decoder triggers its own separate XLA compilation -- at TI2V-5B's full
  resolution that made a first attempt appear to hang (confirmed via
  `py-spy`: it was genuinely still compiling individual ops after 45+
  minutes). `decode_chunk` (wrapped in `jax.jit`, called from a plain
  Python loop over latent frames) instead compiles the *whole* per-frame
  computation as one fused program, at most 3 times total (the cache
  state's pytree structure stabilizes after the first two frames) rather
  than once per op per frame -- see `WanVAEDecoder.decode_chunk`'s
  docstring.
- Verified end-to-end on the real TI2V-5B checkpoint on a v4-8 (4 chips),
  `--tensor_parallel_size 4`, at the default 704x1280x121-frame resolution
  (3 sampling steps, to keep the smoke test's compile time reasonable) —
  unlike Wan2.1's i2v path, this one *has* been run against real weights,
  not just synthetic-shaped ones. The output at 3 steps is (as expected for
  a diffusion model that far from convergence) a coherent but heavily
  under-denoised blur, not a recognizable scene -- that confirms the
  pipeline runs correctly end-to-end, not final output quality, which needs
  the full 50 steps to judge.