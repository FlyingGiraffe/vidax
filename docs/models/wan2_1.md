# Wan2.1 — Usage Guide

Two standalone TPU inference scripts live in `examples/`, one per task. They
share the same building blocks (`vidax.core`, `vidax.schedulers`,
`vidax.translator`) — see [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
for the engineering reasoning behind the parallelism strategy (Megatron
tensor parallelism, flash attention, JIT-safety).

| Script | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- |
| `generate_wan2_1_t2v.py --model_size 1.3B` | 1.3B | Text-to-Video | `Wan2.1-T2V-1.3B` |
| `generate_wan2_1_t2v.py --model_size 14B` | 14B | Text-to-Video | `Wan2.1-T2V-14B` |
| `generate_wan2_1_i2v.py` | 14B | Image-to-Video | `Wan2.1-I2V-14B-480P` or `Wan2.1-I2V-14B-720P` |

Both T2V sizes share the same architecture and script
(`vidax.models.wan.wan2_1.dit.WanDiT`, fully config-driven); `--model_size`
just selects which hyperparameter preset
(`vidax.models.wan.wan2_1.configs.T2V_1_3B_CONFIG`/`T2V_14B_CONFIG`) to
build it with. I2V only ships as 14B (no 1.3B I2V checkpoint exists), so its
script has no `--model_size` flag — it always builds
`vidax.models.wan.wan2_1.configs.I2V_14B_CONFIG`.

Both require the `torch` extra (to deserialize `.pth`/`.safetensors`
checkpoints) and the `text` extra (tokenization):

```bash
pip install -e ".[tpu,torch,text]"
```

`--tokenizer_path` defaults to `<t5_checkpoint_dir>/google/umt5-xxl` for both
scripts, matching the official HuggingFace repo layout; pass it explicitly if
yours differs.

---

## T2V (1.3B / 14B) — `generate_wan2_1_t2v.py`

Checkpoints (DiT `.safetensors`, VAE `.pth`, T5 `.pth` + its
`google/umt5-xxl` tokenizer folder) come from the official
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) or
[Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) repos —
pass `--model_size` to match whichever you downloaded.

### Basic generation (1.3B)

```bash
python examples/generate_wan2_1_t2v.py \
  --model_size 1.3B \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 50 \
  --output_path "out/output.mp4"
```

### 14B generation

The 14B DiT ships sharded across multiple `.safetensors` files with a
`.safetensors.index.json` manifest — pass that manifest's path:

```bash
python examples/generate_wan2_1_t2v.py \
  --model_size 14B \
  --dit_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --num_steps 50 \
  --output_path "out/output_14b.mp4"
```

`--shift` (noise-schedule shift, default 5.0) and classifier-free guidance
(`--guide_scale`, default 5.0, and `--negative_prompt`, defaulting to the
reference's quality-negative-prompt) match the reference pipeline's own
defaults and aren't optional extras — the reference always runs with both,
and skipping CFG in particular produces washed-out, low-contrast output (the
model's raw conditional prediction on its own regresses hard toward an
"average video"). See `RectifiedFlowScheduler` and `single_step` in the
script for why.

### Tensor parallelism, batching, and dtype

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

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `1.3B` | `1.3B` or `14B` — selects `T2V_1_3B_CONFIG`/`T2V_14B_CONFIG` from `vidax.models.wan.wan2_1.configs`. Must match `--dit_checkpoint_path`'s actual checkpoint. |
| `--dit_checkpoint_path` | *required* | DiT `.safetensors` checkpoint (1.3B) or `.safetensors.index.json` manifest (14B, sharded). |
| `--vae_checkpoint_path` | *required* | VAE `.pth` checkpoint. |
| `--t5_checkpoint_path` | *required* | T5 (umt5-xxl encoder) `.pth` checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | umt5-xxl HuggingFace tokenizer directory. |
| `--prompt` | *required*, 1+ values | One prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica. |
| `--negative_prompt` | reference's `sample_neg_prompt` | Negative prompt for CFG. |
| `--guide_scale` | `5.0` | CFG scale: `velocity = uncond + guide_scale * (cond - uncond)`. |
| `--tensor_parallel_size` | `1` | Devices to Megatron-shard attention heads / FFN channels across (see [hardware doc](../hardware_and_sharding.md)). Must divide `num_devices` and `num_heads` (12 for the 1.3B DiT, 40 for the 14B DiT, 64 for T5). `--tensor_parallel_size 4` (4-way TP × 2-way DP) is a reasonable start on a v4-8 at full 1280×720 for 1.3B; the 14B model was verified with `--tensor_parallel_size 4` on a v4-8. Raise it if you hit HBM OOM. |
| `--sequence_parallel_size` | `1` | Devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of `--tensor_parallel_size`'s weight-sharding — the two compose freely (see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s "Combining with Megatron TP"). Not needed at 1.3B scale; may help the 14B model at higher resolutions. |
| `--dtype` | `bfloat16` | `float32` \| `float16` \| `bfloat16`. Compute dtype for T5/VAE, and for the DiT's *activations* specifically (not its weights — see `--dit_dtype`). `float16` will fail at runtime — TPU's XLA backend doesn't implement `float16` matmuls. |
| `--dit_dtype` | `float32` | Cast target for the DiT's *weights*, independent of `--dtype`. Defaults to `float32` because the reference keeps its residual stream in float32 for virtually the entire network even under bf16 autocast, and the released checkpoints ship natively float32 — rounding them to bf16 (the old default here) causes real, visually obvious corruption at large token counts. See [Precision: fp32 DiT weights](#precision-fp32-dit-weights) below. Pass `--dit_dtype bfloat16` to opt back into the old, cheaper-but-lossy-at-scale behavior. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `50` | Sampling steps. |
| `--shift` | `5.0` | Flow-matching noise-schedule shift. Reference default for t2v regardless of resolution. |
| `--height` | `480` | Output video height. |
| `--width` | `832` | Output video width. |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With multiple prompts, each is saved as `<output_path>_<i>.mp4`. |

`--prompt` accepts one or more values. A single prompt is broadcast to every
data-parallel replica (`num_devices // tensor_parallel_size`) with
independent noise, giving that many samples of one prompt "for free"; exactly
`dp_size` prompts gives one video per prompt. The underlying
`WanDiT`/`T5Encoder` architecture supports arbitrary batch sizes (as does the
PyTorch reference's model code) — it's just that the reference pipeline's
`generate()` never uses it.

**Status:** fully verified end-to-end against real checkpoints for both
1.3B and 14B, output confirmed coherent (see the [parity matrix](../../README.md#-model-support)).

---

## I2V (14B) — `generate_wan2_1_i2v.py`

I2V only ships as a **14B** model, and as **two separate checkpoints**
tuned for different resolution ranges — `Wan2.1-I2V-14B-480P` and
`Wan2.1-I2V-14B-720P` on [HuggingFace](https://huggingface.co/Wan-AI) (no
1.3B I2V checkpoint exists). The two checkpoints share an identical
architecture/`config.json` — only the weights (and the resolution they were
trained/tuned at) differ — so both are driven by this one script; there's no
separate `--model_size`-style flag, just point `--dit_checkpoint_path` at
whichever checkpoint and pass a matching `--max_area`/`--shift` (see below).
Both additionally need a **CLIP vision encoder** checkpoint
(`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, bundled in each
repo) to extract image features from the conditioning frame.

### 480P checkpoint

```bash
python examples/generate_wan2_1_i2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --image_path "./checkpoints/Wan2.1-I2V-14B-480P/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --max_area $((480 * 832)) \
  --output_path "out/output_i2v_480p.mp4"
```

`--max_area` (832\*480, aligned to the checkpoint's own trained resolution
range) leaves `--shift` on its default auto-selection, which resolves to
`3.0` at this scale (see `--shift`'s CLI entry below) — the reference's own
behavior for the 480P checkpoint.

### 720P checkpoint

```bash
python examples/generate_wan2_1_i2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
  --clip_checkpoint_path "./checkpoints/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --image_path "./checkpoints/Wan2.1-I2V-14B-720P/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --output_path "out/output_i2v_720p.mp4"
```

`--max_area` defaults to 720\*1280 already, so it's left unset here;
`--shift` auto-resolves to `5.0` at this scale.

Both checkpoints' DiT ships sharded across multiple `.safetensors` files
with a `.safetensors.index.json` manifest, same as the T2V-14B checkpoint —
pass that manifest's path, not a single `.safetensors` file.

Conditioning is built from the image two ways:
`vidax.models.wan.wan2_1.vae.WanVAEEncoder` encodes it (as a single real
frame followed by zero frames, matching the reference's `y` construction)
into the DiT's mask+latent conditioning channels, and
`vidax.models.wan.wan2_1.clip_vision.ClipVisionTransformer` extracts CLIP
features the DiT cross-attends onto through a second, image-only K/V
projection (`WanDiT`'s `model_type="i2v"` path).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | I2V-14B DiT checkpoint — either `Wan2.1-I2V-14B-480P` or `Wan2.1-I2V-14B-720P`'s `.safetensors.index.json` manifest (sharded, same as T2V-14B). |
| `--vae_checkpoint_path` | *required* | VAE `.pth` checkpoint. |
| `--t5_checkpoint_path` | *required* | T5 `.pth` checkpoint. |
| `--clip_checkpoint_path` | *required* | CLIP vision checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | Tokenizer directory. |
| `--image_path` | *required* | Conditioning image. Output resolution is derived from it, not set directly. |
| `--prompt` | *required* | Text prompt (single string, not a list — unlike the t2v script). |
| `--negative_prompt` | reference's i2v `sample_neg_prompt` | Negative prompt for CFG. |
| `--tensor_parallel_size` | `1` | Must divide `num_devices` and `num_heads` (40 for the 14B DiT, 64 for T5). `--tensor_parallel_size 4` (full width on this repo's v4-8, i.e. 4 chips — see [hardware doc](../hardware_and_sharding.md#2-sharding--tpu-topology-megatron-style-tensor-parallelism)) is the typical starting point for the 14B model, same as t2v. |
| `--sequence_parallel_size` | `1` | Same DeepSpeed-Ulysses flag as the t2v script, independent of `--tensor_parallel_size` — the one to reach for once actually running this 14B model at higher resolution, where self-attention activation memory is the more likely bottleneck than at 1.3B scale. Verified to work correctly with the CLIP image cross-attention branch too, as long as `--tensor_parallel_size 1` (combining both with i2v's CLIP branch isn't supported yet — see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s "Combining with Megatron TP"). |
| `--dtype` | `bfloat16` | Same choices/caveats as t2v. |
| `--dit_dtype` | `float32` | Same as t2v — cast target for the DiT's weights specifically, decoupled from `--dtype`. Also applies to the I2V conditioning tensor `y` and the sampling loop's `latents`, since both are concatenated directly into the DiT's own input. See [Precision: fp32 DiT weights](#precision-fp32-dit-weights). |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `40` | Reference i2v default (vs. 50 for t2v). |
| `--shift` | auto (`None`) | Flow-matching noise-schedule shift. When left unset, auto-selects the reference's own default: `3.0` if `--max_area <= 832*480` (480P checkpoint's scale), `5.0` otherwise (720P checkpoint's scale). Pass explicitly to override either way. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--max_area` | `720*1280` | Bounds output pixel count; `compute_latent_grid` picks the largest (height, width) at that budget preserving the input image's aspect ratio, aligned to the VAE's spatial stride and the DiT's patch size — matches `WanI2V.generate`'s own resolution selection exactly. Set to `480*832` when using the 480P checkpoint (the default matches the 720P checkpoint's scale). |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With `dp_size > 1`, each replica's sample is saved as `<output_path>_<i>.mp4`. |

**Status:** verified end-to-end against the real I2V-14B/VAE/T5/CLIP
checkpoints for **both** the 480P and 720P checkpoints, output confirmed
coherent for each at its own resolution (see
[`docs/benchmarking.md`](../benchmarking.md) for measured numbers). Fixed
two bugs found during the first real run, both pre-existing and specific to
the image-conditioning path (never exercised against real weights before):
`build_i2v_conditioning` called a `pre_process` method that doesn't exist on
Wan2.1's `WanVAEEncoder` (that's a Wan2.2-only pixel-unshuffle step —
Wan2.1's `encode_chunk` takes raw RGB pixels directly); and
`Encoder3d`/`WanVAEEncoder`'s `temperal_downsample` default was
`(True, True, False)` (the class's own default) instead of the released
checkpoint's actual `(False, True, True)` (confirmed against the reference's
`_video_vae` config and the checkpoint's own weight keys) — the decoder-side
`temperal_upsample` default happened to already be correct, which is why
this went undetected until the encoder path was actually exercised. A third
issue, specific to running the 480P checkpoint, was caught while adding
proper benchmark coverage for it: earlier code loaded it with `--max_area`
and `--shift` left at their 720P-scale defaults (`720*1280`/`5.0`), which
runs the 480P checkpoint well outside the resolution range it was actually
trained/tuned at — `--shift` now auto-selects `3.0` at 480P scale (see the
CLI reference below), and the 480P usage example explicitly passes
`--max_area 480*832`.

Both checkpoints' "verified coherent" status above predates the fp32-DiT-
weights fix described next; 480P remains verified under the new default
(smaller token count, fits comfortably), but native 720P (81 frames) now
OOMs on this repo's 4-chip machine during the conditioning image's VAE
encode, before generation even starts, since the extra weight memory the
fix requires exceeds available HBM headroom at that scale — see
[Precision: fp32 DiT weights](#precision-fp32-dit-weights) and
[`docs/benchmarking.md`](../benchmarking.md) for the measured error and
current status.

### Precision: fp32 DiT weights

Wan2.1's DiT weights default to `--dit_dtype float32`, not `bfloat16` —
different from every other model in this repo, and different from this
script's own earlier default. The reference implementation's checkpoints
ship natively as float32 on disk, and `WanAttentionBlock.forward` wraps its
residual updates in `amp.autocast(dtype=torch.float32)` without ever
casting back down — so the residual stream stays float32 for virtually the
entire 40-layer network, with bf16 autocast applied only transiently to
individual matmul/conv ops, never to the stored weights or the
accumulating residual. This repo's implementation originally treated
Wan2.1 like every other bf16-weights model (round the checkpoint to bf16
at load, keep the residual stream in bf16 throughout) — correct-looking
and fine at most scales, but it compounds into real, visually obvious
output corruption (hazy, flat, low-detail) specifically at large token
counts (native 720P, 81 frames), since that's where 40 layers' worth of
bf16 rounding error in a long chain of residual additions finally becomes
significant. Fixed by keeping DiT weights and the residual stream at
float32 throughout, matching the reference's actual dtype handling exactly
— `WanDiTBlock`/`WanDiT` now downcast to bf16 only transiently, immediately
before each sub-layer's matmul, then back to float32 immediately after, for
the residual add.

`--dit_dtype bfloat16` remains fully supported as an explicit opt-in
(useful for memory-constrained runs at smaller/safer token counts, where
the corruption doesn't show up), decoupled from the general `--dtype` flag
(still `bfloat16` by default, governing T5/VAE/CLIP, which really are
cheap and correct to run in bf16). The memory cost of the new default is
real: roughly double the DiT's resident weight memory versus bf16, which is
the direct cause of I2V-14B-720P's current OOM at native resolution (see
above). Full investigation, including why this took three compounding bugs
rather than one to actually fix (checkpoint rounding, a
`compute_dtype`/`dit_dtype` decoupling bug, and per-step latent
re-quantization), in
[`docs/lessons/wan2_1_precision_debugging.md`](../lessons/wan2_1_precision_debugging.md).

---

See [`docs/models/wan2_2.md`](wan2_2.md) for Wan2.2 (TI2V-5B, A14B),
[`docs/models/cosmos2_5.md`](cosmos2_5.md) for Cosmos-Predict2.5, and
[`docs/models/cosmos3.md`](cosmos3.md) for Cosmos 3.

See the [Model Support table in the root README](../../README.md#-model-support)
for the up-to-date status across all variants.
