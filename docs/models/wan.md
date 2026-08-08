# Wan (2.1 / 2.2) — Usage Guide

Three standalone TPU inference scripts live in `examples/`, one per model
variant. They share the same building blocks (`vidax.core`, `vidax.schedulers`,
`vidax.translator`) but differ in checkpoint format, resolution defaults, and
parallelism strategy — see [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
for the engineering reasoning behind those differences (Megatron vs. sequence
parallelism, flash attention, JIT-safety, the dtype-casting/decode-speed bugs
found getting Wan2.2 working).

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_wan2_1_t2v.py --model_size 1.3B` | Wan2.1 | 1.3B | Text-to-Video | `Wan2.1-T2V-1.3B` |
| `generate_wan2_1_t2v.py --model_size 14B` | Wan2.1 | 14B | Text-to-Video | `Wan2.1-T2V-14B` |
| `generate_wan2_1_i2v.py` | Wan2.1 | 14B | Image-to-Video | `Wan2.1-I2V-14B-480P`/`720P` |
| `generate_wan2_2_ti2v.py` | Wan2.2 | 5B | Text-to-Video **and** Image-to-Video | `Wan2.2-TI2V-5B` |

Both Wan2.1 T2V sizes share the same architecture and script
(`vidax.models.wan.wan2_1.dit.WanDiT`, fully config-driven); `--model_size`
just selects which hyperparameter preset
(`vidax.models.wan.wan2_1.configs.T2V_1_3B_CONFIG`/`T2V_14B_CONFIG`) to
build it with. I2V only ships as 14B (no 1.3B I2V checkpoint exists), so its
script has no `--model_size` flag — it always builds
`vidax.models.wan.wan2_1.configs.I2V_14B_CONFIG`.

All three require the `torch` extra (to deserialize `.pth`/`.safetensors`
checkpoints) and the `text` extra (tokenization):

```bash
pip install -e ".[tpu,torch,text]"
```

`--tokenizer_path` defaults to `<t5_checkpoint_dir>/google/umt5-xxl` for every
script, matching the official HuggingFace repo layouts; pass it explicitly if
yours differs.

---

## Wan2.1 T2V (1.3B / 14B) — `generate_wan2_1_t2v.py`

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
`.safetensors.index.json` manifest — pass that manifest's path, same as
Wan2.2's DiT:

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
| `--sequence_parallel` | off | Shard the DiT's token sequence itself (DeepSpeed-Ulysses) instead of Megatron TP. Not needed at 1.3B scale; may help the 14B model at higher resolutions — see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses). |
| `--dtype` | `bfloat16` | `float32` \| `float16` \| `bfloat16`. Matches the reference's `bfloat16` T5/DiT, `float32` VAE, unified here for simplicity. `float16` will fail at runtime — TPU's XLA backend doesn't implement `float16` matmuls. |
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
1.3B and 14B, output confirmed coherent (see the [parity matrix](../../README.md#model-support--parity-matrix)).

---

## Wan2.1 I2V (14B) — `generate_wan2_1_i2v.py`

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
| `--dit_checkpoint_path` | *required* | I2V-14B DiT `.safetensors` checkpoint. |
| `--vae_checkpoint_path` | *required* | VAE `.pth` checkpoint. |
| `--t5_checkpoint_path` | *required* | T5 `.pth` checkpoint. |
| `--clip_checkpoint_path` | *required* | CLIP vision checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | Tokenizer directory. |
| `--image_path` | *required* | Conditioning image. Output resolution is derived from it, not set directly. |
| `--prompt` | *required* | Text prompt (single string, not a list — unlike the t2v script). |
| `--negative_prompt` | reference's i2v `sample_neg_prompt` | Negative prompt for CFG. |
| `--tensor_parallel_size` | `1` | Must divide `num_devices` and `num_heads` (40 for the 14B DiT, 64 for T5). Since the 14B DiT is much larger than the 1.3B t2v model, `--tensor_parallel_size 8` (full 8-way on a v4-8) is a more typical starting point than the 4 used for 1.3B t2v. |
| `--sequence_parallel` | off | Same DeepSpeed-Ulysses flag as the t2v script — the one to reach for once actually running this 14B model at higher resolution, where self-attention activation memory is the more likely bottleneck than at 1.3B scale. Verified to work correctly with the CLIP image cross-attention branch too (see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)). |
| `--dtype` | `bfloat16` | Same choices/caveats as t2v. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `40` | Reference i2v default (vs. 50 for t2v). |
| `--shift` | `5.0` | Reference recommends `3.0` for 480p output, `5.0` otherwise. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--max_area` | `720*1280` | Bounds output pixel count; `compute_latent_grid` picks the largest (height, width) at that budget preserving the input image's aspect ratio, aligned to the VAE's spatial stride and the DiT's patch size — matches `WanI2V.generate`'s own resolution selection exactly. |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With `dp_size > 1`, each replica's sample is saved as `<output_path>_<i>.mp4`. |

**Status:** verified end-to-end against the real I2V-14B/VAE/T5/CLIP
checkpoints, output confirmed coherent. Fixed two bugs found during that
first real run, both pre-existing and specific to the image-conditioning
path (never exercised against real weights before): `build_i2v_conditioning`
called a `pre_process` method that doesn't exist on Wan2.1's `WanVAEEncoder`
(that's a Wan2.2-only pixel-unshuffle step — Wan2.1's `encode_chunk` takes
raw RGB pixels directly); and `Encoder3d`/`WanVAEEncoder`'s
`temperal_downsample` default was `(True, True, False)` (the class's own
default) instead of the released checkpoint's actual `(False, True, True)`
(confirmed against the reference's `_video_vae` config and the checkpoint's
own weight keys) — the decoder-side `temperal_upsample` default happened to
already be correct, which is why this went undetected until the encoder
path was actually exercised.

---

## Wan2.2 TI2V (5B) — `generate_wan2_2_ti2v.py`

Wan2.2's TI2V-5B is a single checkpoint that supports **both** text-to-video
and image-conditioned generation in the same script: pass `--image_path` for
i2v, omit it for t2v. Architecturally the two use the model quite
differently — image conditioning works by substituting the known
conditioning frame's latent back into `x` between sampling steps (driven by
a per-token timestep of 0 for that frame's tokens, re-applied after every
step), not by any extra model input the way Wan2.1's i2v does. See
`vidax.models.wan.wan2_2.dit`'s module docstring for the architecture side,
and the reference's `WanTI2V.i2v` (`masks_like`'s frame-0 mask) for the
sampling-loop mechanics this mirrors.

### Text-to-video

```bash
python examples/generate_wan2_2_ti2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v.mp4"
```

### Image-to-video

```bash
python examples/generate_wan2_2_ti2v.py \
  --dit_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_path "./checkpoints/Wan2.2-TI2V-5B/examples/i2v_input.JPG" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v_i2v.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | Points at the `.safetensors.index.json` manifest, not a single `.safetensors` file: the 5B (and 14B) DiT ships sharded across multiple files. `load_torch_checkpoint_to_jax` resolves and merges every shard automatically; a single non-sharded `.safetensors` still works too. |
| `--vae_checkpoint_path` | *required* | `Wan2.2_VAE.pth` — a different file *and architecture* from Wan2.1's `Wan2.1_VAE.pth` (48-channel latent space, 2x2 pixel-patchify wrapping, 16x spatial / 4x temporal compression). See `vidax.models.wan.wan2_2.vae`'s module docstring. |
| `--t5_checkpoint_path` | *required* | Same file format (and even filename) as Wan2.1 — the text encoder is byte-identical across versions (`map_wan_t5_keys`). |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | Tokenizer directory. |
| `--image_path` | `None` | Conditioning image; omit for t2v. When given, output resolution is derived from the image's aspect ratio + `--max_area` instead of `--height`/`--width` (which are then ignored) — matches the reference's `WanTI2V.i2v`, which has no fixed `size` the way `.t2v` does. |
| `--max_area` | `704*1280` | i2v only: target output pixel area; actual (height, width) derived via `best_output_size` (ported directly from the reference). |
| `--prompt` | *required*, 1+ values | Same broadcast semantics as Wan2.1 t2v. |
| `--negative_prompt` | reference's `sample_neg_prompt` | Negative prompt for CFG. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--tensor_parallel_size` | `1` | **Means something different here than in the Wan2.1 scripts** — see below. |
| `--dtype` | `bfloat16` | Same choices/caveats as Wan2.1. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | reference per-mode default | `None` resolves to the reference's own default per mode: 50 for t2v, 40 for i2v (`WanTI2V.generate`/`.i2v`). |
| `--shift` | `5.0` | Reference default for TI2V-5B. |
| `--height` | `704` | Ignored if `--image_path` is given. Must be a multiple of 16 (VAE spatial stride). |
| `--width` | `1280` | Ignored if `--image_path` is given. Must be a multiple of 16. |
| `--num_frames` | `121` | Reference default for TI2V-5B (vs. 81 for Wan2.1). |
| `--fps` | `24` | Reference `sample_fps` for TI2V-5B (vs. 16 for Wan2.1). |
| `--output_path` | `output_video.mp4` | With multiple prompts, each saved as `<output_path>_<i>.mp4`. |

**`--tensor_parallel_size` note:** at TI2V-5B's only supported resolution
(704x1280, 121 frames), the patch-token sequence is ~27k long, and Wan2.2's
per-token AdaLN modulation tensors scale with that directly — Megatron-style
tensor parallelism (what the Wan2.1 scripts use) keeps the *full* sequence on
every device and doesn't shrink those, so it doesn't fit a 4-chip v4 slice's
HBM even after quartering weight memory. This script therefore **always**
uses `WanDiT(sequence_parallel=True)` internally when `--tensor_parallel_size
> 1`, sharding the token sequence itself between blocks instead (DeepSpeed-
Ulysses — see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)).
`--tensor_parallel_size` sets both the DiT's sequence-parallel size and T5's
ordinary Megatron tensor-parallel size (T5's sequence length, 512, was never
the bottleneck) — it must divide both `num_heads` (24 for the DiT, 64 for T5)
and the DiT's patch token count. This is true by construction at the default
704x1280x121 resolution for 1/2/4/5/8-way splits; for i2v's image-derived
resolution it isn't guaranteed, so the script grows the derived width in
32px steps (up to `tensor_parallel_size - 1` times — guaranteed to find a
divisible value) and logs when it does, rather than failing outright.

**Status:** verified end-to-end on the real TI2V-5B checkpoint for **both**
t2v and i2v, on a v4-8 (4 chips), `--tensor_parallel_size 4` (3 sampling
steps, to keep smoke-test compile time reasonable — output at 3 steps is a
coherent but heavily under-denoised blur, as expected that far from
convergence; that confirms the pipeline runs correctly end-to-end, not final
output quality, which needs the full step count to judge). See the
[hardware doc](../hardware_and_sharding.md) for the bugs found and fixed
getting this far (dtype-casting OOM, VAE decode compile time, i2v
resolution-divisibility, host-transfer for decoded chunks).

---

## Coming later

- **Wan2.2 A14B (14B MoE, T2V/I2V)** — not yet implemented; will need its
  own DiT variant support (two-expert high/low-noise split) once checkpoints
  are available.

See [`docs/models/cosmos.md`](cosmos.md) and [`docs/models/cosmos3.md`](cosmos3.md)
for the Cosmos-Predict2.5 and Cosmos 3 model families.

See the [parity matrix in the root README](../../README.md#model-support--parity-matrix)
for the up-to-date status across all variants.
