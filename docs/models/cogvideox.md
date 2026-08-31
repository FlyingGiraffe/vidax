# CogVideoX — Usage Guide

One standalone TPU inference script lives in `examples/`:
`generate_cogvideox.py` — a single script covering every released CogVideoX
checkpoint (THUDM / ZhipuAI), both T2V and I2V (pass `--image_path` for the
latter, omit it for T2V), selected with `--variant`:

| `--variant` | HF repo | Task | in-ch | Notable |
| --- | --- | --- | --- | --- |
| `2b` | `THUDM/CogVideoX-2b` | T2V | 16 | no RoPE — fixed 3D sincos positional embedding; `snr_shift_scale=3`; `scaling_factor=1.15258426` |
| `5b` | `THUDM/CogVideoX-5b` | T2V | 16 | 3D RoPE, v-prediction (the canonical model) |
| `5b-i2v` | `THUDM/CogVideoX-5b-I2V` | I2V | 32 | image latent concatenated on channels; **learned** positional-embedding buffer (locks resolution to 720×480) |
| `1.5-5b` | `THUDM/CogVideoX1.5-5B` | T2V | 16 | `patch_size_t=2` (temporal patchify), `"slice"`-grid RoPE, 81/161 frames, 1360×768 |
| `1.5-5b-i2v` | `THUDM/CogVideoX1.5-5B-I2V` | I2V | 32 | + `ofs` (offset) embedding added to the timestep embedding |

Every variant shares one DiT class (`vidax.models.cogvideo.dit.CogVideoXDiT`,
config-driven from `vidax.models.cogvideo.configs`), one VAE
(`vidax.models.cogvideo.vae.CogVideoXVAE` — a causal 3D-conv VAE, spatial ÷8,
temporal ÷4, `latent_channels=16`), the T5-v1.1-XXL text encoder (reused
verbatim from `vidax.models.ltx_video.t5`), and the
`vidax.schedulers.cogvideox` schedulers (`CogVideoXDDIMScheduler` /
`CogVideoXDPMScheduler`, v-prediction + zero-terminal-SNR + SD3-style SNR
shift + `trailing` spacing) — see this doc's [Architecture
notes](#architecture-notes) for what's specific to CogVideoX.

Requires the `torch`, `text`, and `i2v` extras:

```bash
pip install -e ".[tpu,torch,text,i2v]"
```

---

## CogVideoX (2b / 5b / 5b-i2v / 1.5-5b / 1.5-5b-i2v) — `generate_cogvideox.py`

Download a diffusers-format repo (`huggingface-cli download THUDM/CogVideoX-5b
--local-dir ./checkpoints/CogVideoX-5b`); the script expects the standard
`transformer/`, `vae/`, `text_encoder/`, `tokenizer/` subdirectories under
`--model_dir`. The `t5-v1.1-xxl` weights + tokenizer are **byte-identical
across every CogVideoX repo** (only the stored dtype differs, and bf16→f32 is
exact), so `--t5_dir` / `--tokenizer_dir` can point at any one downloaded
copy — handy when disk is tight (download the other repos with
`--exclude "text_encoder/*" "tokenizer/*"`).

### Text-to-video

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX-5b --variant 5b \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_inference_steps 50 --guidance_scale 6.0 --scheduler dpm \
  --output_path out/cogvideox_5b_t2v.mp4
```

`--scheduler dpm` + `--use_dynamic_cfg` (default) matches the reference
`cli_demo.py` recipe for the 5B models; `--scheduler ddim --no_dynamic_cfg`
matches the recommended recipe for CogVideoX-2b.

### Image-to-video

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX-5b-I2V --variant 5b-i2v \
  --t5_dir ./checkpoints/CogVideoX-5b/text_encoder \
  --tokenizer_dir ./checkpoints/CogVideoX-5b/tokenizer \
  --image_path examples/assets/cat.jpg \
  --prompt "the cat looks around, gentle camera push-in" \
  --output_path out/cogvideox_5b_i2v.mp4
```

CogVideoX-5b-I2V and CogVideoX1.5-5B-I2V use different default resolutions
(720×480 vs 1360×768) and frame counts (49 vs 81) — the script picks the
right default per `--variant`; override with `--height/--width/--num_frames`
(not for `5b-i2v`, whose learned positional embedding locks it to 720×480).
Add `--sequence_parallel_size 4` for `1.5-5b-i2v` at its native 1360×768
(see [Tensor and sequence parallelism](#tensor-and-sequence-parallelism));
without it the 1.5 I2V script falls back to a ~720×480 pixel budget.

`5b-i2v` is locked to 720×480, so its conditioning frame is resized to that
box and — by default (`--match_image_aspect`, on) — the *output video* is
rescaled back to the conditioning image's aspect ratio afterwards (preserving
the generated pixel budget, snapped to /16 for codec compatibility). Pass
`--no_match_image_aspect` to keep the raw generation resolution, or
`--height/--width` to generate at a specific size.

### Tensor and sequence parallelism

```bash
python examples/generate_cogvideox.py \
  --model_dir ./checkpoints/CogVideoX1.5-5B --variant 1.5-5b \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --sequence_parallel_size 4 \
  --output_path out/cogvideox_1_5_5b_t2v.mp4
```

`--tensor_parallel_size` (default: every device) Megatron-shards the DiT's
and T5's attention heads via `vidax.core.sharding.shard_wan_params` (its
name-pattern dispatch already covers `CogVideoXAttention`'s
`to_q`/`to_k`/`to_v`/`to_out_0`, shared with LTX); the FFN stays replicated,
which is fine since the 5B checkpoint's bf16 weights fit replicated on a
single TPU v4 chip. It is capped to a divisor of the head count (30 for 2b,
48 for the 5B models), so 2b runs `tp=2`/`dp=2` on a v4-8.

`--sequence_parallel_size` (default `1`) shards the DiT's **visual token
sequence** itself across devices (DeepSpeed-Ulysses), leaving the 226 text
tokens replicated. This is what runs CogVideoX-1.5 at its native 1360×768
(~45k visual tokens after `patch_size_t=2`, whose per-block activations don't
fit a v4 chip otherwise) — e.g. `--sequence_parallel_size 4` on a v4-8. It
must divide the device count, the head count, and the visual token count.

For CogVideoX the two are **mutually exclusive** (`--tensor_parallel_size`
must be `1` or unset whenever `--sequence_parallel_size > 1`) — the 5B DiT
fits replicated per chip, so the port doesn't thread column/row-parallel
weight-sharding through the sequence-parallel path. See
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)
and [`docs/lessons/cogvideox_debugging.md`](../lessons/cogvideox_debugging.md).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_dir` | *required* | Downloaded diffusers CogVideoX repo (with `transformer/ vae/ text_encoder/ tokenizer/`). |
| `--variant` | `5b` | One of `2b`, `5b`, `5b-i2v`, `1.5-5b`, `1.5-5b-i2v`. |
| `--t5_dir` / `--tokenizer_dir` | `<model_dir>/{text_encoder,tokenizer}` | Override — the `t5-v1.1-xxl` weights are byte-identical across every CogVideoX repo. |
| `--prompt` | *required*, 1+ values | One prompt (broadcast) or exactly `batch_size` prompts. |
| `--negative_prompt` | `""` | CFG negative prompt (diffusers' own default). |
| `--image_path` | `None` | Conditioning image, for the `*-i2v` variants. Omit for T2V. |
| `--match_image_aspect` / `--no_match_image_aspect` | on | I2V: rescale the output video to the conditioning image's aspect ratio (see [Image-to-video](#image-to-video)). |
| `--num_frames` | `49` (1.0) / `81` (1.5) | Output frame count. |
| `--height` / `--width` | per-variant reference (720×480 / 1360×768) | Generation resolution. Must be divisible by 16 (VAE ÷8 × `patch_size=2`). |
| `--num_inference_steps` | `50` | Sampling steps. |
| `--guidance_scale` | `6.0` | CFG scale. |
| `--scheduler` | `dpm` | `ddim` \| `dpm`. |
| `--use_dynamic_cfg` / `--no_dynamic_cfg` | on | diffusers' cosine CFG-scale schedule. |
| `--tensor_parallel_size` | every device | See [Tensor and sequence parallelism](#tensor-and-sequence-parallelism). Capped to a divisor of the head count (30 / 48). Must be `1` alongside `--sequence_parallel_size > 1`. |
| `--sequence_parallel_size` | `1` | See [Tensor and sequence parallelism](#tensor-and-sequence-parallelism). Must divide `num_devices`, the head count, and the visual token count. |
| `--dtype` / `--dit_dtype` | `bfloat16` | Activation / DiT-weight dtype. The T5 encode always runs in float32 regardless (see [Architecture notes](#architecture-notes)). |
| `--seed` | `42` | Initial noise seed. |
| `--fps` | `16` | Output video frame rate. |
| `--output_path` | `output_cogvideox.mp4` | With multiple prompts, each video is saved as `<output_path>_<i>.mp4`. |

### Scope

**Not implemented in this port:**

- **V2V** (video-to-video) — the diffusers pipeline supports it; this script
  does not.
- **Combined `--tensor_parallel_size` + `--sequence_parallel_size`** — the
  two are mutually exclusive for CogVideoX (see [Tensor and sequence
  parallelism](#tensor-and-sequence-parallelism)). The 5B DiT never needs
  both at once on this hardware.

---

## Architecture notes

- **DiT (`vidax.models.cogvideo.dit.CogVideoXDiT`):** a structural port of
  `CogVideoXTransformer3DModel`, config-driven from
  `vidax.models.cogvideo.configs` presets. Each `CogVideoXBlock` runs **one
  joint self-attention** over the concatenated `[text(226); visual]` sequence
  — there is no cross-attention. `CogVideoXLayerNormZero` modulates the text
  and visual halves with separate shift/scale/gate triples sharing one
  `LayerNorm`; the FFN likewise runs once over the re-concatenated sequence.
  AdaLN modulation is **per-sample, not per-token**. RoPE is per-head over
  `attention_head_dim=64` and applied to the **visual slice of q/k only** —
  text tokens are never rotated. 2b has no RoPE (a fixed 3D sincos positional
  embedding is added instead); 5b-I2V adds a learned `pos_embedding` buffer
  that locks it to 720×480; 1.5 uses `patch_size_t=2` temporal patchifying
  with a linear (not conv) patch projection.
- **Text encoder:** CogVideoX conditions on `t5-v1.1-xxl` and — unlike LTX —
  passes the encoder **no attention mask**, so the full 226-token padded
  sequence is attended (and the DiT then attends over the padded text
  embeddings too). T5-XXL's intermediate activations reach ~1e5, so without
  the mask a bf16 encode over that sequence loses 16–37% relative accuracy;
  the encode therefore runs in **float32** regardless of `--dtype`, with only
  the output embeddings cast to bf16 for the DiT and the ~19 GB fp32 T5
  params freed right after the one-time prompt encode. See
  [`docs/lessons/cogvideox_debugging.md`](../lessons/cogvideox_debugging.md).
- **VAE (`vidax.models.cogvideo.vae.CogVideoXVAE`):** a causal 3D-conv VAE
  (channels-last). `encode`/`decode` walk the clip in fixed temporal chunks
  (8 sample / 2 latent frames) carrying a causal-conv cache between chunks,
  exactly as diffusers' `_encode`/`_decode` — chunked and whole-clip results
  genuinely differ at chunk boundaries because the temporal
  `avg_pool`/`interpolate` in the (up/down)sample layers isn't cache-aware,
  so matching diffusers means matching its chunking. Decode also runs in
  overlapping spatial tiles (`_tiled_decode`, matching diffusers'
  `tiled_decode`) — the un-tiled 512-channel 3D-conv feature maps OOM a v4
  chip at the reference resolution. Toggle with
  `CogVideoXVAE(enable_tiling=False)`.
- **Schedulers (`vidax.schedulers.cogvideox`):** `CogVideoXDDIMScheduler` and
  `CogVideoXDPMScheduler` (DPM-Solver++ multistep), both v-prediction with
  `beta_schedule="scaled_linear"`, zero-terminal-SNR, an SD3-style
  `snr_shift_scale` rescale of `alphas_cumprod` (3 for 2b, 1 otherwise), and
  `timestep_spacing="trailing"`. The `use_dynamic_cfg` cosine guidance-scale
  schedule lives in the example script, matching where the diffusers pipeline
  puts it.
- **I2V conditioning** follows `CogVideoXImageToVideoPipeline.prepare_latents`:
  encode the single conditioning frame, scale it (`* scaling_factor`, or
  `/ scaling_factor` for the `invert_scale_latents` 1.5 checkpoints), zero-pad
  to `latent_frames` → `[image_latent, zeros × (N-1)]`, and concatenate it
  onto the channel axis (→ 32) every denoising step. The example uses the VAE
  posterior's `.mode()` (deterministic) where diffusers defaults to
  `.sample()`.
- **Sequence parallelism (`--sequence_parallel_size`):** CogVideoX's single
  joint `[text; visual]` attention means DeepSpeed-Ulysses can't be applied
  as-is — `vidax.core.attention.sequence_parallel_joint_self_attention` sends
  only the visual q/k/v through the head↔sequence all-to-all and slices the
  replicated text q/k/v to the local head range before one local attention
  call. Per-sample AdaLN means only the visual sequence and the RoPE tables
  need chunking. See
  [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses).
- **Checkpoint translator
  (`vidax.translator.mappings.cogvideox.{map_cogvideox_dit_keys,
  map_cogvideox_vae_keys}`):** near-mechanical prefix-strips from the
  diffusers module names (`cogvideox_dit` / `cogvideox_vae` `model_type`s);
  the T5 encoder reuses `ltx_video_t5` unchanged.
