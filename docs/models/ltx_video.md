# LTX-Video (0.9.8) — Usage Guide

One standalone TPU inference script lives in `examples/`:
`generate_ltx_video.py` — a single script covering both T2V and I2V (pass
`--image_path` for the latter, omit it for T2V), for all three released
0.9.8 sizes (2B-distilled, 13B-dev, 13B-distilled). All three share one
architecture (`vidax.models.ltx_video.dit.LTXDiT`, built directly from each
checkpoint's own embedded architecture config — see [Architecture
notes](#architecture-notes)) and only differ in dims. It reuses vidax's
generic building blocks (`vidax.core.sharding`, `vidax.translator`) but is
otherwise a from-scratch port, architecturally unrelated to the Wan/Cosmos
scripts: a different VAE (pixel-unshuffle patchify, `PixelNorm`, a
noise-conditioned decoder), a different text encoder (plain T5-XXL, not
UMT5 or a VLM), a different RoPE convention, and a different scheduler
(`RectifiedFlowScheduler` with a `LinearQuadratic` sigma schedule) — see
this doc's [Architecture notes](#architecture-notes) for what's specific to
LTX-Video.

| Script | Model | Params | Task | Checkpoint file example |
| --- | --- | --- | --- | --- |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 2B (distilled) | T2V, I2V | `ltxv-2b-0.9.8-distilled.safetensors` |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 13B (dev) | T2V, I2V | `ltxv-13b-0.9.8-dev.safetensors` |
| `generate_ltx_video.py` | LTX-Video 0.9.8 | 13B (distilled) | T2V, I2V | `ltxv-13b-0.9.8-distilled.safetensors` |

Requires the `torch` extra (to deserialize the `.safetensors` checkpoints),
the `text` extra (`transformers`, for the T5 tokenizer), and the `i2v`
extra (`pillow`, for I2V's conditioning image):

```bash
pip install -e ".[tpu,torch,text,i2v]"
```

---

## LTX-Video 0.9.8 (2B-distilled / 13B-dev / 13B-distilled) — `generate_ltx_video.py`

Each variant's DiT + VAE ship together as **one flat `.safetensors` file**
(`model.diffusion_model.*`/`vae.*` key prefixes, ComfyUI-style) from
[Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — pass
its path as `--checkpoint_path`. The architecture's hyperparameters
(`num_layers`, `num_attention_heads`, dims, RoPE constants, VAE block
structure, ...) aren't hardcoded per-variant in this repo at all: they're
read directly from the checkpoint's own **embedded safetensors metadata**
(`safetensors.safe_open(path).metadata()["config"]`) via
`vidax.models.ltx_video.configs.load_ltx_checkpoint_metadata` — the same
mechanism the reference itself uses, so a new released variant with the
same architecture family needs no code change here, just a new checkpoint
path. The text encoder is a separate download,
[PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)'s
`text_encoder`/`tokenizer` subfolders (a plain T5-XXL, shared by all three
DiT sizes) — pass `text_encoder/model.safetensors.index.json`'s path as
`--t5_checkpoint_path`; `--tokenizer_path` then defaults to the sibling
`tokenizer/` directory.

### Text-to-video

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-2B-distilled/ltxv-2b-0.9.8-distilled.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 8 --guidance_scale 1.0 \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx_t2v.mp4"
```

`--num_steps 8 --guidance_scale 1.0` matches the distilled checkpoints' own
recommended recipe (trained to look good in very few steps with no
classifier-free guidance at all — `guidance_scale=1.0` skips CFG's
amplification entirely, though this script still runs the unconditional
branch regardless, see [Status](#status)). For `13B-dev` (not distilled),
use real CFG and more steps instead:

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-13B-dev/ltxv-13b-0.9.8-dev.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --num_steps 30 --guidance_scale 3.0 \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx_t2v_dev.mp4"
```

### Image-to-video

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-2B-distilled/ltxv-2b-0.9.8-distilled.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --num_steps 8 --guidance_scale 1.0 \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx_i2v.mp4"
```

The conditioning image is resized to exactly `--height`x`--width` (no
aspect-ratio-preserving crop, unlike Wan/Cosmos's `--max_area`-derived
resolution), VAE-encoded, and lerp'd into the first latent frame at
`--conditioning_strength` (default `1.0` — the first frame *is* the
encoded image, no noise mixed in). See [Architecture
notes](#architecture-notes) for the full per-token conditioning-mask
mechanism this drives during sampling.

### Tensor parallelism

```bash
python examples/generate_ltx_video.py \
  --checkpoint_path "./checkpoints/LTX-Video-0.9.8-13B-distilled/ltxv-13b-0.9.8-distilled.safetensors" \
  --t5_checkpoint_path "./checkpoints/PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --num_steps 8 --guidance_scale 1.0 \
  --output_path "out/output_ltx_tp.mp4"
```

`--tensor_parallel_size` (default `1`) Megatron-shards both the DiT's
attention heads/FFN channels and the T5 encoder's, via
`vidax.core.sharding.shard_wan_params` (its name-pattern dispatch already
covers `LTXDiT`'s `to_q`/`to_k`/`to_v`/`to_out_0` — the same names Cosmos3's
attention happens to use — plus two LTX-specific additions, `ff_proj`/
`ff_out`; T5's names were already covered from Wan's UMT5 support). Needed
for the 13B checkpoints regardless of resolution (their bf16 weights alone,
~26GB, don't fit replicated on a single TPU v4 chip's HBM) — and, less
obviously, needed for **2B too** at the reference's full
704x1216/121-frame resolution: 2B's own weights fit replicated fine, but
the self-attention activations at that token count don't (confirmed OOM at
`tp=1`; see [`docs/benchmarking.md`](../benchmarking.md)'s "why TP" row).
There is no `--sequence_parallel_size` yet — see [Status](#status).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_path` | *required* | The flat `.safetensors` file bundling both the DiT and VAE. |
| `--t5_checkpoint_path` | *required* | `PixArt-XL-2-1024-MS/text_encoder/model.safetensors.index.json`. |
| `--tokenizer_path` | `<t5_checkpoint_dir>/../tokenizer` | HuggingFace tokenizer directory. |
| `--prompt` | *required*, 1+ values | One prompt (broadcast) or exactly `batch_size` prompts. |
| `--negative_prompt` | reference's own default | Negative prompt for CFG. |
| `--image_path` | `None` | Conditioning image, for I2V. Omit for T2V. |
| `--conditioning_strength` | `1.0` | I2V only: how strongly the conditioning image is enforced (see [Image-to-video](#image-to-video)). |
| `--guidance_scale` | `3.0` | CFG scale: `velocity = uncond + guidance_scale * (cond - uncond)`. Use `1.0` for the distilled checkpoints (see [Text-to-video](#text-to-video)). |
| `--dtype` | `bfloat16` | Compute dtype for the VAE, T5, and DiT activations/latents. |
| `--dit_dtype` | `bfloat16` | Cast target for the DiT's weights. Every released checkpoint ships natively as bf16 (unlike Wan2.1, no fp32-weights precision requirement here). |
| `--tensor_parallel_size` | `1` | See [Tensor parallelism](#tensor-parallelism). Must divide `num_devices`, `LTXDiT.num_attention_heads` (32 for every released variant), and the T5 encoder's `num_heads` (64). |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `30` | Sampling steps. Use far fewer (e.g. `8`) for distilled checkpoints. |
| `--sampler` | `LinearQuadratic` | `Uniform` \| `LinearQuadratic` \| `Constant`. Every released checkpoint's own embedded scheduler config uses `LinearQuadratic`. |
| `--shift` | `None` | Required only for `--sampler Constant`. |
| `--text_max_tokens` | `256` | T5 prompt padding/truncation length (the reference pipeline's own default). |
| `--decode_timestep` | `0.05` | VAE decoder noise-conditioning timestep (see [Architecture notes](#architecture-notes)'s VAE section) — the reference's own default for every released checkpoint. |
| `--decode_noise_scale` | `None` | How much fresh noise to mix into the latent before decoding. Defaults to `--decode_timestep`, matching the reference. |
| `--height` | `512` | Output video height. Must be divisible by the VAE's spatial downscale factor (32 for every released checkpoint: 8x from block structure × `patch_size=4`'s pixel-unshuffle). |
| `--width` | `768` | Output video width. Same divisibility rule as `--height`. |
| `--num_frames` | `97` | Output frame count. The reference's causal VAE wants `1 + 8k` (8x temporal downscale) for an exact round-trip; the reference's own default is `121`. |
| `--fps` | `24` | Output video frame rate. |
| `--output_path` | `output_video.mp4` | With multiple prompts, each video is saved as `<output_path>_<i>.mp4`. |

### Status

**Verified end-to-end on real weights, T2V and I2V, 2B and 13B alike:
output is coherent, prompt-matching, temporally consistent video.** Unlike
this repo's other model ports, correctness here was checked two ways, not
one:

- **Bit-exact against the actual reference PyTorch implementation**, run in
  a throwaway conda env (`torch`+`diffusers`+`transformers` pinned to the
  versions the reference was built against) against the real downloaded
  2B-distilled checkpoint, with `jax_default_matmul_precision="highest"`:
  DiT max diff `3.3e-5` (correlation `0.999999999984`), VAE encode+decode
  max diff `2e-5` (correlation `0.9999999999`), T5 text encoder max diff
  `1.2e-5` (correlation `0.9999999999`).
- **Real end-to-end generation**, both variants: T2V from the standardized
  red-panda prompt produces a recognizable red panda in snow next to
  bamboo (13B noticeably sharper/more detailed than 2B, as expected); I2V
  conditioned on a generated frame reproduces it near-exactly at frame 0
  and continues it coherently.

See [`docs/lessons/ltx_video_debugging.md`](../lessons/ltx_video_debugging.md)
for the bit-exact verification methodology used to confirm this.

**Not implemented in this first port** (see
[`examples/generate_ltx_video.py`](../../examples/generate_ltx_video.py)'s
module docstring):

- **Multi-scale two-pass generation** (the reference's default pipeline:
  a low-res first pass, a learned `LatentUpsampler`, then a high-res second
  pass continuing from there) — this port always runs single-scale,
  single-pass.
- **STG (spatio-temporal guidance)** and **`cfg_star_rescale`** — both
  pure inference-loop guidance tricks layered on a working base model, not
  architectural; skipped to keep the first port's surface area minimal.
  Plain CFG only, with one constant `--guidance_scale` for the whole run
  (the reference's own configs use a per-step schedule).
- **`--sequence_parallel_size`** — `LTXDiT` has no sequence-parallel
  wiring yet; `--tensor_parallel_size` alone is what this port has, and
  what every current benchmark row uses (see
  [`docs/benchmarking.md`](../benchmarking.md)).
- **V2V** — the reference's conditioning mechanism (VAE-encode + lerp +
  per-token mask) is the same one I2V already uses, just anchored at an
  interior/later frame instead of frame 0 — architecturally close, just
  not wired into the example script yet.

See the [parity matrix in the root README](../../README.md#-model-support)
for the up-to-date status across all variants.

---

## Architecture notes

- **DiT (`vidax.models.ltx_video.dit.LTXDiT`):** a structural port of the
  reference `Transformer3DModel`/`BasicTransformerBlock`/`Attention`, built
  directly from each checkpoint's own embedded config rather than
  hardcoded per-variant presets (`vidax.models.ltx_video.configs`'
  `DIT_2B_CONFIG`/`DIT_13B_CONFIG` document, but don't drive, the dims
  actually in use). RoPE is computed once over the model's *full*
  `inner_dim` (not per-head) from a fractional-pixel-coordinate,
  `dim // 6`-band exponential frequency schedule, and applied to
  query/key *before* the per-head reshape — a genuinely different
  convention from Wan's per-head, integer-position RoPE (see
  `vidax.models.ltx_video.rope`'s module docstring). `norm1`/`norm2`/
  `norm_out` are RMSNorm with no learnable scale for every released
  checkpoint; `q_norm`/`k_norm` *do* have one, applied over the full
  un-split `inner_dim` (mirroring Wan's own QK-RMSNorm convention).
  Feed-forward is plain tanh-approximate GELU (`activation_fn=
  "gelu-approximate"`), not GEGLU, for every released checkpoint — simpler
  than the reference's generically-activation-configurable `FeedForward`
  class suggests. `timestep` may be `(B,)` (T2V) or `(B, N)` (I2V
  per-token, see below) — both flow through the same AdaLN path via
  `.reshape(batch, -1, ...)`, exactly mirroring the reference.
- **VAE (`vidax.models.ltx_video.vae.LTXVAE`):** a causal 3D-conv VAE, but
  structurally unlike Wan/Cosmos's: `PixelNorm` (RMS-normalize over
  channels, no learnable params) instead of GroupNorm, a `patch_size=4`
  pixel-unshuffle before `conv_in` (contributing most of the 32x spatial
  downscale, on top of 8x from three `compress_*_res`/`compress_all`
  block-level downsamples), and a **noise-conditioned decoder**
  (`timestep_conditioning=True` for every released checkpoint — every
  decoder resnet block, plus a final output-layer AdaLN, is modulated by a
  `--decode_timestep`-driven embedding, so decoding isn't a pure function
  of the latent alone; see `Decoder`'s docstring). Also unlike Wan/Cosmos:
  the **decoder's convs use symmetric temporal padding**
  (`causal_decoder=False` for every released checkpoint — confusingly
  named; it governs the *decoder's* padding, unrelated to whether the
  *encoder* is causal, which it always is). This
  v1 port runs the whole encode/decode in one forward pass over the full
  tensor (matching the reference's own default, non-tiled path), unlike
  Wan's frame-chunked streaming decoder — revisit if a resolution/frame
  count needs the memory savings chunking gives.
- **Patchifier (`vidax.models.ltx_video.patchifier`):** trivial — every
  released checkpoint uses `patch_size=1` at the transformer level (all
  spatial/temporal compression happens in the VAE), so `patchify`/
  `unpatchify` are pure reshapes, sharing token order with
  `get_latent_coords`'s row-major `(f, h, w)` meshgrid by construction.
- **I2V conditioning:** VAE-encode the conditioning image, then linearly
  interpolate (`lerp`) it into the initial noise latent's first frame at
  `--conditioning_strength`, while building a matching per-token
  conditioning mask. During sampling, each token's *effective timestep* is
  clamped to `min(current_timestep, 1 - conditioning_mask)` (conditioning
  tokens at strength 1.0 sit pinned at timestep 0, i.e. never denoised
  further), and the scheduler step itself is gated so a token whose
  eligibility timestep hasn't been reached yet keeps its lerp'd value
  unchanged rather than being stepped — both ported directly from the
  reference's `prepare_conditioning`/`denoising_step`. This requires the
  DiT's AdaLN path to support a genuinely per-token timestep (not just
  per-sample), which it does natively (see the DiT bullet above) — no
  special-casing needed at the model level, only in the sampling loop
  (`examples/generate_ltx_video.py`).
- **Text encoder (`vidax.models.ltx_video.t5.T5Encoder`):** a standard
  (non-UMT5) T5-XXL encoder — `PixArt-alpha/PixArt-XL-2-1024-MS`'s
  `text_encoder`. Same bidirectional relative-position bucketing formula
  and gated-GELU FFN as Wan's UMT5 port (genuinely shared T5-family
  conventions, not UMT5-specific), but with the one real architectural
  difference UMT5 introduced: standard T5 **shares one
  relative-position-bias table across every layer** (confirmed directly
  against the checkpoint's key list — only `block.0`'s `SelfAttention`
  has a `relative_attention_bias` weight), where UMT5 gives each layer its
  own. A self-contained new module, not a subclass/parameterization of
  `vidax.models.wan.common.t5.T5Encoder`, per this port's design: every new
  model family gets its own independent files, so nothing here can regress
  an existing model.
- **Scheduler (`vidax.schedulers.ltx_rectified_flow.RectifiedFlowScheduler`):**
  a separate file from Wan's `RectifiedFlowScheduler`, not an extension of
  it — two real differences: per-token `timestep` support (required for
  I2V, see above; Wan's scheduler only ever handles a per-sample scalar),
  and the `LinearQuadratic`/`Constant` sigma-schedule shapes every released
  checkpoint's own embedded config actually specifies (verified bit-exact
  against the reference's `linear_quadratic_schedule` — see
  [Status](#status)), on top of the plain `Uniform` linspace Wan's
  `shift`-only schedule effectively is.
- **Checkpoint translator
  (`vidax.translator.mappings.ltx_video.{map_ltx_video_dit_keys,
  map_ltx_video_vae_keys, map_ltx_video_t5_keys}`):** the DiT and VAE
  mappers both read from the *same* loaded state_dict (one checkpoint file
  bundles both, distinguished by `model.diffusion_model.`/`vae.` key
  prefixes) — call both on the same `load_torch_checkpoint_to_jax(...,
  model_type=...)` result rather than loading the file twice.
