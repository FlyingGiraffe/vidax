# LTX-2.5 — Usage Guide

One standalone TPU inference script lives in `examples/`:
`generate_ltx2_5.py` — a single script covering both T2V and I2V (pass
`--image_path` for the latter, omit it for T2V), for the two released
22B checkpoints (dev, distilled). Scope is deliberately narrower than
LTX-2.5's full reference (see [Status](#status)): **video-only** (no audio
generation), **single-stage** (no `LatentUpsampler` two-pass refinement).
Architecturally unrelated to `vidax.models.ltx_video`'s port beyond the
shared "LTX" lineage and the same causal-conv-VAE family: a much larger
22B DiT with cross-attention AdaLN and per-head gated attention, an
8-layer "embeddings connector" between the text encoder and the DiT, a
Gemma-4 12B text encoder (not T5), and an ancestral (SDE) Euler sampler
(not a plain deterministic Euler step) — see [Architecture
notes](#architecture-notes).

| Script | Model | Params | Task | Checkpoint file example |
| --- | --- | --- | --- | --- |
| `generate_ltx2_5.py` | LTX-2.5 | 22B (dev) | T2V, I2V | `ltx-2.5-22b-dev-transformer-bf16.safetensors` |
| `generate_ltx2_5.py` | LTX-2.5 | 22B (distilled) | T2V, I2V | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` |

Requires the `torch` extra (to deserialize the `.safetensors` checkpoints),
the `text` extra (`transformers`/`tokenizers`, for the Gemma-4 tokenizer
embedded in its checkpoint), and the `i2v` extra (`pillow`, for I2V's
conditioning image):

```bash
pip install -e ".[tpu,torch,text,i2v]"
```

---

## LTX-2.5 22B (dev / distilled) — `generate_ltx2_5.py`

Three separate checkpoint files from
[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

- `--dit_checkpoint_path`: `diffusion_models/ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors` — bundles the DiT *and* the video embeddings connector (`model.diffusion_model.video_embeddings_connector.*`, confirmed from the real checkpoint's own keys — the connector's weights don't live in the text-encoder file the way its name might suggest).
- `--vae_checkpoint_path`: `vae/ltx-2.5-video-vae-conv-bf16.safetensors` (default `--vae_variant conv`) or `vae/ltx-2.5-video-vae-bf16.safetensors` (`--vae_variant diffusion`, the transformer/neighborhood-attention decoder — see [Status](#status)).
- `--text_encoder_checkpoint_path`: `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` — bundles the Gemma-4 text tower, its embedded HF tokenizer (extracted from the checkpoint's own `tokenizer_json`/`hf_asset__tokenizer_config.json` tensors, not a separate directory download), and the `text_embedding_projection.video_aggregate_embed` feature-extraction Linear.

Every architecture hyperparameter (DiT `num_layers`/dims/`cross_attention_adaln`/
`apply_gated_attention`/RoPE constants, the connector's own layer count/dims,
VAE block structure, Gemma-4's `hidden_size`/`num_hidden_layers`/mixed
sliding-vs-global attention config) is read directly from each checkpoint's
own embedded metadata (`vidax.models.ltx2_5.configs`) rather than
hardcoded — the same discipline as the LTX-Video port.

### Text-to-video

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --prompt "A golden retriever puppy playing in a field of sunflowers, warm afternoon light" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx2_5_t2v.mp4"
```

`--sampler distilled` (the default) uses the distilled checkpoint's fixed
8-step ancestral-Euler sigma schedule. For the `dev` checkpoint, pass
`--sampler dev` instead — 30 plain-Euler steps with real CFG (`--guidance_scale
3.0` by default), on the reference's own token-count-dependent shifted
sigma schedule:

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --sampler dev \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx2_5_t2v_dev.mp4"
```

A custom `--sigmas` (space-separated, descending, ending in `0.0`) can
override either sampler's built-in schedule.

### Image-to-video

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "A white cat sitting on a surfboard at the beach, gentle waves in the background" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx2_5_i2v.mp4"
```

Conditioning works differently from LTX-Video's I2V (see [Architecture
notes](#architecture-notes)): instead of clamping a per-token *timestep*,
LTX-2.5 threads an explicit per-token `denoise_mask` through the whole
sampling loop (`1` = fully denoise, `0` = frozen at the VAE-encoded clean
value) — ported directly from the reference's
`VideoConditionByLatentIndex`/`post_process_latent`.

### Tensor parallelism

```bash
python examples/generate_ltx2_5.py \
  --dit_checkpoint_path "./checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --vae_checkpoint_path "./checkpoints/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --text_encoder_checkpoint_path "./checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --output_path "out/output_ltx2_5_tp.mp4"
```

`--tensor_parallel_size` (default `4`, **not** `1` — see below) Megatron-
shards the DiT's, the connector's, and Gemma-4's attention heads/FFN
channels via `vidax.core.sharding.shard_wan_params`. Unlike LTX-Video
(where TP was optional at small scale), **no `sharding.py` changes were
needed for the DiT/connector/Gemma-4 side of this port**: LTX-2.5's
DiT/connector submodule names (`to_q`/`to_k`/`to_v`/`to_out_0`/`ff_proj`/
`ff_out`) and Gemma-4's (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/
`up_proj`/`down_proj`) already matched the existing whitelist from
LTX-Video/Reason1/Cosmos3. TP is required, not optional, at any real
scale: the 22B DiT's bf16 weights alone (~44GB) and the 12B Gemma
encoder's (~24GB) don't fit replicated on a single TPU v4 chip's ~32GB
HBM. There is no `--sequence_parallel_size` (same gap as LTX-Video's own
port). (`--vae_variant diffusion` *does* TP the VAE too, and needed two
small `sharding.py` additions for that — see the Diffusion VAE decoder
architecture note below.)

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | Bundles the DiT and the video embeddings connector. |
| `--vae_checkpoint_path` | *required* | The VAE checkpoint matching `--vae_variant`. |
| `--vae_variant` | `conv` | `conv` (ResNet + pixel-shuffle-upsample) or `diffusion` (NATTEN-based, the official demos' decoder). |
| `--text_encoder_checkpoint_path` | *required* | Bundles Gemma-4, its embedded tokenizer, and the feature-extraction projection. |
| `--tensor_parallel_size` | `4` | See [Tensor parallelism](#tensor-parallelism). Must divide `num_devices`, the DiT's `num_attention_heads` (32), and Gemma-4's `num_attention_heads` (16). |
| `--prompt` | *required*, 1+ values | One prompt (broadcast) or exactly `batch_size` prompts. |
| `--image_path` | `None` | Conditioning image, for I2V. Omit for T2V. |
| `--conditioning_strength` | `1.0` | I2V only: how strongly the conditioning image is enforced (`1.0` = the first latent frame is frozen at the encoded image, never denoised). |
| `--text_max_tokens` | `256` | Gemma-4 prompt padding/truncation length. |
| `--dtype` | `bfloat16` | Compute dtype for the VAE, Gemma-4, connector, and DiT activations/latents. |
| `--dit_dtype` | `bfloat16` | Cast target for the DiT's weights. Every released checkpoint ships almost entirely as bf16, **except** every `scale_shift_table`/`prompt_scale_shift_table` (the AdaLN modulation tables), which the checkpoint itself ships in float32 — this port always preserves those at float32 regardless of `--dit_dtype` (`cast_dit_params`, not the generic `cast_to_dtype`); downcasting them was a real, measurable quality bug, see `docs/lessons/ltx2_5_debugging.md`. |
| `--seed` | `0` | Initial noise seed. |
| `--sampler` | `distilled` | `distilled` (8-step ancestral-Euler, `eta=1.0`, no CFG) or `dev` (30-step plain-Euler, `eta=0.0`, `guidance_scale=3.0` CFG, token-count-dependent shifted sigma schedule). |
| `--sigmas` | `None` | Explicit sigma schedule override (space-separated, descending, ending in `0.0`), instead of `--sampler`'s built-in schedule. |
| `--eta` | `None` | Euler `eta` (`1.0` = fully ancestral/SDE, `0.0` = plain deterministic). Defaults to `--sampler`'s own real value. |
| `--guidance_scale` | `None` | CFG scale: `velocity = uncond + guidance_scale * (cond - uncond)`. Defaults to `--sampler`'s own real value (`1.0`/no-CFG for distilled, `3.0` for dev). |
| `--guidance_rescale` | `None` | CFG guidance-rescale strength — corrects the over-saturation/washed-out look plain high-`--guidance_scale` CFG produces, by rescaling toward the conditioned-only prediction's std (`factor = guidance_rescale * (cond.std()/pred.std()) + (1 - guidance_rescale)`). Only applied when CFG is active. Defaults to `--sampler`'s own real value (`0.0`/no-op for distilled, `0.7` for dev). |
| `--negative_prompt` | reference's own default | Negative prompt for CFG (`dev`/`guidance_scale != 1.0` only). |
| `--height` | `704` | Output video height. Must be divisible by the VAE's spatial downscale factor (32: 8x from block structure × `patch_size=4`'s pixel-unshuffle). |
| `--width` | `1216` | Output video width. Same divisibility rule as `--height`. |
| `--num_frames` | `121` | Output frame count. Wants `1 + 8k` (8x temporal downscale) for an exact round-trip. |
| `--fps` | `24` | Output video frame rate. |
| `--offload_dit_weights` | off | Streams the DiT's 48 blocks through HBM `--offload_chunk_size` at a time instead of tracing the whole forward pass as one fused program. Needed at the reference's own `1216x704x121` resolution — not because DiT weights don't fit (they do, comfortably), but because the fused trace doesn't free per-block activations across blocks; see `docs/lessons/ltx2_5_debugging.md` and `docs/weight_offloading.md`. |
| `--offload_chunk_size` | `1` | Blocks per offloaded chunk (must divide 48). `8` is the largest value confirmed to fit at the reference resolution (`tp=4`, both checkpoints). |
| `--output_path` | `output.mp4` | With multiple prompts, each video is saved as `<output_path>_<i>.mp4`. |

---

## Architecture notes

- **DiT (`vidax.models.ltx2_5.dit.LTXDiT`):** a structural port of
  `LTXModel(model_type=LTXModelType.VideoOnly)`, built directly from the
  checkpoint's own embedded config. Shares the PixArt-style
  `AdaLayerNormSingle` timestep embedding and `gelu-approximate`
  FeedForward with `vidax.models.ltx_video.dit`, but with real
  architectural deltas confirmed against the checkpoint (not assumed):
  **cross-attention AdaLN** (`cross_attention_adaln: true` — the block's
  `scale_shift_table` grows to `(9, dim)`, and a second, model-level
  `prompt_adaln_single` embeds `sigma` into a per-block key/value
  modulation, independent of any per-token `timestep` masking);
  **per-head gated attention** (`apply_gated_attention: true` — every
  `Attention` module has an extra `to_gate_logits` Dense producing a
  `2*sigmoid(...)` gate); **no `caption_projection`** inside the DiT at
  all (the 22B checkpoints project text embeddings inside the embeddings
  connector instead, `caption_proj_before_connector: true`); a
  weightless-RMSNorm cross-attention query input (taken from the residual
  *after* the self-attention add, not the raw residual — a delta from
  LTX-Video's own block); and a trained `keyframes_abs_pos_embedding`
  (a no-op for plain T2V/I2V with no generated-keyframe conditioning, but
  present in the checkpoint and loaded regardless).
- **RoPE (`vidax.models.ltx2_5.rope`):** the "split" (rotate-half,
  per-head) convention — `x` split into first-half/second-half (not
  LTX-Video's consecutive-pair interleaving), evaluated at each patch's
  *midpoint* (`use_middle_indices_grid=True`, `[start, end)` bounds per
  token per axis) rather than a single corner coordinate, in float64
  precision (`frequencies_precision: "float64"`).
- **Embeddings connector (`vidax.models.ltx2_5.connector.
  Embeddings1DConnector`):** an 8-layer 1D self-attention transformer
  (reusing `vidax.models.ltx2_5.dit.LTXAttention`/`LTXFeedForward`
  directly — architecturally the same gated, RoPE'd, `q_norm`/`k_norm`'d
  block, just with plain weightless-RMSNorm pre-norm instead of AdaLN, and
  self-attention only) that projects Gemma-4's per-token features into the
  DiT's `cross_attention_dim` space. Padded positions get substituted with
  tiled learnable "register" vectors before attention runs, so no
  attention mask is threaded into the DiT afterward — the connector's own
  `(encoded, additive_attention_mask)` return contract (a `(B, 1, 1, L)`
  already-additive mask) isn't the same shape/convention as
  `LTXDiT.pre_process`'s `encoder_attention_mask` parameter (a plain `(B,
  L)` binary mask it broadcasts itself), so the two aren't interchangeable.
  Its weights live inside the *DiT* checkpoint
  (`video_embeddings_connector.*`), not the text-encoder file.
- **VAE (`vidax.models.ltx2_5.vae.LTXVAE`, conv-decoder variant):** the
  same causal-conv3d/`PixelNorm`/pixel-shuffle family as
  `vidax.models.ltx_video.vae`, with one real delta:
  **`timestep_conditioning=False`** for the released checkpoint (LTX-
  Video's VAE is always noise-conditioned; this one never is — no noise
  injection, no final AdaLN, no `timestep` argument needed at decode
  time). Also **self-normalizing**: `encode` returns a deterministic,
  already-normalized latent mean (no exposed sampling — the log-var half
  the encoder's `conv_out` produces is computed but discarded, matching the
  reference exactly), and `decode` un-normalizes internally, so callers
  pass the same convention both directions with no external
  per-channel-statistics step (unlike LTX-Video's VAE, where normalization
  is the caller's job).
- **Diffusion VAE decoder (`vidax.models.ltx2_5.diffusion_vae.
  DiffusionVideoDecoder`):** the alternative, transformer-based decoder
  (`--vae_variant diffusion`) — shares `vidax.models.ltx2_5.vae.Encoder`
  unchanged (confirmed identical between both checkpoints' embedded
  `config.vae.encoder`), but the decoder is architecturally unrelated to
  the conv decoder: 4 deterministic upsampling stages of `NABlock`s (3D
  *neighborhood* attention — a small, clamped local window per axis, not
  global attention — plus `SwiGLU`, both pre-norm/no-AdaLN) build a
  "context" volume, then a 5th stage of AdaLN-Zero-modulated
  `CombinedDiffusionNABlock`s runs a diffusion decode step on noised
  patchified pixels conditioned on that context. The real checkpoint's
  `default_num_inference_steps=1`/`model_output_type="x0"` makes this a
  **single forward pass** (draw noise once, one stage-5 pass predicts the
  clean pixels directly — no iterative Euler loop for this checkpoint's
  real recipe, though a multi-step fallback is implemented too). RoPE here
  is a genuinely different convention from `vidax.models.ltx2_5.rope`
  (interleaved-pair, per-axis absolute positions) — see the module's own
  docstring. **Single full-volume NA tile only** so far (no
  `diffusion_tiling.py` multi-tile schedule) — see
  `docs/lessons/ltx2_5_debugging.md` for the full port + verification
  writeup, including the compile-time/memory investigation that led to
  Megatron-TP-sharding this decoder too (`--vae_variant diffusion` needed
  two new `vidax.core.sharding` entries, `w_gate`/`w_up`/`w_down` for
  `SwiGLU`, plus renaming the attention output `Dense` to `to_out` to
  reuse an existing entry — a fully replicated single-chip decoder didn't
  fit the reference resolution's HBM budget even after every other memory
  fix).
- **Text encoder (`vidax.models.ltx2_5.gemma4.Gemma4TextModel`):** Gemma-4
  12B (`gemma4-12b-ltx-v1`), the real HF `Gemma4UnifiedTextModel`
  architecture (not an LTX-specific invention) — 48 layers mixing
  `sliding_attention` (local, `head_dim=256`, `theta=1e4`, standard NeoX
  RoPE, 8 KV heads) and `full_attention` (every 6th layer, `head_dim=512`,
  `theta=1e6`, "proportional" RoPE — zeros the trailing 75% of inverse
  frequencies rather than slicing the head dim, so the same rotate-half
  math applies uniformly — a single shared KV head, i.e. near-MQA). Real
  quirks worth knowing: attention scaling is a **fixed `1.0`**, not the
  usual `head_dim**-0.5` (the learned per-head `q_norm` scale does that
  job instead); every layer is **sandwich-normed** (four RMSNorms per
  layer, not the usual two); full-attention layers reuse K's *pre-norm,
  pre-RoPE* projection output as V (`attention_k_eq_v`, no separate
  `v_proj` weight); each layer has a trained per-layer output scalar
  (`layer_scalar`, real values like `0.05`–`0.36`, not a fixed `1.0`); and
  the token embedding is scaled by `sqrt(hidden_size)` with the scale
  itself rounded to the embedding weight's dtype *before* multiplying (an
  intentional, preserved bf16-rounding quirk). `extract_video_features`
  (`FeatureExtractorV2`: per-token-per-layer RMSNorm → rescale → a single
  Linear) turns the 49 layers' worth of hidden states into the connector's
  input — the rescale factor is `sqrt(cross_attention_dim /
  gemma_hidden_size)`, **not** `sqrt(cross_attention_dim / (gemma_hidden_size
  * 49))` (the width of the concatenated per-layer tensor being rescaled —
  the tempting-but-wrong quantity; getting this wrong silently produces
  generically-plausible-but-prompt-disconnected video with no crash or
  shape mismatch to catch it, see `docs/lessons/ltx2_5_debugging.md`).
- **Tokenizer (`vidax.models.ltx2_5.gemma4.Gemma4Tokenizer`):** extracted
  directly from the Gemma checkpoint's own embedded `tokenizer_json`/
  `hf_asset__tokenizer_config.json` raw-byte tensors (a full HF
  `tokenizers.Tokenizer`, not a separate directory download the way
  LTX-Video's T5 tokenizer ships) via `tokenizers.Tokenizer.from_buffer` +
  `transformers.PreTrainedTokenizerFast`.
- **I2V conditioning:** a genuinely different mechanism from LTX-Video's
  own I2V (see the module docstring of
  [`examples/generate_ltx2_5.py`](../../examples/generate_ltx2_5.py) for
  the full derivation from the reference's `VideoConditionByLatentIndex`/
  `post_process_latent`/`timesteps_from_mask`). LTX-Video clamps a
  per-token *effective timestep*; LTX-2.5 instead threads an explicit
  per-token `denoise_mask` (`1` = denoise, `0` = frozen) through three
  places every step: the DiT's own per-token `timestep = denoise_mask *
  sigma` input, a blend of the x0 estimate back toward the clean
  conditioning value (`denoised*mask + clean*(1-mask)`), and the same
  blend applied again to the sampler's stepped output. The initial noisy
  latent is built the same way: `lerp(clean_latent, noise, denoise_mask)`.
- **Scheduler (`vidax.schedulers.ltx2_5_ancestral_euler.
  AncestralEulerScheduler`):** a structural port of
  `EulerAncestralDiffusionStep`, covering both real recipes with one
  class via `eta`: the **distilled** checkpoint's own pipeline uses the
  *ancestral* (SDE) form (`eta=1.0`, `should_use_ancestral_sampler`: each
  step advances deterministically to an intermediate `sigma_down`, then
  re-noises back up to `sigma_next`, rescaling to stay
  variance-preserving) on its fixed 9-value sigma schedule; the **`dev`**
  checkpoint's own one-stage pipeline actually defaults to *plain*
  deterministic Euler (`eta=0.0` — confirmed from `ltx_pipelines.utils.
  blocks.DiffusionStage.__call__`'s own default `stepper=
  EulerDiffusionStep()`, not the ancestral loop distilled's stage 1
  explicitly opts into) on `compute_shifted_sigmas` — a token-count-
  dependent time-shift of a uniform `linspace(1, 0, steps+1)` (the same
  family of formula as SD3/Flux's resolution-dependent shift), "stretched"
  so the last non-terminal sigma lands exactly at `0.1`, ported from
  `LTX2Scheduler.execute`. Both are a real, verified departure from the
  plain deterministic Euler step `vidax.schedulers.ltx_rectified_flow`
  implements for LTX-Video. Noise for the ancestral re-injection comes
  from JAX's own PRNG (per-step `--seed`-derived keys), not a
  torch-`Generator`-matched stream — mathematically equivalent to the
  reference, not bit-for-bit reproducible against it for a given seed
  (same as every other model in this repo).
- **Checkpoint translator (`vidax.translator.mappings.ltx2_5.
  {map_ltx2_5_dit_keys, map_ltx2_5_connector_keys, map_ltx2_5_vae_keys,
  map_gemma4_text_keys}`):** the DiT and connector mappers both read from
  the same loaded DiT-checkpoint state_dict (its `video_embeddings_
  connector.*` keys, distinguished by prefix) — call both on the same
  `load_torch_checkpoint_to_jax(...)` result rather than loading the file
  twice, same pattern as LTX-Video's DiT+VAE sharing one file.
