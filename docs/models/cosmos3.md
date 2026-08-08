# Cosmos 3 (Nano) — Usage Guide

One standalone TPU inference script currently lives in `examples/`:
`generate_cosmos3_nano.py`, for the 16B Nano checkpoint's text-to-video and
image-to-video generation — the only two of Cosmos 3's several surfaces this
port covers (see [Scope](#scope) below for what's deliberately left out and
why).

Cosmos 3 is **architecturally unrelated** to Wan or Cosmos-Predict2.5: not
another DiT variant, but an omnimodal Mixture-of-Transformers (MoT)
combining a causal "understanding" (text) pathway with a full-attention
"generation" (diffusion) pathway inside one shared 36-layer transformer, no
AdaLN modulation anywhere, and a genuinely different (interleaved) 3D
rotary position scheme. See [Architecture notes](#architecture-notes) for
the full picture, and [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
for the shared TPU/JAX engineering background this still builds on
(sharding, flash attention, dtype conventions).

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_cosmos3_nano.py` | Cosmos3-Nano | 16B | Text2Video, Image2Video | `Cosmos3-Nano/` |

Requires the `torch` extra is **not** needed here (the checkpoint ships as
`.safetensors`, loaded directly) — but does need the `text` extra
(`transformers`, for the `Qwen2TokenizerFast` tokenizer + chat template) and
the `i2v` extra (`pillow`, for image2video's conditioning frame):

```bash
pip install -e ".[tpu,text,i2v]"
```

---

## Cosmos3-Nano (16B) — `generate_cosmos3_nano.py`

Ships as one self-contained HuggingFace `diffusers`-format repo
(`Cosmos3-Nano/`), with the components this port uses at:
`transformer/diffusion_pytorch_model.safetensors.index.json` (the DiT, `Cosmos3OmniTransformer`
— sharded, pass the `.index.json` manifest), `vae/diffusion_pytorch_model.safetensors`
(the VAE — Wan2.2-TI2V-5B's own VAE, reused verbatim, see
[Architecture notes](#architecture-notes)), and `text_tokenizer/` (a
directory, pass as-is to `--tokenizer_path`). `vision_encoder/` and
`sound_tokenizer/` are part of the checkpoint but never loaded by this port
— see [Scope](#scope).

**Memory note:** at 16B parameters (~29GB in bf16), this model is close to
or larger than a single TPU v4 chip's ~30GB HBM budget on its own, before
any activations. `--tensor_parallel_size` (Megatron-style, sharding
attention heads/FFN channels across devices) is not optional the way it is
for Cosmos-Predict2.5's 2B — use at least `--tensor_parallel_size 4` (all
devices on a v4-8) unless running on a pod slice with proportionally more
HBM per chip.

### Text2Video

```bash
python examples/generate_cosmos3_nano.py \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --max_text_len 256 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_t2v.mp4"
```

### Image2Video

```bash
python examples/generate_cosmos3_nano.py \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "A cat wearing sunglasses on a boat in the ocean, waves splashing" \
  --max_text_len 256 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_i2v.mp4"
```

`--image_path` anchors latent frame 0 to the VAE-encoded conditioning image
(resized to `--height`/`--width`, center-cropped is *not* applied — resize
only) and denoises the remaining frames, matching Cosmos-Predict2.5's own
image2world frame-substitution mechanism (re-clamping the known frame's
latent back into `x` after every sampling step) — not a fundamentally new
mechanism, just carried over.

### Quick testing

```bash
python examples/generate_cosmos3_nano.py \
  --dit_checkpoint_path ... --vae_checkpoint_path ... --tokenizer_path ... \
  --prompt "..." \
  --tensor_parallel_size 4 \
  --height 256 --width 256 --num_frames 9 --num_steps 10 --max_text_len 256 \
  --output_path out/quick_test.mp4
```

Same rationale as Cosmos-Predict2.5's quick-testing section: full-resolution
(720x1280), full-step (35) runs are slow to iterate with. This config still
exercises the full pipeline (tokenization, packed-sequence assembly, mRoPE,
the dual-pathway DiT, Karras-sigma UniPC sampling, VAE decode) end to end —
this exact command is what the first successful real-weight run used,
producing a clearly recognizable, prompt-matching result on the first try
(see [Status](#status)).

Note `--max_text_len` needs to comfortably fit the *negative* prompt too —
`vidax`'s own default negative prompt tokenizes to ~180 tokens; pass a
shorter `--negative_prompt` or raise `--max_text_len` if you hit the
tokenized-length assertion.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | The `transformer/diffusion_pytorch_model.safetensors.index.json` manifest — a flat-layout state_dict (`layers.N.self_attn.to_q.weight`, no `model.`/`net.` prefix), unlike Cosmos-Predict2.5's nested `net.blocks.N...`. |
| `--vae_checkpoint_path` | *required* | `vae/diffusion_pytorch_model.safetensors` — Wan2.2-TI2V-5B's VAE, but in `diffusers`' `AutoencoderKLWan` checkpoint *layout* (different key names than the original Wan repo release, same architecture — loaded via `model_type="wan2.2_vae_diffusers"`, a separate mapper from Wan2.2's own `"wan2.2_vae"`). See [Architecture notes](#architecture-notes). |
| `--tokenizer_path` | *required* | The `text_tokenizer/` directory (Qwen2TokenizerFast + chat template). |
| `--image_path` | `None` | Conditioning image for image2video. Resized (not cropped) to `--height`/`--width`. |
| `--prompt` | *required* | Text prompt. |
| `--negative_prompt` | vidax's own quality-negative-prompt | CFG negative prompt — see the "Quick testing" note above about `--max_text_len`. |
| `--max_text_len` | `128` | Fixed padded text-token length. JAX needs a static shape; the reference uses each prompt's exact tokenized length instead — this port pads to a fixed length with an explicit validity mask so `gen`'s cross-attention correctly excludes padding positions (see [Architecture notes](#architecture-notes)). |
| `--guide_scale` | `6.0` | CFG scale. Matches the reference pipeline's default. |
| `--tensor_parallel_size` | `1` | Devices to Megatron-shard the DiT's attention heads/FFN channels across. Must divide `num_devices`, `num_attention_heads` (32), and `num_key_value_heads` (8, GQA — the binding constraint, so `tp` in `{1,2,4,8}` in practice). See the memory note above — effectively required, not optional, at this model's size. |
| `--dtype` | `bfloat16` | `float32` \| `float16` \| `bfloat16`. `float16` will fail at runtime — TPU's XLA backend doesn't implement `float16` matmuls. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `35` | UniPC sampling steps. |
| `--karras_sigma_min` / `--karras_sigma_max` | `0.147` / `200.0` | Karras noise-schedule bounds — matches `scheduler/scheduler_config.json`'s own `sigma_min`/`sigma_max` (this checkpoint's actual default schedule; a genuinely different curve from Cosmos-Predict2.5's linear/`shift`-warped one, see [Architecture notes](#architecture-notes)). |
| `--height` | `704` | Output video height. Must be divisible by 32 (VAE's 16x spatial compression × the DiT's `latent_patch_size=2`). |
| `--width` | `1280` | Output video width. Same divisibility rule as `--height`. |
| `--num_frames` | `93` | Output frame count. |
| `--fps` | `24.0` | Output video frame rate, also injected into the mRoPE temporal modulation and the prompt's duration-metadata sentence. |
| `--output_path` | `output_cosmos3_nano.mp4` | Output video path. |

### Status

**Verified end-to-end on real weights, both text2video and image2video,
producing coherent, prompt-matching output on the first successful full
run** — a recognizable red panda climbing a bamboo stalk for text2video; a
stable, identity-preserving subject across frames for image2video (256x256,
9 frames, 10 steps; see the [Quick testing](#quick-testing) command). No
extended debugging round was needed this time — the two dominant lessons
from Cosmos-Predict2.5's port (verify architecture pieces in isolation
*before* touching real weights; the sampler/preconditioning boundary is the
highest-leverage place a diffusion port silently breaks) were applied
proactively:

- The interleaved 3D mRoPE (`vidax.models.cosmos3.common.mrope`) was
  unit-tested for its relative-position invariant (`q_i . k_j` depends only
  on `i - j`, checked with fixed content vectors at varying positions)
  *before* any real-weight run.
- Weight loading was verified with an exact key-set + shape match against
  both the DiT's and the VAE's own initialized parameter trees (not just
  "did it load without an exception") before the first forward pass.
- The sampling loop feeds the DiT's raw output directly to the scheduler as
  velocity, with no EDM-style preconditioning wrapper — applying
  Cosmos-Predict2.5's dominant lesson from the start rather than
  rediscovering it. Confirmed directly against the real reference pipeline
  code (`refs/diffusers-cosmos3/pipeline_cosmos3_omni.py`'s own denoising
  loop) before writing this script, not assumed from precedent.

Only tested so far at low resolution/frame count/step count (256x256, 9
frames, 10 steps) — a fuller-resolution, more-steps run to confirm quality
holds at scale is a natural next step, same as it was for Cosmos-Predict2.5.

### Scope

This port covers **text2video and image2video only** — a deliberate,
explicit scope decision, not a partial/interrupted port. Cosmos 3 also
supports several other surfaces this repo doesn't implement:

- **Video2video, action-conditioned generation, sound-conditioned
  generation** — the reference pipeline supports all three (`video`,
  `action`, `enable_sound` arguments), but none are wired up here.
- **The "Reasoner" surface** (causal-LM-only, text/vision-in, text-out —
  world understanding, captioning, grounding) — this port never loads
  `lm_head` or `vision_encoder/` (`Qwen3VLVisionModel`) at all, since
  neither is needed for pure generation: text conditioning goes through
  `embed_tokens` directly (no separate text encoder, unlike
  Cosmos-Predict2.5's Reason1), and image conditioning for image2video goes
  through the *VAE*, not a vision encoder tower.

If any of these become interesting later, the packed dual-pathway
transformer (`vidax.models.cosmos3.nano.dit.Cosmos3Transformer`) and mRoPE
module are already shared, general-purpose infrastructure — most of the
additional work would be in the pipeline script (packing/position-id
construction for the extra modality) and, for the Reasoner surface, porting
`vision_encoder`/`lm_head` and switching the packed-sequence design back
toward something closer to the reference's ragged/multi-item batching (see
[Architecture notes](#architecture-notes)'s note on why this port doesn't
need that for T2V/I2V specifically).

---

## Architecture notes

- **Not sparse MoE.** The checkpoint config's `"use_moe": true` doesn't mean
  token-routing sparse MoE with a router/gate — it means each of the 36
  decoder layers carries **two full parallel weight sets**: one for the
  causal "und" (understanding/text) pathway, one for the full-attention
  "gen" (generation/diffusion) pathway (`mlp` vs `mlp_moe_gen`,
  `input_layernorm` vs `input_layernorm_moe_gen`, etc. in the checkpoint's
  own key names) — sharing one packed token sequence, never sharing
  weights.
- **Packed sequence, fixed-shape port.** The reference packs text + vision
  (+ optionally sound/action) tokens into one flat, ragged
  `[sequence_length, hidden_size]` buffer per forward call, addressed by
  index arrays — built for its flexible multi-item *training* batches. This
  port instead uses an ordinary `(B, seq_len, hidden)` tensor with a real
  batch axis: for a fixed `(height, width, num_frames)` and a fixed padded
  text length (`--max_text_len`), the packed sequence length is static and
  known ahead of time, so JAX's static-shape requirement is satisfied
  without porting any of the reference's ragged/list-of-tensors machinery.
  `und` (text) is padded to `--max_text_len` with an explicit validity mask
  (`und_valid_mask`) so `gen`'s cross-attention over `und`'s keys/values
  correctly excludes padding positions via an additive bias — `und`'s own
  causal self-attention needs no such mask, since causal masking alone
  already keeps every real token from seeing any (necessarily later)
  padding position.
- **Dual-pathway attention** (`vidax.models.cosmos3.common.dit_layers.Cosmos3PackedMoTAttention`):
  `und` tokens self-attend causally (GQA, 32 query heads / 8 KV heads,
  head_dim 128 — a small in-line causal LM); `gen` tokens do full
  (non-causal) attention over **both** `und` and `gen` keys/values —
  one-directional information flow, `und -> gen` only, `und` never reads
  `gen`. The large `gen` pathway (tens of thousands of video-patch tokens
  at real resolutions) uses an *additive* bias (not a boolean mask) for its
  padding exclusion specifically so it can still take the Pallas
  flash-attention path — `vidax.core.attention.dot_product_attention` was
  extended to carry a `bias` through the single-device flash path (it
  previously forced the slow XLA-materializing fallback whenever *either*
  `bias` or `mask` was given).
- **Interleaved 3D mRoPE** (`vidax.models.cosmos3.common.mrope`): a
  genuinely different scheme from both Wan's (interleaved-pair rotation,
  unevenly-split block axes) and Cosmos-Predict2.5's (rotate-half,
  concatenated-block axes) RoPE. A single shared `inv_freq` table (over the
  full `head_dim // 2` channels) is evaluated at all three (T, H, W) axes'
  position ids, then the first `min(rope_axes_dim[1], rope_axes_dim[2]) * 3`
  channels are rearranged into repeating `(T, H, W)` triples — each triple
  shares the *same* underlying frequency, just evaluated at each axis's own
  position — with the remaining tail channels staying purely T-indexed.
  `rope_theta=5e6` (vs. Cosmos-Predict2.5's `1e4`). Verified via the
  relative-position invariant (`q_i . k_j` after rotation depends only on
  `i - j`) before any real-weight run.
- **No AdaLN modulation anywhere** (unlike every other DiT in this repo).
  The diffusion timestep is injected exactly **once**, additively, directly
  into the noisy vision tokens themselves (matching the reference's
  `_apply_timestep_embeds_to_noisy_tokens`: a sinusoidal embedding +
  2-layer MLP, added into each noisy frame's patch tokens, masked to
  noisy — not conditioned/clean — frames only), then flows through
  ordinary pre-norm transformer blocks with no further timestep-conditioned
  scale/shift. Same `timestep_scale=0.001` DiT-internal rescale mechanism as
  Cosmos-Predict2.5's `MinimalV1LVGDiT` (see that model's docs for why this
  specific detail matters — missing it was one of the four real bugs found
  there).
- **No preconditioning wrapper — the DiT's raw output is the velocity,
  used directly.** Confirmed directly against `refs/diffusers-cosmos3/
  pipeline_cosmos3_omni.py`'s own denoising loop (not assumed from
  Cosmos-Predict2.5's precedent, though it turned out to match): the raw
  noisy latent is passed to the DiT unscaled, its timestep is
  `sigma * num_train_timesteps` (the DiT's own `timestep_scale` divides
  this back down internally), and the DiT's raw output (after CFG
  combination) is fed straight into the scheduler's `x0 = sample - sigma_t
  * model_output` — no `c_in`/`c_skip`/`c_out` reconstruction anywhere.
- **VAE:** Cosmos3-Nano's VAE is Wan2.2-TI2V-5B's own VAE, confirmed
  directly (`vae/config.json`'s `_name_or_path` and every architecture
  field — `base_dim=160`, `decoder_base_dim=256`, `z_dim=48`,
  `dim_mult=[1,2,4,4]` — match `vidax.models.wan.wan2_2.vae`'s defaults
  exactly), but shipped in `diffusers`' own `AutoencoderKLWan` checkpoint
  *layout*: `down_blocks`/`up_blocks` with nested `resnets`/`downsampler`/
  `upsampler` submodules and `quant_conv`/`post_quant_conv`, rather than the
  original Wan repo's `downsamples`/`upsamples` naming and fused
  `conv1`/`conv2`. `vidax.translator.mappings.wan2_2_diffusers` is a new,
  separate mapper for this layout (`model_type="wan2.2_vae_diffusers"`) —
  the underlying `vidax.models.wan.wan2_2.vae.WanVAEEncoder`/`WanVAEDecoder`
  modules are reused completely unchanged, only the checkpoint *parsing*
  differs. Verified via the same exact key-set + shape match technique used
  for the DiT.
- **Text tokenization/templating** (`generate_cosmos3_nano.py`'s
  `tokenize_prompt`): matches the reference's own `Cosmos3OmniPipeline.
  tokenize_prompt` — a fixed system prompt, resolution ("This video is of
  HxW resolution.") and duration ("The video is N seconds long and is of F
  FPS.") metadata sentences appended to the prompt (inverse-phrased for the
  negative prompt), chat-templated, with `<|vision_start|>` + eos appended
  as the generation-start markers. No separate text encoder is invoked —
  token ids feed `embed_tokens` directly inside the shared transformer.
- **Sampler:** `vidax.schedulers.unipc.FlowUniPCMultistepScheduler`, same
  core predictor-corrector solver as Cosmos-Predict2.5, but with a new
  Karras-sigma schedule path added (`use_karras_sigmas=True`) — this
  checkpoint's actual default (`scheduler/scheduler_config.json`:
  `use_karras_sigmas: true`, `use_flow_sigmas: true`,
  `sigma_min: 0.147`, `sigma_max: 200.0`), a genuinely different curve from
  Cosmos-Predict2.5's linear-sigma/`shift`-warp schedule, not just a
  different `shift` value. Matches `diffusers`' own
  `UniPCMultistepScheduler._convert_to_karras` (an elucidating-diffusion-
  models power-law ramp) followed by its `sigma / (sigma + 1)` remap into
  this scheduler's own `alpha = 1 - sigma` convention.

## Coming later

- **Cosmos3-Super/Edge** (the 64B and 4B siblings), if useful.
- **Video2video, action-conditioned, and sound-conditioned generation**,
  and the **Reasoner** surface — see [Scope](#scope) for why these are out
  for now and what porting them would involve.

See the [parity matrix in the root README](../../README.md#model-support--parity-matrix)
for the up-to-date status across all variants.
