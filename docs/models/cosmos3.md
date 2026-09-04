# Cosmos 3 (Nano / Edge) — Usage Guide

One standalone TPU inference script lives in `examples/`: `generate_cosmos3.py`,
covering both released checkpoint sizes' text-to-video and image-to-video
generation via `--model_size {nano,edge}` — the only two of Cosmos 3's
several surfaces this port covers (see [Scope](#scope) below for what's
deliberately left out and why).

Cosmos 3 is **architecturally unrelated** to Wan or Cosmos-Predict2.5: not
another DiT variant, but an omnimodal Mixture-of-Transformers (MoT)
combining a causal "understanding" (text) pathway with a full-attention
"generation" (diffusion) pathway inside one shared transformer, no AdaLN
modulation anywhere, and a genuinely different (interleaved) 3D rotary
position scheme. See [Architecture notes](#architecture-notes) for the full
picture, and [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
for the shared TPU/JAX engineering background this still builds on
(sharding, flash attention, dtype conventions).

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_cosmos3.py --model_size nano` | Cosmos3-Nano | 16B | Text2Video, Image2Video | `Cosmos3-Nano/` |
| `generate_cosmos3.py --model_size edge` | Cosmos3-Edge | 4B | Text2Video, Image2Video | `Cosmos3-Edge/` |

This model doesn't actually touch `torch` (its checkpoint ships as
`.safetensors`, loaded directly); it does use `transformers` (the
`Qwen2TokenizerFast` tokenizer + chat template) and `pillow` (image2video's
conditioning frame). All dependencies are installed by default — on a Cloud
TPU VM add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

Both sizes share the exact same weight layout and DiT code
(`vidax.models.cosmos3.dit.Cosmos3Transformer`); `--model_size` just selects
which hyperparameter preset (`vidax.models.cosmos3.configs.NANO_CONFIG` /
`EDGE_CONFIG`) to build it with. Edge additionally uses a squared-ReLU MLP
(`hidden_act="relu2"`, no `gate_proj`) instead of Nano's SwiGLU, and skips
`und`'s own q/k RMSNorm in favor of a separate norm just for what `gen`
reads from `und` (`k_norm_und_for_gen`) — both handled transparently by the
shared code, see [Architecture notes](#architecture-notes).

---

## Cosmos3-Nano (16B) — `--model_size nano`

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
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --max_text_len 3072 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_nano_t2v.mp4"
```

A short plain-text `--prompt` like the one above works for Nano, but see
[Prompting](#prompting) below — the checkpoints are documented to expect
richer, JSON-structured prompts, and Edge in particular needs this for good
output quality.

### Image2Video

```bash
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Nano/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Nano/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Nano/text_tokenizer" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "A cat wearing sunglasses on a boat in the ocean, waves splashing" \
  --negative_prompt "Low quality, blurry, oversaturated, static, distorted." \
  --max_text_len 128 \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos3_nano_i2v.mp4"
```

A short `--prompt` here needs a comparably short `--negative_prompt` too — see
[Prompting](#prompting) for why (this port's default negative prompt is a
much longer, JSON-structured one, meant to pair with an equally detailed
positive prompt).

`--image_path` anchors latent frame 0 to the VAE-encoded conditioning image
(resized to `--height`/`--width`, center-cropped is *not* applied — resize
only) and denoises the remaining frames, matching Cosmos-Predict2.5's own
image2world frame-substitution mechanism (re-clamping the known frame's
latent back into `x` after every sampling step) — not a fundamentally new
mechanism, just carried over.

---

## Cosmos3-Edge (4B) — `--model_size edge`

Same `diffusers`-format layout as Nano (`transformer/`, `vae/`,
`text_tokenizer/`), smaller (`hidden_size=2048`, 28 layers, 16 attention
heads / 8 KV heads vs. Nano's 4096/36/32/8) and comfortably fits a single
TPU v4 chip, so `--tensor_parallel_size 1` works, though multi-chip still
helps at higher resolution/step counts. Uses its own tokenizer under
`Cosmos3-Edge/text_tokenizer/` (different vocab size than Nano's, 131072 vs
151936 — pass `--model_size edge`'s matching `--tokenizer_path`, don't mix
checkpoints across sizes).

**Resolution/frame count/scheduler are not interchangeable with Nano's, and
T2V and I2V use *different* scheduler values from each other** — always
pass Edge's explicit `--height`/`--width`/`--num_frames`/`--num_steps`/
`--use_karras_sigmas`/`--shift` as shown below. Reusing Nano's defaults or
mixing T2V's/I2V's values doesn't error — it just produces badly wrong
output.

### Text2Video

```bash
python examples/generate_cosmos3.py \
  --model_size edge \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Edge/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Edge/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Edge/text_tokenizer" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --max_text_len 3072 \
  --tensor_parallel_size 4 \
  --height 480 --width 832 --num_frames 121 \
  --num_steps 35 --use_karras_sigmas false --shift 10.0 \
  --add_duration_template false --add_resolution_template false \
  --output_path "out/output_cosmos3_edge_t2v.mp4"
```

For Edge especially, quality depends heavily on prompt richness — see
[Prompting](#prompting).

### Image2Video

```bash
python examples/generate_cosmos3.py \
  --model_size edge \
  --dit_checkpoint_path "./checkpoints/Cosmos3-Edge/transformer/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Cosmos3-Edge/vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "./checkpoints/Cosmos3-Edge/text_tokenizer" \
  --image_path "./checkpoints/Cosmos3-Edge/assets/example_i2v_input.jpg" \
  --prompt "A car driving along a coastal mountain road" \
  --negative_prompt "Low quality, blurry, oversaturated, static, distorted." \
  --max_text_len 128 \
  --tensor_parallel_size 4 \
  --height 480 --width 832 --num_frames 121 \
  --num_steps 20 --use_karras_sigmas false --shift 12.0 \
  --add_duration_template false --add_resolution_template false \
  --output_path "out/output_cosmos3_edge_i2v.mp4"
```

### Quick testing

```bash
python examples/generate_cosmos3.py \
  --model_size nano \
  --dit_checkpoint_path ... --vae_checkpoint_path ... --tokenizer_path ... \
  --prompt "..." \
  --tensor_parallel_size 4 \
  --height 256 --width 256 --num_frames 9 --num_steps 10 --max_text_len 3072 \
  --output_path out/quick_test.mp4
```

Same rationale as Cosmos-Predict2.5's quick-testing section: full-resolution
(704x1280), full-step (35) runs are slow to iterate with. This config still
exercises the full pipeline (tokenization, packed-sequence assembly, mRoPE,
the dual-pathway DiT, Karras-sigma UniPC sampling, VAE decode) end to end —
**for Nano.** This exact quick-test config is *not* valid for Edge: swapping
only `--model_size` (keeping 256x256/9 frames, Karras sigmas) reliably
produces degraded, incoherent Edge output — a real trap, since it doesn't
error, it just looks badly wrong. For a genuine Edge T2V smoke test, use its
real recipe's scheduler (`--use_karras_sigmas false --shift 10.0`) even at
a reduced frame count for faster iteration, e.g.
`--height 480 --width 832 --num_frames 50 --num_steps 35`.

`--max_text_len` needs to comfortably fit the *negative* prompt too — the
default negative prompt (see [Prompting](#prompting)) tokenizes to ~2800
tokens, hence the `3072` default; pass a shorter `--negative_prompt` or
raise `--max_text_len` further if you hit the tokenized-length assertion.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `nano` | `nano` or `edge` — selects `NANO_CONFIG`/`EDGE_CONFIG` from `vidax.models.cosmos3.configs`. Must match `--dit_checkpoint_path`'s actual checkpoint. |
| `--dit_checkpoint_path` | *required* | The `transformer/diffusion_pytorch_model.safetensors.index.json` manifest — a flat-layout state_dict (`layers.N.self_attn.to_q.weight`, no `model.`/`net.` prefix), unlike Cosmos-Predict2.5's nested `net.blocks.N...`. |
| `--vae_checkpoint_path` | *required* | `vae/diffusion_pytorch_model.safetensors` — Wan2.2-TI2V-5B's VAE, but in `diffusers`' `AutoencoderKLWan` checkpoint *layout* (different key names than the original Wan repo release, same architecture — loaded via `model_type="wan2.2_vae_diffusers"`, a separate mapper from Wan2.2's own `"wan2.2_vae"`). Identical for both Nano and Edge. See [Architecture notes](#architecture-notes). |
| `--tokenizer_path` | *required* | The checkpoint's own `text_tokenizer/` directory (Qwen2TokenizerFast + chat template) — Nano and Edge ship different tokenizers, don't mix them. |
| `--image_path` | `None` | Conditioning image for image2video. Resized (not cropped) to `--height`/`--width`. |
| `--prompt` | *required* | Text prompt — see [Prompting](#prompting) for format recommendations. |
| `--negative_prompt` | a real, JSON-structured negative prompt (see [Prompting](#prompting)) | CFG negative prompt. |
| `--add_duration_template` / `--add_resolution_template` | `true` | Whether to append the reference's duration/FPS and resolution metadata sentences to the prompt. Every real usage example in `refs/cosmos-main` passes `false` for both — see [Prompting](#prompting). |
| `--max_text_len` | `3072` | Fixed padded text-token length. JAX needs a static shape; the reference uses each prompt's exact tokenized length instead — this port pads to a fixed length with an explicit validity mask so `gen`'s cross-attention correctly excludes padding positions (see [Architecture notes](#architecture-notes)). |
| `--guide_scale` | `6.0` | CFG scale. Matches the reference pipeline's default. |
| `--tensor_parallel_size` | `1` | Devices to Megatron-shard the DiT's attention heads/FFN channels across. Must divide `num_devices`, `num_attention_heads`, and `num_key_value_heads` (32/8 for Nano, 16/8 for Edge — GQA's KV-head count is the binding constraint, so `tp` in `{1,2,4,8}` in practice for either). Effectively required (not just a memory-saving option) for Nano at its size; optional but still useful for Edge. |
| `--dtype` | `bfloat16` | `float32` \| `float16` \| `bfloat16`. `float16` will fail at runtime — TPU's XLA backend doesn't implement `float16` matmuls. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `35` | UniPC sampling steps. Nano default; also Edge T2V's real value (unchanged from Nano) — Edge I2V's real recipe uses `20`. |
| `--use_karras_sigmas` | `true` | Nano's schedule (its `scheduler_config.json` default). Edge's real recipe (T2V and I2V alike) instead uses a plain shift-warped (non-Karras) schedule — pass `--use_karras_sigmas false` for Edge, with `--shift` set per task (see below). |
| `--shift` | `5.0` | Shift-warp factor for the non-Karras schedule; only read when `--use_karras_sigmas false`. Edge's real value is `10.0` for **T2V** (same as Nano's) but `12.0` for **I2V** — don't reuse one task's value for the other, see [Cosmos3-Edge](#cosmos3-edge-4b--model_size-edge). |
| `--karras_sigma_min` / `--karras_sigma_max` | `0.147` / `200.0` | Karras noise-schedule bounds, only read when `--use_karras_sigmas true` — matches Nano's `scheduler/scheduler_config.json` `sigma_min`/`sigma_max` (a genuinely different curve from Cosmos-Predict2.5's linear/`shift`-warped one, see [Architecture notes](#architecture-notes)). |
| `--height` | `704` | Output video height. Must be divisible by 32 (VAE's 16x spatial compression × the DiT's `latent_patch_size=2`). This default targets **Nano's** documented spec — Edge needs an explicit override (e.g. `480`, see [Cosmos3-Edge](#cosmos3-edge-4b--model_size-edge)'s note above). |
| `--width` | `1280` | Output video width. Same divisibility rule as `--height`. Nano default — override for Edge (e.g. `832`). |
| `--num_frames` | `93` | Output frame count. Nano default — Edge's real recipe uses `121`. |
| `--fps` | `24.0` | Output video frame rate, also injected into the mRoPE temporal modulation and the prompt's duration-metadata sentence. |
| `--output_path` | `output_cosmos3.mp4` | Output video path. |

### Prompting

Cosmos3's checkpoints document that prompts should be **"upsampled into a
specific JSON structure"** for optimal quality (see each checkpoint's own
`README.md`), not passed as a short plain-text sentence. The JSON structure
covers subjects, background/setting, lighting, aesthetics, cinematography,
and a temporal caption — see `examples/assets/cosmos3_t2v_prompt.json` for
a real example, and NVIDIA's `cosmos-framework` repo for the upsampling
tool itself (an LLM prompt-expansion step, not something this repo
implements).

This matters more than it might look: a short prompt like `"A red panda
climbing a bamboo tree"` still produces recognizable output on Nano (16B),
but on Edge (4B) the same short prompt produces flat, oversaturated, mostly
featureless output — swapping in a real JSON-upsampled prompt (same
scheduler settings, same everything else) produces fully photorealistic,
detailed output instead. `--prompt` still accepts a plain string (this port
doesn't require JSON), but for Edge specifically, use a JSON-structured one
for real work.

`generate_cosmos3.py`'s **default negative prompt** is a real, JSON-structured
negative prompt (`examples/assets/cosmos3_t2v_negative_prompt.json`, from
`refs/cosmos-main`), not a short placeholder — pair it with a comparably
detailed positive prompt. Pairing a short positive prompt with a much
longer/richer negative prompt is a real trap: the large difference in real
token count between the cond and uncond passes inflates the vision-segment
mRoPE offset gap between them and can produce badly corrupted output. Keep
positive and negative prompts roughly comparable in scale.

`--add_duration_template`/`--add_resolution_template` (both default `true`,
matching the reference pipeline's own default) append metadata sentences
like *"This video is of 832x480 resolution."* to the prompt. Every real
usage example in `refs/cosmos-main` passes `false` for both — do the same
unless you have a specific reason not to.

---

## Architecture notes

- **Not sparse MoE.** The checkpoint config's `"use_moe": true` doesn't mean
  token-routing sparse MoE with a router/gate — it means each decoder layer
  carries **two full parallel weight sets**: one for the causal "und"
  (understanding/text) pathway, one for the full-attention "gen"
  (generation/diffusion) pathway (`mlp` vs `mlp_moe_gen`, `input_layernorm`
  vs `input_layernorm_moe_gen`, etc. in the checkpoint's own key names) —
  sharing one packed token sequence, never sharing weights.
- **Shared code, per-checkpoint toggles.** Nano and Edge use the exact same
  `Cosmos3Transformer`/`Cosmos3VLTextMoTDecoderLayer`/`Cosmos3PackedMoTAttention`
  classes (`vidax.models.cosmos3.dit{,_layers}.py`), parameterized by
  `vidax.models.cosmos3.configs.NANO_CONFIG`/`EDGE_CONFIG`:
  - `hidden_act`: `"silu"` (Nano, SwiGLU: `gate_proj`+`up_proj`) vs.
    `"relu2"` (Edge, squared ReLU: `down_proj(relu(up_proj(x))**2)`, no
    `gate_proj` — matches Edge's checkpoint having no `mlp.gate_proj` weight).
  - `qk_norm_for_text`: whether `und`'s own q/k get an RMSNorm (`True` for
    Nano; `False` for Edge, whose checkpoint has no `self_attn.norm_q`/
    `norm_k` weights at all).
  - `use_und_k_norm_for_gen`: only takes effect when `qk_norm_for_text` is
    `False`. When `True` (Edge), `gen`'s cross-attention reads `und`'s keys
    through a second, separately-normed-and-RoPE'd projection
    (`k_norm_und_for_gen`) instead of the plain `k_und` that `und`'s own
    causal self-attention uses; when `False` (Nano), both pathways share the
    same `k_und`.
  - `rope_theta`: `5e6` (Nano) vs. `1e8` (Edge); `rope_axes_dim=(24,20,20)`
    for both.
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
- **Dual-pathway attention** (`vidax.models.cosmos3.dit_layers.Cosmos3PackedMoTAttention`):
  `und` tokens self-attend causally (GQA, KV-head-grouped); `gen` tokens do
  full (non-causal) attention over **both** `und` and `gen` keys/values —
  one-directional information flow, `und -> gen` only, `und` never reads
  `gen`. The large `gen` pathway (tens of thousands of video-patch tokens at
  real resolutions) uses an *additive* bias (not a boolean mask) for its
  padding exclusion specifically so it can still take the Pallas
  flash-attention path — `vidax.core.attention.dot_product_attention` was
  extended to carry a `bias` through the single-device flash path (it
  previously forced the slow XLA-materializing fallback whenever *either*
  `bias` or `mask` was given).
- **Interleaved 3D mRoPE** (`vidax.models.cosmos3.mrope`): a genuinely
  different scheme from both Wan's (interleaved-pair rotation, unevenly-
  split block axes) and Cosmos-Predict2.5's (rotate-half, concatenated-block
  axes) RoPE. A single shared `inv_freq` table (over the full `head_dim // 2`
  channels) is evaluated at all three (T, H, W) axes' position ids, then the
  first `min(rope_axes_dim[1], rope_axes_dim[2]) * 3` channels are
  rearranged into repeating `(T, H, W)` triples — each triple shares the
  *same* underlying frequency, just evaluated at each axis's own position —
  with the remaining tail channels staying purely T-indexed. Verified via
  the relative-position invariant (`q_i . k_j` after rotation depends only
  on `i - j`) before any real-weight run.
- **No AdaLN modulation anywhere** (unlike every other DiT in this repo).
  The diffusion timestep is injected exactly **once**, additively, directly
  into the noisy vision tokens themselves (matching the reference's
  `_apply_timestep_embeds_to_noisy_tokens`: a sinusoidal embedding +
  2-layer MLP, added into each noisy frame's patch tokens, masked to
  noisy — not conditioned/clean — frames only), then flows through
  ordinary pre-norm transformer blocks with no further timestep-conditioned
  scale/shift. Same `timestep_scale=0.001` DiT-internal rescale mechanism as
  Cosmos-Predict2.5's `MinimalV1LVGDiT`.
- **No preconditioning wrapper — the DiT's raw output is the velocity,
  used directly**, matching `refs/diffusers-cosmos3/pipeline_cosmos3_omni.py`'s
  own denoising loop: the raw noisy latent is passed to the DiT unscaled,
  its timestep is `sigma * num_train_timesteps` (the DiT's own
  `timestep_scale` divides this back down internally), and the DiT's raw
  output (after CFG combination) is fed straight into the scheduler's
  `x0 = sample - sigma_t * model_output` — no `c_in`/`c_skip`/`c_out`
  reconstruction anywhere.
- **VAE:** both Nano's and Edge's VAE is Wan2.2-TI2V-5B's own VAE, confirmed
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
- **Text tokenization/templating** (`generate_cosmos3.py`'s
  `tokenize_prompt`): matches the reference's own `Cosmos3OmniPipeline.
  tokenize_prompt` — a fixed system prompt, resolution ("This video is of
  HxW resolution.") and duration ("The video is N seconds long and is of F
  FPS.") metadata sentences appended to the prompt (inverse-phrased for the
  negative prompt), chat-templated, with `<|vision_start|>` + eos appended
  as the generation-start markers. No separate text encoder is invoked —
  token ids feed `embed_tokens` directly inside the shared transformer.
- **Sampler:** `vidax.schedulers.unipc.FlowUniPCMultistepScheduler`, same
  core predictor-corrector solver as Cosmos-Predict2.5, with a Karras-sigma
  schedule path added (`use_karras_sigmas=True`) matching `diffusers`' own
  `UniPCMultistepScheduler._convert_to_karras` (an elucidating-diffusion-
  models power-law ramp) followed by its `sigma / (sigma + 1)` remap into
  this scheduler's own `alpha = 1 - sigma` convention — a genuinely
  different curve from Cosmos-Predict2.5's linear-sigma/`shift`-warp
  schedule, not just a different `shift` value. Both checkpoints'
  bundled `scheduler/scheduler_config.json` defaults to
  `use_karras_sigmas: true`, `sigma_min: 0.147`, `sigma_max: 200.0` — the
  real recipe for Nano, but not for Edge (see
  [Cosmos3-Edge](#cosmos3-edge-4b--model_size-edge)'s `--use_karras_sigmas`/
  `--shift` values).
