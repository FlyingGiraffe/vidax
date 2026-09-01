# HunyuanVideo (1.0) — Usage Guide

T2V only (I2V lives in a separate, un-cloned upstream repo and is out of
scope). One standalone TPU inference script lives in `examples/`:
`generate_hunyuan_video.py`, covering the single released checkpoint
variant (`tencent/HunyuanVideo`'s `hunyuan-video-t2v-720p`, the
`"HYVideo-T/2-cfgdistill"` preset). `--model-resolution`/544p in the
reference is dead code for the default CLI path (only ever consulted when
`--dit-weight` is *not* given, which it always is in practice) — there is
no separate 544p checkpoint, just a different runtime `--height`/`--width`
on this one 720p-native checkpoint.

| Script | Model | Params | Task | Checkpoint |
| --- | --- | --- | --- | --- |
| `generate_hunyuan_video.py` | HunyuanVideo (1.0) | 13B DiT | T2V | `hunyuan-video-t2v-720p/` |

Conditioning requires two separate text towers on top of the DiT itself: a
Llama3-8B decoder-only text encoder (extracted from
`xtuner/llava-llama-3-8b-v1_1-transformers`'s `.language_model`, per the
reference's own recommendation — HunyuanMLLM was never publicly released)
and a CLIP-L pooled text vector (`openai/clip-vit-large-patch14`).

Requires the `torch` extra (to deserialize `.pt`/`.safetensors`
checkpoints) and the `text` extra (`transformers`, for the Llama/CLIP
tokenizers):

```bash
pip install -e ".[tpu,torch,text]"
```

---

## HunyuanVideo (1.0, 720p, T2V) — `generate_hunyuan_video.py`

`--checkpoint_dir` should point at `tencent/HunyuanVideo`'s downloaded root
(containing `config.json`, `hunyuan-video-t2v-720p/{transformers,vae}/`);
`--text_encoder_dir` at the *extracted* Llama text-decoder tower (see
[Preparing the Llama text encoder](#preparing-the-llama-text-encoder)
below); `--clip_checkpoint_dir` at `openai/clip-vit-large-patch14`'s
downloaded root.

```bash
python examples/generate_hunyuan_video.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo" \
  --text_encoder_dir "./checkpoints/HunyuanVideo/text_encoder" \
  --clip_checkpoint_dir "./checkpoints/HunyuanVideo/clip-vit-large-patch14" \
  --prompt "A golden retriever running on a beach at sunset, cinematic, high detail" \
  --height 720 --width 1280 --num_frames 129 --num_steps 50 \
  --output_path "out/output_hunyuan_1_t2v.mp4"
```

### Preparing the Llama text encoder

Unlike HunyuanVideo-1.5's Qwen2.5-VL tower (downloaded pre-packaged),
HunyuanVideo 1.0's LLM text encoder must be **extracted** from the full
`xtuner/llava-llama-3-8b-v1_1-transformers` checkpoint — only the
`.language_model` sub-module (a plain `LlamaModel`, no vision tower) is
ever used:

```bash
python -c "
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
m = LlavaForConditionalGeneration.from_pretrained('xtuner/llava-llama-3-8b-v1_1-transformers', dtype=torch.float32, low_cpu_mem_usage=True)
p = AutoProcessor.from_pretrained('xtuner/llava-llama-3-8b-v1_1-transformers')
m.language_model.save_pretrained('./checkpoints/HunyuanVideo/text_encoder')
p.tokenizer.save_pretrained('./checkpoints/HunyuanVideo/text_encoder')
"
```

(the reference's own
`hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py` does the same
thing but unconditionally calls `.to(0)` — a GPU-only assumption; the
snippet above is the CPU-portable equivalent.) The full
`llava-llama-3-8b-v1_1-transformers` download (~17GB) can be deleted once
the extracted `text_encoder/` directory is confirmed to load — only the
extracted tower (~16GB, `LlamaModel`, 4096 hidden / 32 layers) is needed
at inference time.

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_dir` | *required* | `tencent/HunyuanVideo`'s downloaded root. |
| `--text_encoder_dir` | *required* | The extracted Llama text-decoder tower (see above). |
| `--clip_checkpoint_dir` | *required* | `openai/clip-vit-large-patch14`'s downloaded root. |
| `--model` | `HYVideo-T/2-cfgdistill` | Named hyperparameter preset (`configs.py`'s `DIT_CONFIGS`) — the released checkpoint is the `guidance_embed=True` cfgdistill variant. |
| `--prompt` | *required* | Text prompt. |
| `--negative_prompt` | `None` | Defaults to the reference's own `NEGATIVE_PROMPT` when `--guidance_scale != 1.0`, else empty. |
| `--height`/`--width` | `720`/`1280` | Output resolution in pixels (aligned to 16). |
| `--num_frames` | `129` | Must be `1 + 4k` (VAE temporal compression=4) — matches `config.py`'s `--video-length` default. |
| `--num_steps` | `50` | Flow-matching Euler sampling steps. |
| `--shift` | `7.0` | Flow-match schedule shift (`config.py`'s `--flow-shift` default). |
| `--guidance_scale` | `1.0` | Real classifier-free guidance. Default `1.0` (off) matches the reference's own `sample_video.py` default — this checkpoint's embedded/distilled guidance is the primary guidance mechanism. |
| `--embedded_guidance_scale` | `6.0` | Embedded/distilled guidance fed to `guidance_in` (`config.py`'s `--embedded-cfg-scale` default). |
| `--seed` | `0` | Initial noise seed. |
| `--dtype` | `bfloat16` | Compute dtype for the VAE/text encoders/DiT activations. |
| `--dit_dtype` | `bfloat16` | Cast target for the DiT's weights (checkpoint ships as float32/bf16-mixed). |
| `--fps` | `24` | Output video frame rate. |
| `--tensor_parallel_size` | every local device | Megatron-shards the DiT's double/single-stream blocks' Q/K/V/output/FFN Dense layers across this many chips. Must divide `heads_num` (24). Required in practice — the 13B DiT doesn't fit replicated on one TPU v4 chip. |
| `--vae_tile_latent_size` | reference default | Latent-space spatial tile size for the tiled VAE decode. Shrink (e.g. `8`) if VAE decode OOMs. |
| `--output_path` | `output.mp4` | Output video path. |

---

## Architecture notes

- **DiT (`vidax.models.hunyuan_video.hunyuan_video.dit.
  HunyuanVideoDiT`, shared blocks in `vidax.models.hunyuan_video.common.
  dit_layers`/`common.rope`):** the same dual-stream (`MMDoubleStreamBlock`
  ×20) + single-stream (`MMSingleStreamBlock` ×40) MMDiT family as
  HunyuanVideo-1.5, `hidden_size=3072`, `heads_num=24` (head_dim=128),
  `mlp_width_ratio=4`, `gelu_tanh` MLP, `qk_norm=True`, zero-init AdaLN
  modulation, RoPE `theta=256`/`rope_dim_list=(16,56,56)` (image tokens
  only) — every one of these building blocks is reused **unmodified** from
  the HunyuanVideo-1.5 port (`common/dit_layers.py`/`common/rope.py`), not
  reimplemented.
- **Real (non-degenerate) patchify.** Unlike 1.5's `patch_size=(1,1,1)`,
  1.0 uses `patch_size=(1,2,2)` — a real spatial downsample, implemented as
  a reshape-then-`Dense` (`_patchify`/`_unpatchify` in `dit.py`, duplicated
  from `hunyuan_video1_5/dit.py`'s identical functions rather than
  imported): mathematically exact for any `patch_size` because the
  reference's `Conv3d`(kernel==stride) has no overlap between patches.
- **Text conditioning is a single LLM + a separate pooled CLIP-L vector —
  not 1.5's multi-encoder token-concatenation stack.** `text_states` (the
  Llama tower's hidden states, `text_states_dim=4096`,
  `hidden_state_skip_layer=2`, `crop_start=95`-cropped per the reference's
  video chat-template convention) is refined through `txt_in`
  (`SingleTokenRefiner`, reused unmodified) and becomes the DiT's *entire*
  `txt` sequence — no byT5, no SigLIP, no `cond_type_embedding`, no
  token-stream reordering. A separate pooled CLIP-L vector
  (`text_states_dim_2=768`) feeds `vector_in` (`MLPEmbedder`:
  `Dense→SiLU→Dense`, no sinusoidal timestep step) into the AdaLN
  modulation vector `vec` alongside the timestep embedding
  (`vec = time_in(t) + vector_in(text_states_2) [+ guidance_in(guidance)]`,
  confirmed by reading `HYVideoDiffusionTransformer.forward` directly).
- **Text encoder — Llama3-8B decoder tower
  (`vidax.models.hunyuan_video.hunyuan_video.llama_text`):** a fresh,
  small port (not a reuse of `cosmos2_5.reason1.Qwen2TextModel` — that
  module hardcodes Qwen2's `q`/`k`/`v_proj` bias convention, which this
  Llama checkpoint doesn't have; see `llama_text.py`'s module docstring for
  the full reasoning), structurally mirroring `reason1.py`'s pattern
  (RMSNorm, SwiGLU MLP, GQA, rotate-half RoPE). Real config, confirmed
  against the extracted `xtuner/llava-llama-3-8b-v1_1-transformers`'s
  `.language_model`: `hidden_size=4096`, 32 layers, 32 query / 8 KV heads
  (head_dim=128), `intermediate_size=14336`, `rope_theta=500000`,
  `vocab_size=128320` (larger than stock Llama-3-8B's 128256 — xtuner added
  special image tokens), every Dense layer bias-free. See `docs/lessons/
  hunyuan_video_debugging.md` for the right-vs-left-padding pitfall
  verifying this tower caught.
- **Text encoder — CLIP-L pooled vector
  (`vidax.models.hunyuan_video.hunyuan_video.clip_text`):** a fresh,
  standard pre-LN CLIP text tower port (`hidden_size=768`, 12 layers, 12
  heads, `quick_gelu`, causal self-attention, `max_position_embeddings=77`)
  — no existing CLIP *text* tower port in vidax (`wan/wan2_1/clip_vision.py`
  is CLIP *vision*, a different tower). Pooled via the original CLIP
  tokenizer's `argmax(input_ids)` EOS-position rule.
- **VAE (`vidax.models.hunyuan_video.hunyuan_video.vae.
  HunyuanVideoVAE`, `"884-16c-hy"`):** diffusers' standard
  `AutoencoderKLCausal3D` — GroupNorm (32 groups, not RMSNorm), plain
  strided `CausalConv3d` down/upsample (no pixel-(un)shuffle scheme, unlike
  1.5's VAE), a single-head diffusers-style `Attention` mid-block with the
  same causal (query frame *i* attends to key frames ≤ *i*) mask as 1.5's
  VAE. Channel-last internally, matching this repo's convention. Real
  config (`vae/config.json`, not hardcoded): `block_out_channels=
  [128,256,512,512]`, `layers_per_block=2`, `latent_channels=16`,
  `scaling_factor=0.476986`, `time_compression_ratio=4`,
  `spatial_compression_ratio=8` (the reference ctor's own default — not a
  `config.json` key). No `shift_factor` (unlike 1.5's VAE, this one never
  applies one). `UpsampleCausal3D`'s frame-0 special case (the first latent
  frame is only ever spatially upsampled, never temporally duplicated) is
  ported exactly. Only spatial tiling is implemented (matching the
  reference, which never supports temporal tiling for this VAE either).
- **Guidance embedding.** The released checkpoint is the
  `"HYVideo-T/2-cfgdistill"` variant (`guidance_embed=True`, confirmed by
  the real checkpoint's own `guidance_in.*` keys) — `guidance_in` (a
  `TimestepEmbedder`-shaped MLP) is always exercised, fed
  `embedded_guidance_scale * 1000`. Real classifier-free guidance
  (`--guidance_scale`) is still supported on top of this (the reference's
  own `sample_video.py` defaults it to `1.0`, i.e. off, relying on the
  embedded guidance alone).
- **Checkpoint format.** `hunyuan-video-t2v-720p/transformers/
  mp_rank_00_model_states.pt` is a raw DeepSpeed-style `.pt` (this model
  predates broad safetensors adoption) wrapping the actual state_dict one
  level down under a `"module"` key — unwrapped by
  `map_hunyuan_video_dit_keys` itself (not the generic loader, since this
  nesting is specific to this one checkpoint's save format).
- **Checkpoint translator (`vidax.translator.mappings.hunyuan_video`):**
  the DiT's one real structural difference from 1.5's mapper: 1.0's
  checkpoint stores **fused** QKV Linears (`img_attn_qkv`/`txt_attn_qkv`/
  `single_blocks.N.linear1`, 856 leaves total across 20 double blocks / 40
  single blocks / 2 token-refiner blocks, plus `guidance_in.*`) — split
  into contiguous per-projection chunks (`_split_fused_linear`) before
  writing to the Flax param tree, unlike 1.5's mapper, which copies each
  already-split Q/K/V weight straight across.
- **Scheduler:** `vidax.schedulers.flow_match.RectifiedFlowScheduler`,
  reused directly (same as 1.5) — the reference's own example scripts
  always pass `--flow-reverse` (sampling from `t=1 -> t=0`), matching this
  scheduler's own (only) convention.
