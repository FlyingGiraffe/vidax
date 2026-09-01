# HunyuanVideo-1.5 — Usage Guide

One standalone TPU inference script lives in `examples/`:
`generate_hunyuan_video1_5.py` — a single script covering both T2V and
I2V for a given `--resolution`/task variant (pass `--image_path` for I2V,
omit it for T2V; both share the same DiT class and checkpoint-shape
family for a given `--resolution`/task, differing only in which
conditioning tensors are real vs. zero — see [Architecture
notes](#architecture-notes)). Covers the 4 core (non-distilled,
non-sparse-attention, non-super-resolution) checkpoint variants:
480p/720p × T2V/I2V.

| Script | Model | Params | Task | Checkpoint variant |
| --- | --- | --- | --- | --- |
| `generate_hunyuan_video1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | T2V | `transformer/480p_t2v/` |
| `generate_hunyuan_video1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | I2V | `transformer/480p_i2v/` |
| `generate_hunyuan_video1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | T2V | `transformer/720p_t2v/` |
| `generate_hunyuan_video1_5.py` | HunyuanVideo-1.5 | 8.3B DiT | I2V | `transformer/720p_i2v/` |

Conditioning requires three separate text/vision towers on top of the DiT
itself: a Qwen2.5-VL-7B-Instruct text-only tower (the MLLM), a byT5-small
(Glyph-SDXL-v2) glyph/color encoder, and (I2V only) a SigLIP vision
encoder. All weights are pulled from `tencent/HunyuanVideo-1.5`'s own
`transformer/`/`vae`/`text_encoder/llm` subfolders except the byT5/SigLIP
towers, which come from separate upstream repos (see
`checkpoints-download.md` in the reference source for the exact `hf
download`/`modelscope download` commands — SigLIP's source repo,
`black-forest-labs/FLUX.1-Redux-dev`, is gated and needs a Hugging Face
access request approved first).

Requires the `torch` extra (to deserialize `.safetensors`/`.pt`
checkpoints), the `text` extra (`transformers`, for the Qwen2.5-VL and
byT5 tokenizers), and the `i2v` extra (`pillow`, for I2V's conditioning
image):

```bash
pip install -e ".[tpu,torch,text,i2v]"
```

---

## HunyuanVideo-1.5 (480p/720p, T2V/I2V) — `generate_hunyuan_video1_5.py`

`--checkpoint_dir` should point at `tencent/HunyuanVideo-1.5`'s downloaded
root (containing `transformer/`, `vae/`, `text_encoder/{llm,byt5-small,
Glyph-SDXL-v2}/`); `--siglip_checkpoint_dir` (I2V only) at
`black-forest-labs/FLUX.1-Redux-dev`'s downloaded root (containing
`image_encoder/`, `feature_extractor/`).

Every architecture hyperparameter (DiT `hidden_size`/block depths/RoPE
dims, VAE `block_out_channels`/compression factors, SigLIP dims) is read
directly from each checkpoint's own `config.json` (`vidax.models.
hunyuan_video.hunyuan_video1_5.configs`), never hardcoded.

### Text-to-video

```bash
python examples/generate_hunyuan_video1_5.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo-1.5" \
  --resolution 480p \
  --prompt "A golden retriever running on a beach at sunset, cinematic, high detail" \
  --height 480 --width 832 --num_frames 121 --num_steps 50 \
  --output_path "out/output_hunyuan_1_5_t2v.mp4"
```

### Image-to-video

Omit `--height`/`--width` for I2V and the output resolution is derived
from the conditioning image's own aspect ratio (see [CLI
reference](#cli-reference)'s `--max_area`) — a portrait image produces a
portrait video, not a squished landscape one.

```bash
python examples/generate_hunyuan_video1_5.py \
  --checkpoint_dir "./checkpoints/HunyuanVideo-1.5" \
  --siglip_checkpoint_dir "./checkpoints/FLUX.1-Redux-dev" \
  --resolution 480p \
  --image_path "./assets/dog.jpg" \
  --prompt "The dog starts running toward the camera" \
  --num_frames 121 --num_steps 50 \
  --output_path "out/output_hunyuan_1_5_i2v.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint_dir` | *required* | `tencent/HunyuanVideo-1.5`'s downloaded root. |
| `--siglip_checkpoint_dir` | `None` | `black-forest-labs/FLUX.1-Redux-dev`'s downloaded root. Required for I2V. |
| `--resolution` | `480p` | `480p` or `720p` — selects the checkpoint variant and the default `--shift`. |
| `--prompt` | *required* | Text prompt. |
| `--negative_prompt` | `""` | Negative prompt for classifier-free guidance. |
| `--image_path` | `None` | Conditioning image, for I2V. Omit for T2V. |
| `--height`/`--width` | `None` | Output resolution in pixels (must be divisible by the VAE's `ffactor_spatial`, 16). T2V: defaults to `--resolution`'s own default (480p: 480×832, 720p: 720×1280). I2V: if *both* are omitted, derived from the conditioning image's own aspect ratio instead (see `--max_area`) — giving both explicitly overrides that. |
| `--max_area` | `None` | I2V only, when `--height`/`--width` aren't both given: target pixel area combined with the conditioning image's aspect ratio to pick (height, width), matching `generate_wan2_1_i2v.py`'s convention. Defaults to `--resolution`'s own default area. |
| `--num_frames` | `121` | Output frame count; the VAE's causal temporal compression (`ffactor_temporal=4`) works out to `1 + 4k` latent-frame-aligned counts. |
| `--num_steps` | `50` | Flow-matching Euler sampling steps. |
| `--shift` | `None` | Flow-match schedule shift. Defaults to the real per-(resolution, task) value (480p: 5.0 both tasks; 720p: 9.0 T2V, 7.0 I2V). |
| `--guidance_scale` | `6.0` | Classifier-free guidance: `v = uncond + guidance_scale * (cond - uncond)`. These checkpoints have no embedded/distilled guidance path (`guidance_embed=False`), so this is real CFG, not a no-op knob. |
| `--seed` | `0` | Initial noise seed. |
| `--dtype` | `bfloat16` | Compute dtype for the VAE, text/vision encoders, and DiT activations. |
| `--dit_dtype` | `bfloat16` | Cast target for the DiT's weights (checkpoints ship as float32). |
| `--fps` | `24` | Output video frame rate. |
| `--tensor_parallel_size` | every local device | Megatron-shards the DiT's double/single-stream blocks' Q/K/V/output/FFN Dense layers across this many chips (`vidax.core.sharding.shard_wan_params`); the rest (VAE, Qwen, byT5, SigLIP) is simply replicated across the same mesh. Must divide `heads_num` (16). Required in practice — the 8.3B DiT doesn't fit replicated on one TPU v4 chip. |
| `--vae_tile_latent_size` | reference default (16) | Latent-space spatial tile size for the tiled VAE decode (pixel tile = this × `ffactor_spatial`). Shrink (e.g. `8`) if VAE decode OOMs — more likely at `--tensor_parallel_size > 1`, where the other (replicated) components leave less per-chip headroom. |
| `--output_path` | `output.mp4` | Output video path. |

---

## Architecture notes

- **DiT (`vidax.models.hunyuan_video.hunyuan_video1_5.dit.
  HunyuanVideo15DiT`, shared blocks in `vidax.models.hunyuan_video.common.
  dit_layers`):** a dual-stream + single-stream MMDiT, structurally the
  same family as HunyuanVideo 1.0/Flux (`MMDoubleStreamBlock`/
  `MMSingleStreamBlock`), but the *real* checkpoint config differs
  substantially from the reference ctor's own defaults — confirmed
  directly against all 4 downloaded `config.json`s, not assumed:
  `hidden_size=2048`, `heads_num=16` (head_dim=128), **54 double-stream
  blocks and 0 single-stream blocks** (not the reference's 20/40 default
  split), `patch_size=[1,1,1]` (no additional DiT-side patchify beyond the
  VAE's own compression — `img_in` is architecturally a per-voxel Dense,
  not a real spatial-downsampling conv). `qk_norm=True` (per-head
  RMSNorm), zero-init AdaLN modulation throughout.
- **Text/glyph/vision conditioning is token-concatenation into the joint
  self-attention sequence, not cross-attention.** The final `txt` stream
  fed to the double-stream blocks is `[byT5 glyph tokens | Qwen2.5-VL MLLM
  tokens | SigLIP vision tokens]`, each region tagged by a small learned
  `cond_type_embedding` and merged via a stable, mask-based reorder
  (`dit.py`'s `_reorder_tokens`, a JAX `lexsort`-based reproduction of the
  reference's per-source `cat([valid], [invalid])` grouping) so that
  padded/invalid tokens always sort to the end. I2V and T2V share this
  exact code path — the only difference is whether the SigLIP tokens and
  the reference-frame channel-concat block are real or all-zero.
- **RoPE (`vidax.models.hunyuan_video.common.rope`):** 3D axial RoPE,
  `theta=256`, `rope_dim_list=(16, 56, 56)` (T/H/W split of the 128-dim
  head). The reference's own `rotate_half` helper is misleadingly named —
  it's actually the same **interleaved-pair** rotation convention
  `vidax.core.rope3d.apply_rope3d` already implements for Wan (confirmed
  by reading the actual pairing math, not the function name), so that
  function is reused directly; only the per-axis frequency construction
  (a different T/H/W split, different `theta`) is HunyuanVideo-1.5-
  specific. Applied to image tokens only — text/glyph/vision tokens carry
  no positional rotation.
- **Attention masking:** the reference's real `attn_mode="flash"` path
  (what every one of the 4 core checkpoints ships with) masks only
  **key** positions (via `flash_attn_no_pad`'s variable-length packing),
  not a symmetric query+key mask like its `torch`-SDPA fallback branch —
  this port implements the key-only-masked form (`masked_self_attention`
  in `common/dit_layers.py`), proven equivalent at every position that
  feeds the final image output (see that module's docstring for the
  argument). On TPU this calls `vidax.core.attention`'s Pallas
  flash-attention kernel directly (O(sequence length) memory) rather than
  materializing the full attention matrix — required at real video-token
  sequence lengths; the naive dense form OOMs a single TPU v4 chip well
  before 480p resolution.
- **Text encoder — Qwen2.5-VL-7B-Instruct MLLM
  (`vidax.models.hunyuan_video.hunyuan_video1_5.qwen_text`):** reuses
  `vidax.models.cosmos2_5.reason1.Qwen2TextModel` **unmodified** — it's
  architecturally the exact same Qwen2.5-VL-7B-Instruct text-only decoder
  tower Cosmos-Predict2.5-2B's Reason1 text encoder already ports,
  confirmed by translating the real `tencent/HunyuanVideo-1.5` `text_
  encoder/llm` checkpoint with Cosmos's own `map_reason1_text_encoder_
  keys` translator (zero code changes, exact param-tree shape match,
  7,070,619,136 params). Only the embedding-extraction glue
  (`HunyuanVideoMLLMTokenizer`/`extract_hunyuan_mllm_embeddings`) is
  HunyuanVideo-specific: a fixed system-prompt chat template (different
  for image-caption vs. video-prompt data types), `crop_start`
  auto-detection (position right after the `<|im_start|>user\n` marker),
  and `hidden_states[-3]` selection (`hidden_state_skip_layer=2`, no final
  norm reapplied) — different from Cosmos's own usage of the same tower
  (which mean/std-normalizes and concatenates all 28 layers instead).
- **Glyph/color encoder — byT5-small
  (`vidax.models.hunyuan_video.hunyuan_video1_5.byt5`):** the Glyph-SDXL-
  v2 fine-tune of `google/byt5-small`'s encoder is architecturally
  identical to `vidax.models.ltx_video.t5.T5Encoder` (bidirectional
  relative-position bucketing, gated-GELU FFN, one shared
  relative-position-bias table) — reused directly via a `byt5_encoder()`
  factory rather than re-ported, parameterized from the real checkpoint's
  `config.json` (`d_model=1472`, 6 heads, `d_kv=64`, `d_ff=3584`, 12
  layers). `vocab_size` is **not** the base checkpoint's 384 — Glyph-SDXL-
  v2 adds per-language color/font special tokens, so it must be read off
  the real checkpoint's embedding weight shape (1510 for the released
  checkpoint), never hardcoded. Note:
  `Glyph-SDXL-v2/checkpoints/byt5_mapper.pt` (despite the name) is
  *unrelated* to the DiT's own `byt5_in`/`ByT5Mapper` — checked directly,
  it's a different, Glyph-SDXL-v2-internal 4-block tower for that
  project's own SDXL LoRA pipeline. The real `byt5_in.*` weights the DiT
  uses live inside the main DiT checkpoint itself.
- **Vision encoder — SigLIP
  (`vidax.models.hunyuan_video.hunyuan_video1_5.siglip`):** a standard
  pre-LN ViT (`SiglipVisionEncoder`, patch embed + learned absolute
  position embed + N pre-LN transformer blocks + final LayerNorm, no CLS
  token, no pooling head — only `last_hidden_state` is ever consumed),
  ported by reading the installed `transformers` package's own
  `modeling_siglip.py` source directly (the real checkpoint,
  `black-forest-labs/FLUX.1-Redux-dev`'s `image_encoder`, is gated).
  Real config: `hidden_size=1152`, `patch_size=14`, `image_size=384`
  (729 patch tokens), `num_hidden_layers=27`, `num_attention_heads=16`,
  `hidden_act="gelu_pytorch_tanh"`.
- **I2V conditioning:** dual mechanism, both active simultaneously (T2V
  zeroes both, same DiT code path — see above). (1) The reference frame's
  VAE-encoded latent + a binary mask are channel-concatenated into the
  patchify input (`concat_condition=True` doubles+1's `PatchEmbed`'s
  input channels: `in_channels*2+1` — a Wan2.1-I2V-style scheme). (2)
  SigLIP's 729 patch tokens (per conditioning frame) are projected
  (`VisionProjection`: LN→Linear→GELU→Linear→LN) and merged into the
  joint text/glyph/vision token stream described above.
- **VAE (`vidax.models.hunyuan_video.hunyuan_video1_5.vae.
  HunyuanVideo15VAE`):** causal 3D-conv KL-VAE, channel-last internally
  (matching this repo's `wan/common/vae_layers.py` convention). 16×
  spatial / 4× temporal compression, `RMS_norm` (not GroupNorm)
  throughout, `CausalConv3d` with **replicate** (not zero) padding — H/W
  symmetric by `kernel//2`, T causally front-padded by `kernel-1`. The
  `Downsample`/`Upsample` blocks use a distinctive
  pixel-(un)shuffle-with-channel-group-averaged-shortcut scheme (ported
  precisely from the reference's einops `rearrange` patterns, translated
  to explicit reshape/transpose in channel-last layout) rather than a
  plain strided conv. One bottleneck `AttnBlock` per encoder/decoder, with
  a causal (query frame *i* attends to key frames ≤ *i*) full-spatial
  attention mask. Only spatial tiling is ported (matching
  `AutoencoderKLConv3D.spatial_tiled_decode` exactly — latent-space H/W
  tiling, `blend_h`/`blend_v` linear cross-fade of overlaps); temporal
  tiling isn't supported by the reference VAE at all, so isn't ported here
  either. The example script's
  `spatial_tiled_vae_decode` additionally decodes each
  tile through a per-decoder-block staged pipeline (`decode_stage_level_
  block`/`decode_stage_level_upsample`, each its own `jax.jit` call) —
  needed because a single fused decode (even of one small tile) doesn't
  free one stage's temporaries before the next, the same class of issue
  `docs/lessons/ltx2_5_debugging.md` documents for LTX-2.5's DiT (see
  `docs/lessons/hunyuan_video1_5_debugging.md` for the specific OOM
  numbers this fixed at the reference's real 121-frame/480p default).
- **Tensor parallelism:** Megatron-style 1D TP for `double_blocks`/
  `single_blocks`' Q/K/V/output/FFN Dense layers is plain GSPMD
  auto-partitioning (`vidax.core.sharding.shard_wan_params`) — no
  `shard_map` needed for ordinary Dense/reshape/norm ops. The one
  exception is `masked_self_attention`'s Pallas flash-attention kernel:
  Mosaic custom calls can't be auto-partitioned, so it's wrapped in
  `shard_map` whenever `mesh` is given — **for every caller**, including
  `SingleTokenRefiner` (whose own weights are never TP-sharded, so it
  gets a fully-replicated `shard_map` variant,
  `_flash_attention_tpu_segment_masked_replicated`, rather than the
  head-sharded one) — omitting this for even one caller crashes the whole
  program (`NotImplementedError: Mosaic kernels cannot be automatically
  partitioned`), since the requirement is "any Pallas call in a
  multi-device-partitioned program", not "any Pallas call whose own
  operands happen to be split".
- **Scheduler:** `vidax.schedulers.flow_match.RectifiedFlowScheduler`,
  reused directly — the reference's `FlowMatchDiscreteScheduler` uses the
  same SD3-style `sigma' = shift*sigma/(1+(shift-1)*sigma)` warp and the
  same `sigmas * num_train_timesteps`-scaled model conditioning; the two
  schedulers' Euler update rules are algebraically the same once the sign
  convention (reversed 1→0 sigma schedule) is accounted for.
- **Checkpoint translator (`vidax.translator.mappings.
  hunyuan_video1_5`):** 5 weight-bearing components — DiT (8,326,608,160
  params), VAE (1,260,634,115), byT5 (219,314,944), SigLIP (412,987,248),
  plus Qwen2.5-VL, which reuses Cosmos's existing translator unmodified.
