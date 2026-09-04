# Cosmos-Predict2.5 — Usage Guide

One standalone TPU inference script currently lives in `examples/`:
`generate_cosmos2_5.py`, for both released sizes (2B and 14B, selected via
`--model_size`) — a single script covering all three tasks the checkpoints
support (text2world, image2world, video2world), selected by which
conditioning flag you pass. Both sizes share one architecture
(`vidax.models.cosmos2_5.dit.CosmosDiT`, config-driven — see
`vidax.models.cosmos2_5.configs`) and only differ in `dim`/`ffn_dim`/
`num_heads`/`num_layers`. It shares the same building blocks as the Wan
scripts (`vidax.core`, `vidax.schedulers`, `vidax.translator`) but differs in
checkpoint layout, sampler (UniPC instead of Euler), and text encoder (a
7B-parameter VLM instead of T5) — see
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the shared
engineering background, and this doc's [Architecture notes](#architecture-notes)
section below for what's specific to Cosmos.

| Script | Model | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- | --- |
| `generate_cosmos2_5.py` | Cosmos-Predict2.5 | 2B | Text2World, Image2World, Video2World | `Cosmos-Predict2.5-2B/base/pre-trained` |
| `generate_cosmos2_5.py` | Cosmos-Predict2.5 | 14B | Text2World, Image2World, Video2World | `Cosmos-Predict2.5-14B/base/pre-trained` |

Uses `torch` (to deserialize the `.pt` DiT/VAE checkpoints), `transformers`
(the Reason1/Qwen2.5-VL-7B tokenizer), and `pillow` (image2world/video2world
conditioning frame(s)) — all installed by default. On a Cloud TPU VM also
add the `tpu` extra:

```bash
pip install -e ".[tpu]"    # or just: pip install -e .
```

---

## Cosmos-Predict2.5 (2B / 14B) — `generate_cosmos2_5.py`

Checkpoints come from three separate repos: the DiT and VAE from
[nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
or [nvidia/Cosmos-Predict2.5-14B](https://huggingface.co/nvidia/Cosmos-Predict2.5-14B)
(the DiT as a flat PyTorch state_dict, `base/pre-trained/<uuid>/model_ema_bf16.pt`
for 2B, `base/pre-trained/<uuid>_ema_bf16.pt` for 14B; the VAE — Wan2.1's
causal VAE, reused verbatim by Cosmos-Predict2.5, see
[Architecture notes](#architecture-notes) — as `tokenizer.pth`, shipped only
in the 2B repo and shared by both sizes, since neither the DiT architecture
nor the VAE depend on which DiT size you're running), and the text encoder
(shared by both sizes too) from
[nvidia/Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
(a standard HuggingFace-format repo: sharded
`model-NNNNN-of-NNNNN.safetensors` + a `model.safetensors.index.json`
manifest, plus `tokenizer.json`/`tokenizer_config.json`/`chat_template.json`
— pass the `.index.json` manifest's path as `--reason1_checkpoint_path`;
`--tokenizer_path` then defaults to that same directory). Pass
`--model_size 2B` (the default) or `--model_size 14B` to match whichever
`--dit_checkpoint_path` you point at — the two configs only differ in
`dim`/`ffn_dim`/`num_heads`/`num_layers` (see
`vidax.models.cosmos2_5.configs`); every other flag/behavior below applies
identically to both.

Unlike Wan2.1's i2v (a separate 14B model, CLIP cross-attention + an extra
concatenated channel) but *like* Wan2.2 TI2V-5B's i2v, image2world/
video2world conditioning here works by substituting the known frames'
latents back into `x` between sampling steps — except Cosmos additionally
tells the DiT which frames are conditioning via a concatenated mask channel
*and* a tiny (not exactly zero) per-frame timestep for those frames, not
frame substitution alone. See [Architecture notes](#architecture-notes) for
the full mechanism, and the script's module docstring for one place this
implementation knowingly simplifies vs. the reference (the UniPC corrector's
internal history isn't separately patched at conditioning-frame positions,
only `x` itself is — likely a small effect, worth revisiting if generated
output looks off specifically near conditioning frames).

### Text2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_t2v.mp4"
```

`--shift` (noise-schedule shift, default 5.0), `--solver_order` (UniPC order,
default 2), and classifier-free guidance (`--guide_scale`, default 7.0, and
`--negative_prompt`) match the reference pipeline's own inference defaults.
Unlike the Wan scripts (Euler sampling, ~50 steps), this uses
`FlowUniPCMultistepScheduler` — a higher-order predictor-corrector solver
that reaches comparable quality in far fewer steps (35 vs. Wan's 50), at the
cost of a small rolling history of previous model outputs; see
`vidax.schedulers.unipc`'s module docstring for the full mechanism.

### Image2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --image_path "./examples/assets/cat.jpg" \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
  --tensor_parallel_size 4 \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_i2v.mp4"
```

The conditioning image is resized/center-cropped (preserving aspect ratio)
to the largest resolution under `--max_area` divisible by the VAE's spatial
stride × the DiT's patch size, then VAE-encoded into a single conditioning
latent frame — output resolution is derived from the image, ignoring
`--height`/`--width`.

### Video2World

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --video_path "./checkpoints/Cosmos-Predict2.5-2B/assets/example_clip.mp4" \
  --num_conditional_latent_frames 2 \
  --prompt "The scene continues, the camera slowly pans right" \
  --num_steps 35 \
  --output_path "out/output_cosmos2_5_v2v.mp4"
```

Same resolution-derivation as image2world, but conditions on the input
video's first `1 + 4 * (--num_conditional_latent_frames - 1)` pixel frames
(1 frame for 1 conditioning latent frame, 5 for 2) — matching the reference's
own training distribution over {0 (=t2v), 1, 2} conditioning latent frames.

### Tensor parallelism, sequence parallelism, batching, and dtype

```bash
python examples/generate_cosmos2_5.py \
  --dit_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/308eb96c-c4c0-4a06-9cc1-103a43beff28/model_ema_bf16.pt" \
  --vae_checkpoint_path "./checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth" \
  --reason1_checkpoint_path "./checkpoints/Cosmos-Reason-1-7B/model.safetensors.index.json" \
  --prompt "A red panda in the snow" "A dog running on a beach" \
  --tensor_parallel_size 4 \
  --dtype bfloat16 \
  --num_steps 35 --height 704 --width 1280 --num_frames 93 \
  --output_path "out/output_cosmos2_5.mp4"
```

Unlike the plain single-checkpoint version of this script from before
tensor/sequence parallelism existed, `--tensor_parallel_size` (default `1`)
now Megatron-shards both the DiT's attention heads/FFN channels *and*
Reason1's (its 7B params are by far the largest of the three checkpoints,
so this is where TP matters most for fitting on fewer chips — see
[Architecture notes](#architecture-notes)). `--sequence_parallel_size`
(default `1`) shards the DiT's token sequence itself (DeepSpeed-Ulysses),
independent of `--tensor_parallel_size`'s weight-sharding — the two compose
freely (see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s
"Combining with Megatron TP"), and `--sequence_parallel_size` alone is what
matters for pushing to much higher resolution/frame counts where self-
attention activation memory becomes the bottleneck rather than weight
memory — see `vidax.models.cosmos2_5.dit.CosmosDiT`'s module
docstring, ported directly from Wan's DiTs (`--sequence_parallel_size` only
affects the DiT; Reason1 always uses plain Megatron TP via
`--tensor_parallel_size`, its 512-token sequence being far too short for
sequence parallelism to matter). Passing `num_devices // tensor_parallel_size
// sequence_parallel_size` prompts (instead of one, broadcast) gives one
video per prompt instead of that many independent samples of the same
prompt (the conditioning image/video, if any, is shared across every
replica — only the text prompt varies).

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--model_size` | `2B` | `2B` \| `14B`. Selects which of `vidax.models.cosmos2_5.configs`' two presets to build `CosmosDiT` with — must match `--dit_checkpoint_path`'s actual size (the checkpoint carries no size metadata of its own to check against). |
| `--dit_checkpoint_path` | *required* | DiT `.pt` checkpoint (`model_ema_bf16.pt`/`model_ema_fp32.pt` — the flat, EMA-only state_dict; **not** the raw `model.pt` training checkpoint, which is a much larger PyTorch distributed-checkpoint (DCP) shard set). |
| `--vae_checkpoint_path` | *required* | `tokenizer.pth` — Wan2.1's own VAE checkpoint format, loaded via `model_type="wan2.1_vae"` (see [Architecture notes](#architecture-notes)). |
| `--reason1_checkpoint_path` | *required* | Path to Reason1's `model.safetensors.index.json` (a separate download, shared by both DiT sizes — see this section's intro). |
| `--tokenizer_path` | directory of `--reason1_checkpoint_path` | HuggingFace tokenizer id/path for Reason1's chat-template + BPE tokenization. The released repo bundles the tokenizer files alongside the model shards, so the default avoids a network call in the common case. |
| `--image_path` | `None` | Conditioning image, for image2world. Mutually exclusive with `--video_path`. When given, output resolution is derived from the image + `--max_area` (`--height`/`--width` ignored). |
| `--video_path` | `None` | Conditioning video, for video2world. Mutually exclusive with `--image_path`. Same resolution-derivation as `--image_path`. |
| `--num_conditional_latent_frames` | `1` | `--video_path` only: 1 or 2 conditioning latent frames (forced to 1 for `--image_path`). |
| `--max_area` | `704*1280` | image2world/video2world only: target output pixel area; actual (height, width) derived via `best_output_size` (ported from the reference, same algorithm the Wan scripts use). |
| `--prompt` | *required*, 1+ values | One prompt (broadcast to every data-parallel replica) or exactly `num_devices // tensor_parallel_size` prompts, one per replica. |
| `--negative_prompt` | vidax's own quality-negative-prompt | Negative prompt for CFG (the reference's own default is a long boilerplate string; vidax uses a shorter equivalent — see the script's module-level `DEFAULT_NEGATIVE_PROMPT`). |
| `--guide_scale` | `7.0` | CFG scale: `velocity = uncond + guide_scale * (cond - uncond)`. Matches the reference's default. |
| `--tensor_parallel_size` | `1` | Devices to Megatron-shard the DiT's attention heads/FFN channels *and* Reason1's weights across (see [hardware doc](../hardware_and_sharding.md)). Must divide `num_devices`, `CosmosDiT.num_heads` (16 for 2B, 40 for 14B), and `Qwen2TextModel.num_key_value_heads` (4, GQA — the binding constraint if Reason1 is meaningfully sharded, so `tp` in `{1,2,4}` in practice). |
| `--sequence_parallel_size` | `1` | Devices to shard the DiT's token sequence itself across (DeepSpeed-Ulysses), independent of `--tensor_parallel_size`'s weight-sharding. Not needed at 2B scale/typical resolutions; more likely to matter at 14B or at much higher resolution/frame counts — see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s "Combining with Megatron TP". Only affects the DiT — Reason1 always uses Megatron TP (`--tensor_parallel_size`) regardless. Requires the latent frame count to divide evenly by this value. |
| `--dtype` | `bfloat16` | `float32` \| `float16` \| `bfloat16`. `float16` will fail at runtime — TPU's XLA backend doesn't implement `float16` matmuls. |
| `--offload_dit_weights` | off | Keep the DiT's per-block weights host-resident, offloading `--offload_chunk_size` blocks' worth into HBM at a time during the sampling loop, instead of the whole DiT staying HBM-resident for the entire script. Needed for 14B to reach the reference's full 93-frame default on this repo's 4-chip machine (without it, reduced to 45 frames — see [`docs/benchmarking.md`](../benchmarking.md)). Composes with `--sequence_parallel_size`. See [`docs/weight_offloading.md`](../weight_offloading.md). |
| `--offload_chunk_size` | `1` | Number of consecutive DiT blocks grouped into one offloaded HBM buffer when `--offload_dit_weights` is set. Must divide `num_layers` (28 for 2B, 36 for 14B). |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `35` | UniPC sampling steps. Reference default for the 2B base checkpoint. |
| `--solver_order` | `2` | UniPC solver order. Reference default. |
| `--shift` | `5.0` | Flow-matching noise-schedule shift. Matches the reference's default. |
| `--height` | `704` | Output video height. Ignored if `--image_path`/`--video_path` is given. Must be divisible by 16 (VAE's 8x spatial compression × the DiT's `patch_size=(1,2,2)`, another 2x). |
| `--width` | `1280` | Output video width. Same ignore/divisibility rules as `--height`. |
| `--num_frames` | `93` | Output frame count. The reference trains primarily around 93-frame (~5.8s @ 16fps) clips at 720p. |
| `--fps` | `16` | Output video frame rate. Matches the reference's training/inference fps. |
| `--output_path` | `output_cosmos2_5.mp4` | With multiple prompts, each video is saved as `<output_path>_<i>.mp4`. |

---

## Architecture notes

- **VAE:** Cosmos-Predict2.5's own tokenizer config wraps Wan2.1's causal VAE
  unchanged (confirmed in the reference source, not assumed) — same
  8x spatial / 4x-with-a-+1 causal temporal compression, same 16-channel
  latent space. `generate_cosmos2_5.py` imports
  `vidax.models.wan.wan2_1.vae.WanVAEDecoder`/`WanVAEEncoder` directly rather
  than duplicating them; there is no `vidax.models.cosmos2_5.vae` module.
- **DiT (`vidax.models.cosmos2_5.dit.CosmosDiT`):** config-driven, like
  Wan's DiTs — 28 transformer blocks / 2048-dim / 16 heads for 2B, 36 blocks
  / 5120-dim / 40 heads for 14B (both head_dim 128 — see
  `vidax.models.cosmos2_5.configs`), `patch_size=(1,2,2)` (no
  temporal compression at the DiT level — all of it happens in the VAE).
  Departs from Wan's DiT in several concrete ways: per-head (not per-tensor)
  QK-RMSNorm, rotate-half RoPE (not Wan's interleaved-pair convention, see
  `vidax.models.cosmos2_5.rope`), and AdaLN-LoRA modulation where each
  block's three sublayers (self-attn/cross-attn/MLP) get their own small
  modulation MLP plus a shared global correction term, rather than Wan's
  single 6-way-split-per-block vector. Supports both of Wan's parallelism
  strategies, ported directly: Megatron-style tensor parallelism (`mesh`
  alone, via `vidax.core.sharding.shard_wan_params` — its name table covers
  `CosmosDiT`'s Dense-layer names too) and DeepSpeed-Ulysses sequence
  parallelism (`sequence_parallel=True`, chunking the token sequence between
  blocks and reshuffling only for the duration of self-attention). The one
  Cosmos-specific wrinkle: sequence-parallel chunking has to land on a
  *frame* boundary (`t_p`, the latent frame count, divisible by the
  sequence-parallel size), not an arbitrary token offset, since the
  per-frame timestep/AdaLN-LoRA state (`emb`/`adaln_lora`, shape `(B, T,
  ...)`) is chunked separately, along the frame axis, and only broadcast to
  per-token *inside* each block — see `CosmosDiT.__call__`'s chunking
  comment. Reason1's own weights get Megatron-sharded too (see below), but
  Reason1 itself has no sequence-parallel path — its fixed 512-token length
  never needed one.
- **Image2world/video2world conditioning:** driven by two DiT inputs, both
  per-*frame* (`(B, T, ...)`, not per-sample): `condition_video_mask` (1 for
  frames that are given, concatenated as an extra input channel alongside
  the 16 VAE latent channels + a padding-mask channel) and `timesteps`
  itself (conditioning frames sit at a small fixed sigma, `CONDITIONAL_SIGMA
  = 0.0001` in the script, rather than the current sampling noise level).
  `generate_cosmos2_5.py`'s sampling loop additionally substitutes the known
  conditioning frames' VAE-encoded latents directly into `x` — both for the
  initial noise and after every UniPC step — matching Wan2.2 TI2V-5B's i2v
  precedent, with the one documented simplification described at the top of
  this doc and in the script's module docstring.
- **Text encoder (`vidax.models.cosmos2_5.reason1`):** Reason1 is a
  Reason1-finetuned Qwen2.5-VL-7B-Instruct; only its text-only decoder tower
  is ported (the vision encoder is never invoked for text prompts — image2world/
  video2world's conditioning frames go through the *VAE* encoder, not
  Reason1's vision tower). Conditioning isn't the raw final hidden state —
  the reference extracts **all 28 decoder layers'** hidden states, per-token
  mean/std-normalizes each, and concatenates them into a single
  `(B, 512, 28*3584=100352)` tensor, which the DiT down-projects once
  (`crossattn_proj`) before every block's cross-attention. See
  `compute_reason1_embeddings`'s docstring for the exact pipeline, including
  the resolved M-RoPE-for-text-only question (plain 1D RoPE turns out to be
  bit-exact for this call path, not an approximation — full reasoning in
  that module's docstring). At 7B params, easily the largest of the three
  checkpoints -- `--tensor_parallel_size` Megatron-shards its weights too
  (`vidax.core.sharding.shard_wan_params`'s name table covers
  `Qwen2Attention`'s/`Qwen2MLP`'s bare `q_proj`/`k_proj`/`v_proj`/`o_proj`/
  `gate_proj`/`up_proj`/`down_proj` names), needing no changes inside
  `reason1.py` itself: its attention always passes a causal `mask`, which
  already routes `vidax.core.attention.dot_product_attention` to the plain
  (GSPMD-auto-partitioned) fallback path regardless of `mesh` -- the same
  reason T5's own Megatron sharding never needed mesh-threading into its
  attention call either.
- **Sampler (`vidax.schedulers.unipc.FlowUniPCMultistepScheduler`):** a
  from-scratch JAX port of the reference's flow-matching UniPC multistep
  predictor-corrector solver, independently cross-checked against a
  hand-transcribed numpy reference implementation (~1e-4 relative error
  across `solver_order` in {1,2,3} and several step counts) and a toy
  constant-velocity ODE (recovered to float32 precision regardless of
  order/step count). See its module docstring for the JAX-specific state
  threading (`UniPCState`, since JAX prefers explicit immutable state over
  the reference's in-place-mutated scheduler object).
- **No preconditioning wrapper — the DiT's raw output is the velocity,
  used directly** (`generate_cosmos2_5.py`'s `compute_velocity`, not the DiT
  itself — sampling-loop orchestration). The noisy latent is passed to the
  DiT unscaled; the timestep passed in is `sigma * num_train_timesteps`
  (the DiT's own `timestep_scale=0.001` divides this back down internally);
  the DiT's raw output (after CFG combination) is fed straight into
  `FlowUniPCMultistepScheduler.step` as `model_output`, which internally
  computes `x0 = sample - sigma_t * model_output` — no
  `c_skip`/`c_out`/`c_in` reconstruction anywhere.
