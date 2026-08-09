# Wan2.2 — Usage Guide

Three standalone TPU inference scripts live in `examples/`. They share the
same building blocks (`vidax.core`, `vidax.schedulers`, `vidax.translator`)
but differ in checkpoint format, resolution defaults, and parallelism
strategy — see [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
for the engineering reasoning behind those differences (Megatron vs.
sequence parallelism, flash attention, JIT-safety, the dtype-casting/decode-
speed bugs found getting Wan2.2 working).

| Script | Params | Task | Checkpoint dir example |
| --- | --- | --- | --- |
| `generate_wan2_2_ti2v.py` | 5B | Text-to-Video **and** Image-to-Video | `Wan2.2-TI2V-5B` |
| `generate_wan2_2_t2v_a14b.py` | 14B (MoE, two experts) | Text-to-Video | `Wan2.2-T2V-A14B` |
| `generate_wan2_2_i2v_a14b.py` | 14B (MoE, two experts) | Image-to-Video | `Wan2.2-I2V-A14B` |

All three build `vidax.models.wan.wan2_2.dit.WanDiT` (fully config-driven,
per-token AdaLN modulation) from a named preset in
`vidax.models.wan.wan2_2.configs` (`TI2V_5B_CONFIG`/`T2V_A14B_CONFIG`/
`I2V_A14B_CONFIG`) — the architecture is identical across all three; only
the size and, for I2V, `in_dim` (extra mask+latent conditioning channels)
differ. All three require the `torch` extra (to deserialize `.pth`/
`.safetensors` checkpoints) and the `text` extra (tokenization):

```bash
pip install -e ".[tpu,torch,text]"
```

`--tokenizer_path` defaults to `<t5_checkpoint_dir>/google/umt5-xxl` for
every script.

---

## TI2V (5B) — `generate_wan2_2_ti2v.py`

TI2V-5B is a single checkpoint that supports **both** text-to-video and
image-conditioned generation in the same script: pass `--image_path` for
i2v, omit it for t2v. Architecturally the two use the model quite
differently — image conditioning works by substituting the known
conditioning frame's latent back into `x` between sampling steps (driven by
a per-token timestep of 0 for that frame's tokens, re-applied after every
step), not by any extra model input the way A14B's i2v does. See
`vidax.models.wan.wan2_2.dit`'s module docstring for the architecture side,
and the reference's `WanTI2V.i2v` (`masks_like`'s frame-0 mask) for the
sampling-loop mechanics this mirrors.

Uses `Wan2.2_VAE.pth` — a different file *and architecture* from Wan2.1's
`Wan2.1_VAE.pth` (48-channel latent space, 2x2 pixel-patchify wrapping, 16x
spatial / 4x temporal compression). See `vidax.models.wan.wan2_2.vae`'s
module docstring.

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
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --tensor_parallel_size 4 \
  --output_path "out/output_ti2v_i2v.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--dit_checkpoint_path` | *required* | Points at the `.safetensors.index.json` manifest, not a single `.safetensors` file — the 5B DiT ships sharded across multiple files. `load_torch_checkpoint_to_jax` resolves and merges every shard automatically; a single non-sharded `.safetensors` still works too. |
| `--vae_checkpoint_path` | *required* | `Wan2.2_VAE.pth`. |
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
| `--num_frames` | `121` | Reference default for TI2V-5B (vs. 81 for Wan2.1/A14B). |
| `--fps` | `24` | Reference `sample_fps` for TI2V-5B (vs. 16 for Wan2.1/A14B). |
| `--output_path` | `output_video.mp4` | With multiple prompts, each saved as `<output_path>_<i>.mp4`. |

**`--tensor_parallel_size` note:** at TI2V-5B's only supported resolution
(704x1280, 121 frames), the patch-token sequence is ~27k long, and Wan2.2's
per-token AdaLN modulation tensors scale with that directly — Megatron-style
tensor parallelism keeps the *full* sequence on every device and doesn't
shrink those, so it doesn't fit a 4-chip v4 slice's HBM even after
quartering weight memory. This script therefore **always** uses
`WanDiT(sequence_parallel=True)` internally when `--tensor_parallel_size >
1`, sharding the token sequence itself between blocks instead (DeepSpeed-
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

## T2V (A14B) — `generate_wan2_2_t2v_a14b.py`

A14B is a **Mixture-of-Experts** model: two separately-checkpointed 14B DiTs
(`high_noise_model`/`low_noise_model` in the checkpoint repo), each the same
`WanDiT` architecture/config (`T2V_A14B_CONFIG`), switched per sampling step
by comparing that step's timestep against `--boundary * num_train_timesteps`
— `high_noise_model` handles the noisier early steps, `low_noise_model` the
later ones, matching the reference's `_prepare_model_for_timestep`. This is
a plain Python-level choice of which params pytree to feed the same jitted
`single_step` on a given iteration (the script logs which expert is used per
step), not a traced/data-dependent branch.

Only **one expert is ever device-resident at a time** — both experts' cast
weights stay in host RAM, and the currently-needed one is `jax.device_put`
onto the mesh only when the schedule crosses the boundary (once, since
`scheduler.timesteps` is monotonic), then dropped before the other is
placed. This matters on small chip counts: at `--tensor_parallel_size 4`, a
single sharded 14B expert's weights (~7.5GB/device in bf16) already leave
little headroom, and Wan2.2's *per-token* AdaLN modulation (unlike Wan2.1's
per-sample modulation — see `vidax.models.wan.wan2_2.dit`'s module
docstring) means the forward pass's own working-set memory scales with the
full token count under Megatron TP alone, since TP shards attention
heads/FFN channels but not the token axis. Keeping both experts resident at
once (as every other multi-checkpoint script in this repo does) was the
very first thing tried here and reliably ran out of HBM.

`--tensor_parallel_size` and `--sequence_parallel_size` now compose freely
(see [hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s
"Combining with Megatron TP") — the currently-resident expert can have both
its weights *and* the token sequence sharded at once, e.g.
`--tensor_parallel_size 2 --sequence_parallel_size 2` on this repo's 4
chips. This measurably helps: it's what let A14B run at a noticeably larger
resolution than either trick alone reached here (see Status below), though
still short of the reference's full 1280x720x81 on just 4 chips.

Unlike TI2V-5B, A14B reuses **Wan2.1's causal VAE** (`Wan2.1_VAE.pth`,
`vae_stride=(4,8,8)`) — the checkpoint repo ships that file, not
`Wan2.2_VAE.pth`. Default resolution/frame count also match Wan2.1
(1280x720, 81 frames), not TI2V-5B's 704x1280x121.

```bash
python examples/generate_wan2_2_t2v_a14b.py \
  --high_noise_dit_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --low_noise_dit_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --prompt "A majestic red panda climbing a bamboo tree in the snow, 4k" \
  --tensor_parallel_size 4 \
  --output_path "out/output_t2v_a14b.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--high_noise_dit_checkpoint_path` / `--low_noise_dit_checkpoint_path` | *required* | The two experts' DiT `.safetensors` checkpoints (or `.safetensors.index.json` manifests, sharded). |
| `--vae_checkpoint_path` | *required* | `Wan2.1_VAE.pth` (bundled in the A14B checkpoint repo). |
| `--t5_checkpoint_path` | *required* | T5 `.pth` checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | Tokenizer directory. |
| `--prompt` | *required*, 1+ values | Same broadcast semantics as Wan2.1 t2v. |
| `--negative_prompt` | reference's `sample_neg_prompt` | Negative prompt for CFG. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--boundary` | `0.875` | Fraction of `num_train_timesteps` (1000) above which `high_noise_model` is used instead of `low_noise_model`. |
| `--tensor_parallel_size` | `1` | Megatron-style, same semantics as `generate_wan2_1_t2v.py`'s 14B path (must divide `num_heads`=40 per expert, 64 for T5). Only one expert is ever device-resident at a time (see above), so this behaves like sharding a single 14B model, not two. Composes with `--sequence_parallel_size` — their product is the real head-divisibility constraint. |
| `--sequence_parallel_size` | `1` | Devices to shard the token sequence itself across (DeepSpeed-Ulysses), independent of `--tensor_parallel_size`'s weight-sharding. Worth trying together with `--tensor_parallel_size` at resolutions where even one device-resident 14B expert alone doesn't fit — see the note above. |
| `--dtype` | `bfloat16` | Same choices/caveats as Wan2.1. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `50` | Sampling steps. |
| `--shift` | `12.0` | Flow-matching noise-schedule shift. Reference default for A14B T2V. |
| `--height` | `720` | Output video height. |
| `--width` | `1280` | Output video width. |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With multiple prompts, each saved as `<output_path>_<i>.mp4`. |

**Status:** verified end-to-end against the real T2V-A14B checkpoints (both
experts, weight shapes/keys confirmed to exactly match
`T2V_A14B_CONFIG`'s param tree) on a 4-chip v4 slice, at two sharding
configurations: `--tensor_parallel_size 4` (128x128, 9 frames) and
`--tensor_parallel_size 2 --sequence_parallel_size 2` (256x256, 9 frames —
4x the pixel count, the combined scheme's actual payoff). Both confirmed
both experts engage correctly (high_noise_model for the earlier steps,
low_noise_model for the rest, matching `boundary=0.875` at `shift=12.0`).
This reduced resolution/frame count is a limitation of this specific
4-chip, ~30GB/chip environment (see the memory note above and the
[hardware doc](../hardware_and_sharding.md#3-sequence-parallelism-deepspeed-ulysses)'s
"Combining with Megatron TP"), not of the model, pipeline, or sharding
code — the reference's default 1280x720x81 needs substantially more
accelerator memory than this repo's Wan2.1-14B/TI2V-5B runs did, and even
combining every parallelism trick this repo has, 4 chips isn't enough to
reach it: a compile-time HLO-temporaries estimate at 480x832x9 frames
(`--tensor_parallel_size 2 --sequence_parallel_size 2`) still came in at
~33.75GB against a 30.75GB/chip budget. More chips (a real v4-16/v4-32 pod
slice, not just this 4-chip machine) should close the remaining gap, since
`--tensor_parallel_size`/`--sequence_parallel_size` both scale with however
many chips are available. Not yet run at full resolution.

---

## I2V (A14B) — `generate_wan2_2_i2v_a14b.py`

Same two-expert MoE switching as T2V-A14B above. Unlike Wan2.1's I2V-14B,
A14B has **no CLIP vision cross-attention branch** at all (Wan2.2's `WanDiT`
never had one). Image conditioning instead concatenates a mask+VAE-latent
`y` (built the same way as Wan2.1 I2V's, from the same Wan2.1 causal VAE)
directly onto the noisy latent's channel axis *before* the DiT call —
matching the reference `WanModel.forward`'s `x = cat([x, y], dim=channel)`
— which is why `I2V_A14B_CONFIG` sets `in_dim=36` (16 noise channels + 20
conditioning channels) instead of Wan2.1 I2V's separate-argument `y`.

```bash
python examples/generate_wan2_2_i2v_a14b.py \
  --high_noise_dit_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --low_noise_dit_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  --vae_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/Wan2.1_VAE.pth" \
  --t5_checkpoint_path "./checkpoints/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_path "./checkpoints/Wan2.2-I2V-A14B/examples/i2v_input.JPG" \
  --prompt "A red panda in the snow" \
  --tensor_parallel_size 4 \
  --output_path "out/output_i2v_a14b.mp4"
```

### CLI reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--high_noise_dit_checkpoint_path` / `--low_noise_dit_checkpoint_path` | *required* | The two experts' DiT checkpoints. |
| `--vae_checkpoint_path` | *required* | `Wan2.1_VAE.pth`. |
| `--t5_checkpoint_path` | *required* | T5 `.pth` checkpoint. |
| `--tokenizer_path` | `<t5_dir>/google/umt5-xxl` | Tokenizer directory. |
| `--image_path` | *required* | Conditioning image. Output resolution is derived from it. |
| `--prompt` | *required* | Text prompt (single string). |
| `--negative_prompt` | reference's i2v `sample_neg_prompt` | Negative prompt for CFG. |
| `--guide_scale` | `5.0` | CFG scale. |
| `--boundary` | `0.900` | Fraction of `num_train_timesteps` above which `high_noise_model` is used. Reference default for I2V (vs. 0.875 for T2V). |
| `--tensor_parallel_size` | `1` | Same semantics as T2V-A14B — only one expert is ever device-resident at a time, so this behaves like sharding a single 14B model (see the memory note in the T2V section above). |
| `--sequence_parallel_size` | `1` | Same DeepSpeed-Ulysses flag, independent of `--tensor_parallel_size` (see the T2V section's memory note). |
| `--dtype` | `bfloat16` | Same choices/caveats. |
| `--seed` | `0` | Initial noise seed. |
| `--num_steps` | `40` | Reference i2v default (vs. 50 for t2v). |
| `--shift` | `5.0` | Reference default for A14B I2V (vs. 12.0 for T2V). |
| `--max_area` | `720*1280` | Bounds output pixel count; same `compute_latent_grid` as Wan2.1 I2V. |
| `--num_frames` | `81` | Output frame count. |
| `--output_path` | `output_video.mp4` | With `dp_size > 1`, each replica's sample is saved as `<output_path>_<i>.mp4`. |

**Status:** verified end-to-end against the real I2V-A14B checkpoints (both
experts, weight shapes/keys confirmed to exactly match
`I2V_A14B_CONFIG`'s param tree, including the channel-concat conditioning
path) on a 4-chip v4 slice, at `--tensor_parallel_size 4` (`--max_area
16384`, 96x144, 9 frames) and at `--tensor_parallel_size 2
--sequence_parallel_size 2` (`--max_area 65536`, 208x288, 9 frames — 4x the
pixel count), both confirming both experts engage correctly
(high_noise_model for the earlier steps, low_noise_model for the rest,
matching `boundary=0.900` at `shift=5.0`). Same reduced-resolution caveat as
T2V-A14B above — not yet run at full resolution.

---

See [`docs/models/wan2_1.md`](wan2_1.md) for Wan2.1 (1.3B/14B T2V, 14B I2V),
[`docs/models/cosmos.md`](cosmos.md) for Cosmos-Predict2.5, and
[`docs/models/cosmos3.md`](cosmos3.md) for Cosmos 3.

See the [Model Support table in the root README](../../README.md#-model-support)
for the up-to-date status across all variants.
