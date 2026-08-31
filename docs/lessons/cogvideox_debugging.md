# CogVideoX Debugging Lessons

Things about CogVideoX that aren't obvious from the diffusers reference and
that shaped the JAX port — where CogVideoX differs from the other model
families in this repo in a way that mattered. See
[`docs/models/cogvideox.md`](../models/cogvideox.md) for the full port and
its usage.

## bf16 T5-XXL with no attention mask is catastrophically unstable

CogVideoX conditions on `t5-v1.1-xxl` and — unlike LTX-Video — passes the
encoder **no attention mask** (`_get_t5_prompt_embeds` calls
`self.text_encoder(text_input_ids)` with nothing else), so the full
226-token padded sequence is attended.

`vidax.models.ltx_video.t5.T5Encoder` (reused verbatim) is numerically exact
vs HF **in float32** (rel ~3e-5). But T5-XXL's intermediate residual stream
reaches ~1e5 in magnitude; in bfloat16 that is ~2 significant digits, and
over 24 layers × 226 unmasked positions the error compounds to **16–37%
relative** — and JAX-bf16 and torch-bf16 diverge from *each other* by that
much too. Per-block parity against an fp32 reference input stays at ~6e-7; it
is purely the bf16 residual stream that blows up.

LTX-Video never hit this because it always passes the mask, so the ~200 pad
positions get `-inf` bias and never enter the sum.

**So:** `examples/generate_cogvideox.py` runs the T5 encode in float32
regardless of `--dtype`, casts only the *output* embeddings to bf16 for the
DiT, and frees the ~19 GB fp32 T5 params right after (it is a one-time
prompt encode, not the per-step bottleneck). Don't try to make this work in
bf16 — the reference pipeline in bf16 has the same instability, so the
target is a stable fp32 reference, not bf16 parity.

## The VAE's spatial-tiling blend is stateful

At the reference resolution the un-tiled VAE decode's 512-channel 3D-conv
feature maps OOM a TPU v4 chip, so `CogVideoXVAE.decode` mirrors diffusers'
`tiled_decode` — overlapping tiles blended at the seams with
`blend_v`/`blend_h`. The non-obvious part: diffusers **mutates `rows[i][j]`
in place** as it blends, so each tile is blended against neighbours that
were *themselves* already blended in earlier iterations, not against the
pristine decoded grid. A first pass that blended against the pristine grid
was ~1e-4 everywhere *except* a thin band at each stitch line, where it was
off by up to ~0.9 — that localized-to-the-seams error pattern is the tell.
`_tiled_decode` now writes each blended tile back into the grid before it is
used as a neighbour.

## Joint text+visual attention forces a bespoke sequence-parallel scheme

CogVideoX's DiT has **one joint self-attention** per block over the
concatenated `[text(226); visual]` sequence — there is no separate
cross-attention (`CogVideoXLayerNormZero` modulates the two halves with
their own shift/scale/gate but they share the attention). This is what makes
DeepSpeed-Ulysses sequence parallelism (needed to fit CogVideoX-1.5 at its
native 1360×768 — ~45k visual tokens after `patch_size_t=2`) not a
drop-in of the scheme used for Wan/Cosmos: naively all-to-all-ing the
concatenated `[text; visual_chunk]` would replicate the text tokens
`sp_size` times in the reshuffled KV.

`vidax.core.attention.sequence_parallel_joint_self_attention` handles it:
only the visual q/k/v go through the head↔sequence all-to-all; the small,
fully-replicated text q/k/v is sliced to this device's local head range and
concatenated back in before one local flash-attention call over
`[text(full); visual(full)]`; the visual output is reshuffled back and the
text output is `all_gather`ed over the head axis. Because CogVideoX's AdaLN
modulation is **per-sample, not per-token** (contrast Wan2.2), only the
visual token sequence and the RoPE tables need chunking — there is no
per-token modulation state to shard. TP and SP are kept mutually exclusive
for CogVideoX (the 5B DiT fits replicated per chip in bf16), so none of the
column/row-parallel weight-sharding machinery is threaded through the DiT.
