# Cosmos 3 Debugging Lessons

Bugs found only once Cosmos3-Nano/Edge's dual-pathway MoT DiT/mRoPE/UniPC
port was run end-to-end against real checkpoints. Split out to keep
[`docs/models/cosmos3.md`](../models/cosmos3.md) focused on current usage —
see that doc's Status section for the up-to-date verification summary, and
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
shared TPU/JAX engineering conventions this port also relies on.

Verification methodology before any of these were even looked for: full
checkpoint-value verification (542/542 DiT tensors byte-exact against raw
safetensors, VAE byte-identical between Nano and Edge), an independent
from-scratch PyTorch reimplementation of the reference's attention/MLP/
norm/RoPE math loaded with Edge's real weights (matched the JAX port to
floating-point noise), and the mRoPE relative-position invariant verified at
Edge's actual `rope_theta`. This ruled out the architecture port itself well
before the bugs below were identified — all three were in the example
script's *usage* of the model (position-id construction, resolution/
scheduler defaults, prompt format), not the model port itself.

## Bug 1: vision-segment mRoPE offset used the padded text length, not the real one

The reference has no padding at all (a ragged, per-item design), so it never
faces this. This port pads text to a fixed `--max_text_len` for JAX's static
shapes, and using that padded length to compute the vision segment's mRoPE
temporal offset inflated the relative RoPE gap between vision tokens and the
text tokens they cross-attend to by `max_text_len - real_length` —
differently between the cond and uncond CFG passes whenever their real
lengths differ (which they almost always do, since positive and negative
prompts are rarely the same token count). Fixed by computing each pass's
vision temporal offset from its own real (unpadded) token count
(`generate_cosmos3.py`'s `_vision_position_ids_for`). This alone fully
resolved Nano; Edge improved substantially but not completely — see Bug 2.

## Bug 2: Edge's resolution/frame count and scheduler were both wrong

This repo's shared `--height`/`--width`/`--num_frames` defaults target
Nano's spec (1280x704, 93 frames); Edge needs 480x832, 121 frames instead
(its checkpoint documents a narrower 256p/480p, 50-150 frame range).
Separately, `generate_cosmos3.py` hardcoded `use_karras_sigmas=True`
unconditionally — correct for Nano, wrong for Edge, whose real recipe uses a
non-Karras, shift-warped schedule instead. Running Edge at Nano's resolution,
or with Karras sigmas, doesn't error — it just produces degraded/incoherent
output, including a temporal-instability artifact that escalated past a
fixed absolute frame position on longer clips (initially suspected as a real
directional divergence inside the DiT, before the scheduler was identified
as the actual cause). Fixed by adding `--use_karras_sigmas`/`--shift` CLI
flags (Nano's `use_karras_sigmas=True` default untouched) and correcting the
resolution/frame-count defaults used for Edge in `benchmarks/run_cosmos3.py`.
T2V and I2V need *different* `--shift`/`--num_steps` values for Edge
(confirmed by cross-checking three independent reference backend notebooks)
— don't reuse one task's values for the other.

## Bug 3: prompt format

Both models' checkpoints document that "for optimal quality, prompts should
be upsampled into a specific JSON structure" — this repo had been using
short plain-text prompts throughout. Nano tolerates this reasonably well;
Edge does not — Bugs 1-2's fixes resolved Edge's instability and gross
incoherence, but output remained flat, oversaturated, and short on detail
until switching to a real, JSON-structured prompt/negative-prompt pair,
which produced fully photorealistic, detailed output. See
[`docs/models/cosmos3.md#prompting`](../models/cosmos3.md#prompting) for the
JSON structure and a real example.

## Summary table

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Cross-attention subtly wrong between vision and text tokens, differently for cond vs. uncond | Vision-segment mRoPE offset computed from the padded `--max_text_len` instead of each pass's real token count | Compute the offset from each pass's own real (unpadded) token count |
| Edge output degraded/incoherent, including a temporal-instability artifact on longer clips | Edge run at Nano's resolution/frame count and with Nano's Karras-sigma schedule | Add `--use_karras_sigmas`/`--shift` flags; correct Edge's resolution/frame-count/scheduler defaults |
| Edge output flat, oversaturated, short on detail even after the above | Short plain-text prompts, but Edge's checkpoint expects JSON-structured ones | Use a real, JSON-structured prompt/negative-prompt pair |

---

See [`docs/models/cosmos3.md`](../models/cosmos3.md) for current usage and
CLI reference, and [`docs/lessons/cosmos2_5_debugging.md`](cosmos2_5_debugging.md)/
[`docs/lessons/wan2_1_precision_debugging.md`](wan2_1_precision_debugging.md)
for the other two models' real-checkpoint debugging stories.
