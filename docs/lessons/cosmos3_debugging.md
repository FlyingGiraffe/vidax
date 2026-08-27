# Cosmos 3 Debugging Lessons

Findings from porting Cosmos3-Nano/Edge worth keeping in mind for this
model family generally. See [`docs/models/cosmos3.md`](../models/cosmos3.md)
for the full port and its current status.

## Prompt richness matters far more for the smaller model

Both models' checkpoints document that "for optimal quality, prompts
should be upsampled into a specific JSON structure" covering subjects,
background/setting, lighting, aesthetics, cinematography, and a temporal
caption (see each checkpoint's own `README.md`) — not a short plain-text
sentence. Nano (16B) tolerates a short prompt reasonably well; Edge (4B)
does not — a short prompt like `"A red panda climbing a bamboo tree"`
produces flat, oversaturated, mostly featureless output on Edge, while
swapping in a real JSON-upsampled prompt (same scheduler settings, same
everything else) produces fully photorealistic, detailed output instead.

**Lesson:** prompt engineering/formatting requirements documented for a
model family aren't a nice-to-have that only matters for polish — for a
smaller model in particular, the gap between a terse prompt and the
checkpoint's documented format can be the difference between unusable and
excellent output, with nothing in the pipeline (no error, no obviously
broken shape) to flag that the prompt itself is the limiting factor.

---

See [`docs/models/cosmos3.md`](../models/cosmos3.md) for current usage and
CLI reference, and [`docs/lessons/cosmos2_5_debugging.md`](cosmos2_5_debugging.md)/
[`docs/lessons/wan2_1_debugging.md`](wan2_1_debugging.md)
for the other two models' real-checkpoint debugging stories.
