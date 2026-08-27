# Cosmos-Predict2.5 Debugging Lessons

The one finding from porting Cosmos-Predict2.5 worth a longer writeup: a
"grid of random colors" output symptom whose real cause was a wrongly
*added* preconditioning transform, and the diagnostic technique that found
it. See [`docs/models/cosmos2_5.md`](../models/cosmos2_5.md) for the full
port and its current status; the several translation-fidelity bugs found
along the way (a tokenizer API mismatch, a dtype-promotion gotcha, a
missing static-argument declaration, a resolution-divisibility constant
copied from the wrong model) aren't repeated here.

## "Output is a grid of random colors"

Once a full real-checkpoint run was visually inspected for the first time,
output was a rigid, perfectly regular grid of small scrambled color
blobs — not noise, not a blurry-but-recognizable scene, a literal grid.
None of the individual pieces (RoPE, attention dispatch, checkpoint
mapping) showed anything wrong under isolated unit tests with synthetic
inputs, so diagnosing this took line-by-line comparison against the actual
reference PyTorch source, plus a series of ablations against the real
checkpoint to narrow down which stage of the pipeline was responsible. Two
translation bugs (an unpatchify channel-order mismatch, a missing
DiT-internal timestep rescale) turned out to be necessary but not
sufficient — after fixing both, output still looked like the same kind of
grid, which in hindsight should have been the tell that a third, larger
bug was still present.

## The dominant bug: a wrongly-added EDM-style preconditioning wrapper

An earlier revision of this port wrapped every DiT call in
`generate_cosmos2_5.py`'s `compute_velocity` with an EDM-style
`c_in`/`c_skip`/`c_out`/`c_noise` preconditioning transform, modeled on
`RectifiedFlowScaling` (`cosmos_predict2/_src/imaginaire/modules/
denoiser_scaling.py`) — reasonable-looking, since that class genuinely
exists in the reference and genuinely implements EDM-style preconditioning.
The bug: `RectifiedFlowScaling` is never imported by
`Text2WorldModelRectifiedFlow`/`Video2WorldModelRectifiedFlow` — the model
classes this checkpoint's own rectified-flow training config actually
instantiates. It belongs to a *different* reference model class this
checkpoint doesn't use at all. The real reference feeds the raw noisy
latent to the DiT **unscaled**, and uses the DiT's raw output **directly**
as the velocity prediction, fed straight into the UniPC scheduler's own
`x0 = sample - sigma_t * model_output`. Removing the entire preconditioning
wrapper — not adjusting its constants — was the fix.

**The diagnostic that found it**: a real-photo low-noise denoising probe —
encode a real image with the VAE, add a *small* amount of noise, run one
DiT forward pass, decode. This reconstructed the photo almost perfectly at
low noise and degraded into the same meaningless texture at high noise,
isolating the bug specifically to *noise-level conditioning*, not the
network's weights or architecture — a useful diagnostic technique for any
diffusion model producing texture-like or grid-like garbage: it separates
"the network's weights/attention are wrong" from "the model is being told
the wrong noise level" far more directly than staring at RoPE or attention
code. Two other diagnostics tried in the same investigation looked
promising but turned out to be red herrings once the real bug was found
(attention-entropy/gate measurements — a reusable technique even though
this particular use of it was a dead end); the bug was ultimately confirmed
not by any single clever test but by re-reading the reference source a
second time and checking *which model class this specific checkpoint's
training config actually instantiates* — a reminder that a class existing
somewhere in a large reference codebase doesn't mean it's on the path this
checkpoint actually uses.

**Effect confirmed real and large**: before this fix (and the two smaller
translation fixes above), output was a rigid, perfectly regular grid of
scrambled color blobs at every resolution/step-count/guidance-scale tried.
After, output is coherent, prompt-matching video. See
[`docs/models/cosmos2_5.md`](../models/cosmos2_5.md)'s Status section for
the current, up-to-date verification details.

---

See [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
general sharding/JIT engineering conventions, and
[`docs/lessons/wan2_1_debugging.md`](wan2_1_debugging.md)
for a different (and much larger) real-checkpoint debugging story.
