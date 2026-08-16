# Cosmos-Predict2.5 Debugging Lessons

Bugs found only once the Cosmos-Predict2.5 2B DiT/VAE/UniPC-scheduler/
Reason1-text-encoder port was run end-to-end against real checkpoints — none
of them were caught by architecture-level review or synthetic-weight
testing. Split out of [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md)
(which covers general sharding/JIT engineering, not model-specific
debugging) to keep that doc focused; see it for the shared TP/SP/flash-
attention/JIT conventions this port also relies on.

## Bugs that only surfaced against real checkpoints

The port was initially built and verified with a mix of real-weight
shape/forward checks and synthetic conditioning (no local Reason1 checkpoint
existed yet). Once all three real checkpoints (DiT, VAE, and — from a
separate `nvidia/Cosmos-Reason1-7B` download — the text encoder) became
available and the actual `generate_cosmos2_5.py` script was run end-to-end
against them, six real bugs surfaced. None of them were exotic — every one
is a good instance of a class of bug worth watching for generally.

### `apply_chat_template` returning a `BatchEncoding`, not a bare list

`Reason1Tokenizer.__call__` called
`tokenizer.apply_chat_template(conversation, tokenize=True,
add_generation_prompt=False)` expecting a plain `list[int]` back (matching
older `transformers` behavior/most examples in the wild). Against the
`transformers` version actually installed, this returns a `BatchEncoding`
(dict-like, `{"input_ids": [...], "attention_mask": [...]}`) instead —
`ids + [pad_id] * n` then fails with a `TypeError` (`BatchEncoding` doesn't
support `+` with a list). Fixed by unwrapping `ids["input_ids"]` when the
return value is dict-like. A synthetic-tokenizer test would never catch
this — it only shows up against a real `AutoTokenizer.from_pretrained(...)`.

### RoPE dtype: float32 cos/sin tables silently upcast q/k but not v

Reason1's RoPE (`_apply_rotary_pos_emb` in
`vidax.models.cosmos2_5.reason1`) computed `cos`/`sin` tables in
float32 (needed for numerical range/precision) and applied them via `x *
cos + rotate_half(x) * sin` *without* casting the result back to `x`'s
original dtype. Under bf16 (the real checkpoint's native dtype), this
silently upcast `q`/`k` to float32 after RoPE while `v` — never touched by
RoPE — stayed bf16. `jax.nn.dot_product_attention` hard-requires `q`/`k`/`v`
to share one dtype and raises (`"value dtype should be float32, but got
bfloat16"`) rather than promoting for you. Every earlier synthetic test used
float32 throughout, so this never triggered. Fixed by explicitly casting
back to the input's original dtype at the end of `_apply_rotary_pos_emb`
(the same convention `vidax.core.rope3d.apply_rope3d` and
`vidax.models.cosmos2_5.rope.apply_cosmos_rope3d` already followed
correctly — this was an inconsistency between two independently-written
RoPE implementations in the same repo, not a new idea to invent).

### `UniPCState` wasn't a registered JAX pytree

`vidax.schedulers.unipc.UniPCState` is a plain `@dataclasses.dataclass` —
which is *not* automatically a JAX pytree. Passing one as an argument to a
`jax.jit`-wrapped function (`single_step` in `generate_cosmos2_5.py`) fails
immediately: `"...was not marked as static using the static_argnums..."`.
Every earlier test of the scheduler called `.step()` eagerly, outside any
`jax.jit`, so this never came up until the real end-to-end script (which
jits the whole per-step DiT-forward-plus-scheduler-step for speed, like
every other example script in this repo) was actually run. Fixed with
`jax.tree_util.register_dataclass(UniPCState, data_fields=[...],
meta_fields=["this_order"])` — `this_order` specifically has to be a *meta*
(static, hashed-not-traced) field, not a data field, because it's consumed
by genuine Python-level branching inside the predictor/corrector math (see
the next bug for why that distinction matters).

### UniPC's `step_index` needs `static_argnums`, unlike Wan's Euler scheduler

Once `UniPCState` could cross the `jit` boundary, the next failure was a
`step_index` concretization error. Wan's `RectifiedFlowScheduler.step` only
ever *indexes* an array with `step_index` (`self.sigmas[step_index]`), which
works fine whether `step_index` is a concrete Python int or a traced value —
so the existing example scripts never needed to mark it static.
`FlowUniPCMultistepScheduler.step`, by contrast, has genuine Python-level
control flow keyed on `step_index`'s *value* (`if step_index > 0:` to decide
whether to run the corrector, `min(self.num_steps - step_index, ...)` for
the `lower_order_final` ramp) — tracing it as an abstract value hits
`jax.jit` head-on. Fixed by adding `step_index` to `static_argnums` in
`generate_cosmos2_5.py`'s `single_step`. Costs at most `num_steps`-many
retraces per run (one per distinct `step_index`), the same cost class the
per-value-not-per-shape reasoning elsewhere in this repo already accepts.

### A float32 conditioning mask silently upcast the whole DiT to float32

`generate_cosmos2_5.py` built its image2world/video2world conditioning mask
(`cond_mask`/`cond_mask_full`) as plain float32 (convenient for the
timestep-blending arithmetic elsewhere), then passed it straight into
`CosmosDiT`'s `condition_video_mask` input and into the sampling loop's
latent re-clamp arithmetic — both of which mix it with bf16 tensors
(`latents`, VAE-encoded conditioning frames). `jnp.concatenate` and ordinary
elementwise ops silently promote to the *widest* operand dtype, so the
DiT's entire input `x` (and therefore every downstream activation) became
float32 for the whole forward pass, and the sampling loop's `latents`
buffer's dtype flipped from bf16 to float32 after the very first
conditioning-frame re-clamp (which would additionally have broken
`single_step`'s `donate_argnums=(0,)`, which requires a donated buffer's
dtype to stay fixed across calls). The failure that actually surfaced first
was indirect and confusing: a cross-attention dtype mismatch between query
(derived from the now-float32 `x`) and key (derived from `context`, which
was never touched by this bug and stayed properly bf16) — nothing about the
error pointed at the mask. Fixed two ways: defensively inside
`CosmosDiT.__call__` (cast `padding_mask`/`condition_video_mask` to
`latents.dtype` before concatenating, protecting any future caller) and at
the source in the script (build `cond_mask_full` in the target compute
`dtype` directly, keeping a separate float32 `cond_frame_mask` only for the
timestep arithmetic that actually needs it).

(This is a specific instance of a general pattern worth watching for beyond
Cosmos-Predict2.5 too — see
[`docs/lessons/wan2_1_precision_debugging.md`](wan2_1_precision_debugging.md)
for a much larger, harder-to-diagnose version of the same "mixed dtypes
silently promote instead of erroring" class of bug.)

### Conflating the VAE's compression with the VAE+patch combined compression

The image2world/video2world resolution-derivation code computed the
*latent tensor's* spatial size as `pixel_size // 16`, copying Wan2.2 TI2V's
own `dw, dh = pw * 16, ph * 16` divisibility-target constant verbatim. That
constant is correct *for Wan2.2*, whose own VAE already compresses spatially
16x (an 8x causal encoder plus an internal 2x pixel-patchify wrapper) before
the DiT's patch_size divides it further — but Cosmos-Predict2.5 reuses
**Wan2.1's** VAE, which only compresses 8x; the DiT's own `patch_size=(1,2,2)`
is a separate, later factor that must *not* be pre-baked into the latent
tensor's own shape (it's applied internally, inside `CosmosDiT.__call__`,
to the already-VAE-compressed latent). The bug was silent for pure
text2video (no crash — the script just self-consistently allocated,
sampled, and decoded a latent tensor at *half* the intended resolution,
producing a 64x64 video when 128x128 was requested, with nothing to flag
the discrepancy) and only turned into a hard crash for image2world, where
the independently-VAE-encoded conditioning frame (correctly at `pixel/8`)
no longer matched the wrongly-`pixel/16`-sized latent tensor it was being
inserted into. Fixed by using `pixel_size // 8` (the VAE's actual stride)
for the latent tensor's shape, and `pw * 8, ph * 8` (not `* 16`) for the
pixel-resolution divisibility target, so the resulting latent grid still
comes out evenly divisible by `patch_size` as intended. The lesson: a
silently-wrong-but-self-consistent resolution is much harder to catch than
a crash — this one shipped through an earlier real-checkpoint text2video
smoke test undetected because nothing in that test compared the *output*
shape against the *requested* one.

## "Output is a grid of random colors" (real diffusion bugs)

Everything above was found by running the pipeline and checking for
crashes/NaNs/shapes — none of it touches whether the *generated video
actually looks right*, since none of those checks decode and look at a
frame. Once a full real-checkpoint run was visually inspected for the first
time, the output was a rigid, perfectly regular grid of small scrambled
color blobs — not noise, not a blurry-but-recognizable scene, a literal grid.
Diagnosing this took a different kind of verification than the bugs above:
line-by-line comparison against the actual reference PyTorch source (not
just the earlier architecture research notes), plus a series of ablations
run against the real checkpoint to narrow down which stage of the pipeline
was responsible, since none of the individual pieces (RoPE, attention
dispatch, checkpoint mapping) showed anything wrong under isolated unit
tests with synthetic inputs. Four bugs were found and fixed, sequentially —
the fourth and dominant one only surfaced once the first three were fixed
and output was still texture, not a scene.

### Bug 1: `unpatchify`'s channel order isn't the inverse of `patchify`'s

The reference's `PatchEmbed` flattens each patch as `"b c (t r)(h m)(w n)
-> b t h w (c r m n)"` (channel outermost, then temporal-patch, height-
patch, width-patch) — this port replicated that correctly. But the
reference's `unpatchify` uses a *different, non-symmetric* order:
`"B T H W (p1 p2 t C) -> B C (T t)(H p1)(W p2)"` — height-patch, width-patch,
temporal-patch, **channel innermost**. An earlier version of this code
assumed unpatchify was simply patchify's inverse (same channel order) —
a reasonable-looking assumption that happens to be wrong for this specific
reference implementation, and was explicitly flagged as an unconfirmed risk
in `CosmosDiT`'s module docstring before it was checked against source.
Getting this wrong scrambles which of the final layer's 64 output values
maps to which (channel, row-in-patch, col-in-patch) position — every patch
still decodes to *something* locally smooth (each patch's own values are
self-consistent), but adjacent patches don't relate to each other in image
space at all, which is exactly what a literal grid of unrelated blobs looks
like. **Necessary but not sufficient** — fixing this alone made only a
modest per-pixel numerical difference and the output still looked like the
same kind of grid, which in hindsight should have been the tell that a
second, larger bug was still present (see Bug 3).

### Bug 2: `single_step` recompiled the entire 2B-parameter DiT every sampling step

`step_index` was passed as a `static_argnums` of the single `jax.jit`-wrapped
function that *also* contained both DiT forward passes (conditional +
unconditional) — necessary because UniPC's `step()` has genuine Python-level
branching on `step_index`'s value (unlike Wan's Euler scheduler, which only
ever indexes an array with it). Marking any argument of a jitted function
static forces a full retrace — and recompile — of the *entire* function
every time that argument's value changes, not just the small piece of code
that reads it. Since `step_index` changes every step, this meant `num_steps
* 2` full recompiles of a 28-block 2B-parameter model per run (confirmed:
30 low-resolution steps took 16+ minutes wall-clock before the fix, almost
entirely compile time). Fixed by splitting the per-step work: the DiT
forward pass (`compute_velocity`) is its own `jax.jit` with *no*
static/step-dependent argument at all (compiles exactly once, reused for
every step), while `scheduler.step(...)`'s cheap UniPC arithmetic runs
eagerly outside any `jit`, where a static `step_index` costs nothing.

### Bug 3: a missing DiT-internal `timestep_scale=0.001` rescale

Distinct from Bug 4 below and easy to conflate with it: the reference's
`MinimalV1LVGDiT.forward` applies its own internal `timesteps *
self.timestep_scale` (0.001 for this checkpoint) before the sinusoidal
timestep embedding — a rescale that lives *inside* the DiT itself, separate
from whatever preconditioning transform (if any) the surrounding sampling
loop applies before calling it. This port initially omitted it, leaving the
DiT's embedding table conditioned on a ~1000x out-of-distribution noise
value at every step (the same qualitative failure mode as Bug 4, but a
different, DiT-internal source). Fixed by adding `timestep_scale: float =
0.001` as a `CosmosDiT` field (`src/vidax/models/cosmos2_5/dit.py`) and
multiplying it into `timesteps` right before the embedding, matching the
reference exactly.

### Bug 4 (the dominant one): a wrongly-added EDM-style preconditioning wrapper

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
as the velocity prediction, fed straight into
`FlowUniPCMultistepScheduler.step`'s own `x0 = sample - sigma_t *
model_output` — exactly what `vidax.schedulers.unipc
.FlowUniPCMultistepScheduler` was already written to expect (its own
docstring says as much; the scheduler was never the problem). Removing the
entire preconditioning wrapper — not adjusting its constants — was the fix.
`generate_cosmos2_5.py`'s `compute_velocity` now passes `net_in =
current_latents` unscaled and `c_noise = sigma_vec * scheduler
.num_train_timesteps` (the DiT's own `timestep_scale` from Bug 3 divides
this back down internally) directly as `model_output`, with no
reconstruction step at all.

**The diagnostic that found it**: a real-photo low-noise denoising probe —
encode a real image with the VAE, add a *small* amount of noise, run one
DiT forward pass, decode. This reconstructed the photo almost perfectly at
low noise and degraded into the same meaningless texture at high noise,
isolating the bug specifically to *noise-level conditioning*, not the
network's weights or architecture. Two other diagnostics, run in the same
investigation, initially looked promising but turned out to be red herrings
once the real bug was found: attention-entropy/gate measurements (checking
whether cross-attention meaningfully discriminates among text tokens, and
how much its AdaLN gate contributes relative to self-attention's) — a
reusable diagnostic technique even though this particular use of it was a
dead end — and ablations that ruled out other hypotheses first (VAE
round-trip fidelity, RoPE relative-position invariants, weight-loading
exact-match checks, `--tensor_parallel_size 1` vs `4`, `--guide_scale 1.0`).
The bug was ultimately confirmed not by any single clever test but by
re-reading the actual reference source end-to-end a second time, checking
*which model class this specific checkpoint's training config actually
instantiates* rather than assuming any EDM-preconditioning-shaped class
found in the reference applied here — a reminder that a class existing
somewhere in a large reference codebase doesn't mean it's on the path this
checkpoint actually uses.

### Two more real bugs, found in the same investigation

- **Swapped mask-channel concatenation order** at the DiT's input. The
  correct order is `[latents, condition_video_mask, padding_mask]`; this
  port originally concatenated them in the opposite order. The two mask
  channels carry opposite-meaning constants for plain text2world, so the
  swap fed the trained embedding weights inverted per-channel semantics —
  silent, no crash, just wrong conditioning.
- The float32-conditioning-mask dtype-promotion bug described above (a
  separate issue from the channel-order swap: that one was about *dtype*
  silently upcasting the whole DiT input, this one is about channel
  *order*).

**Effect of Bugs 3+4 confirmed real and large**: before these fixes, output
was a rigid, perfectly regular grid of scrambled color blobs at every
resolution/step-count/guidance-scale tried (this was also what Bug 1,
unpatchify, independently caused — see its note about being "necessary but
not sufficient"). After all four bugs were fixed, output is coherent,
prompt-matching video — a recognizable red panda climbing a bamboo stalk
for T2V, a stable identity-preserving subject for I2V. See
[`docs/models/cosmos2_5.md`](../models/cosmos2_5.md)'s Status section for
the current, up-to-date verification details (including 14B's status).

## Summary table

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Reason1 tokenizer crashes with a `TypeError` on `+` | `apply_chat_template` returns a `BatchEncoding`, not a bare list, in the installed `transformers` version | Unwrap `ids["input_ids"]` when the return value is dict-like |
| Reason1 cross-attention raises a q/k/v dtype mismatch under bf16 | RoPE's float32 cos/sin tables upcast q/k but the result was never cast back | Cast RoPE output back to the input's original dtype |
| `single_step` raises "not marked as static" the first time UniPC is jitted | `UniPCState` is a plain dataclass, not a registered JAX pytree | `jax.tree_util.register_dataclass(...)`, with `this_order` as a static meta field |
| UniPC raises a concretization error on `if step_index > 0` | Unlike Euler, UniPC's `step()` has real Python control flow keyed on `step_index`'s value | Add `step_index` to `static_argnums` |
| Cross-attention dtype mismatch that has nothing to do with cross-attention | A float32 conditioning mask upcasts the whole DiT input via `jnp.concatenate`'s dtype promotion | Cast masks to the compute dtype both defensively (in `CosmosDiT`) and at the source (in the script) |
| image2world output resolution silently half of what was requested (t2v version: no error at all) | Latent tensor shape computed as `pixel // 16` (Wan2.2's VAE+patch combined compression), but Cosmos reuses Wan2.1's 8x-only VAE | `pixel // 8` for the latent shape; `patch_size * 8` (not `* 16`) for the resolution divisibility target |
| Generated video is a rigid, perfectly regular grid of scrambled color blobs | `unpatchify`'s channel-flatten order assumed symmetric with `patchify`'s; reference uses a genuinely different order for the two | Reorder unpatchify's reshape/transpose to `(height-patch, width-patch, temporal-patch, channel)`, matching the reference exactly |
| 30 low-resolution sampling steps takes 16+ minutes wall-clock | `step_index` as `static_argnums` of a jit containing both full DiT forward passes forces a full recompile every step | Split into a step-independent jitted DiT forward + an eagerly-run (unjitted) UniPC scheduler step |
| DiT conditioned on a ~1000x out-of-distribution noise level | Missing DiT-internal `timesteps * timestep_scale` (0.001) rescale, distinct from any sampling-loop-level conversion | Add `timestep_scale` as a `CosmosDiT` field, applied before the sinusoidal timestep embedding |
| Grid artifact persists after the unpatchify + timestep_scale fixes, at every resolution/step-count/guidance-scale | A wrongly-**added** EDM-style preconditioning wrapper (`c_skip`/`c_out`/`c_in`/`c_noise`), borrowed from a reference model class (`RectifiedFlowScaling`) this checkpoint's actual training config never uses | Remove the wrapper entirely — pass the raw noisy latent unscaled, use the DiT's raw output directly as `model_output` |
| Trained embedding weights receive inverted per-channel semantics for plain text2world | DiT input mask channels concatenated in the wrong order (`[padding_mask, condition_video_mask]` instead of `[condition_video_mask, padding_mask]`) | Concatenate as `[latents, condition_video_mask, padding_mask]`, matching the reference |

---

See [`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
general sharding/JIT engineering conventions, and
[`docs/lessons/wan2_1_precision_debugging.md`](wan2_1_precision_debugging.md)
for a different (and much larger) real-checkpoint debugging story.
