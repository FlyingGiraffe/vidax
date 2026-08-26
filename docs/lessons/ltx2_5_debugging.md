# LTX-2.5 Debugging Lessons

Bugs and gotchas found while porting LTX-2.5's DiT/VAE/embeddings-connector/
Gemma-4 to JAX/Flax — same bit-exact-against-the-real-reference methodology
as [`docs/lessons/ltx_video_debugging.md`](ltx_video_debugging.md), reused
here (a new `ltx2-verify` conda env, since `ltx-core`'s own pins don't match
the one pinned for LTX-Video). See
[`docs/models/ltx2_5.md`](../models/ltx2_5.md) for the full port.

## The real 22B checkpoint needed more than the plan assumed

The initial port plan assumed the DiT's cross-attention had no AdaLN
modulation and no per-head gating, based on the architecture looking
structurally close to LTX-Video's. The real `ltx-2.5-22b-distilled`
checkpoint's own embedded `config.transformer` set `cross_attention_adaln:
true` and `apply_gated_attention: true`, plus an entire extra component
(an 8-layer "embeddings connector" between Gemma-4 and the DiT,
`use_embeddings_connector: true`) neither the reference's naming nor a
shape-only read would surface.

**Lesson:** read a checkpoint's own embedded config directly before
assuming a "family resemblance" port needs only dimension changes —
confirmed repeatedly in this repo's history (see LTX-Video's own
`causal_temporal_positioning`/`causal_decoder` flags), and here the gap
was large enough to change the shape of the whole plan mid-port.

## VAE: `lax.conv_general_dilated`'s list-padding spec silently disagrees with explicit-pad + `"VALID"`

`vidax.models.ltx2_5.vae.causal_conv3d` originally passed
`padding=[(0, 0), (1, 1), (1, 1)]` directly to `nn.Conv` (matching
`vidax.models.ltx_video.vae`'s already-working equivalent). For most convs
this matched the reference exactly; for the VAE decoder's largest conv
(the 256→512-channel `compress_space` upsample), the two frameworks'
outputs diverged by up to `~6.0` absolute, concentrated at
spatially-zero-padded boundary positions, growing worse through
subsequent blocks (correlation dropped to `~0.12` on the final output).

The divergence reproduced identically whether called through `nn.Conv` or
raw `jax.lax.conv_general_dilated`, and **persisted even at
`precision=jax.lax.Precision.HIGHEST`** — ruling out "just a bf16/default-
precision rounding difference" (the usual `jax_default_matmul_precision`
gotcha already documented for LTX-Video). Replacing the list-spec padding
with an explicit `jnp.pad(...)` followed by `padding="VALID"` reproduced
the reference to `~1e-12` — confirmed independently via a brute-force
manual numpy convolution at several output positions (both `nn.Conv`'s
list-spec output *and* a hand-rolled `lax.conv_general_dilated` call using
the same spec disagreed with the numpy ground truth; the explicit-pad
version matched it).

**Lesson:** `lax.conv_general_dilated`'s convenience padding-list argument
is not provably equivalent to manual-pad-then-`"VALID"` for every shape,
even at the highest requested precision. Fixed unconditionally in
`causal_conv3d` (not gated behind a verification flag) since there's no
principled reason to trust the list-spec path is safe at production bf16
precision either, once it's known to be wrong at float64.

## VAE: the real `VideoEncoder.forward` is deterministic, self-normalizing, mean-only

Unlike LTX-Video's `Encoder` (returns raw `(mean, log_var)` moments; the
*caller* samples and normalizes), LTX-2.5's real
`ltx_core.model.video_vae.video_vae.VideoEncoder.forward` builds the
`latent_log_var="uniform"` expanded `(means, repeated_logvar)` pair
internally purely to reuse a `chunk(2)`-shaped code path, then **discards
the log-var half entirely** and returns only
`self.per_channel_statistics.normalize(means)` — no exposed sampling, no
noise argument, and normalization happens *inside* the encoder rather
than being the caller's job. `ConvVideoDecoder.forward` correspondingly
un-normalizes *inside* itself, right after any (here, always-off)
noise-conditioning step.

An earlier version of `vidax.models.ltx2_5.vae.Encoder`, written by
analogy with LTX-Video's differently-shaped encoder, called
`jnp.split(moments, 2, axis=-1)` on the raw `latent_channels + 1`-wide
`conv_out` output — a naive chunk that silently produces a
half-sized-channel, wrong-but-plausible-shaped result instead of
erroring. Caught only by running the real reference on random input and
inspecting its actual returned shape (`(B, 128, F', H', W')`, not
`(B, 256, ...)`), not by reading the class in isolation.

**Lesson:** "moments in, sample outside" is a common enough VAE pattern
that it's tempting to assume by analogy — check the actual `forward`
return value against the real reference on a dummy input before writing
the calling convention into the port, rather than deriving it from the
architecture's *name*.

## `jnp.tanh` at float64 on this TPU backend returns NaN for large-magnitude inputs

The embeddings connector's `gelu-approximate` activation
(`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`) produced `NaN` partway
through an 8-layer float64 bit-exact check, even though the *input* to
that layer was small (`maxabs ~4.7`) and every weight was finite. Isolated
down to `jnp.tanh` itself: on this session's TPU backend,
`jnp.tanh(jnp.array([100.0], dtype=jnp.float64))` returns `NaN`, while
`numpy.tanh(100.0)` (and `jnp.tanh` on a CPU backend) correctly saturates
to `1.0`. The connector's own pre-final-norm residual stream legitimately
grows into the thousands over its 8 layers (confirmed against the real
reference, which reaches the same magnitudes without producing NaN,
because it never leaves float64/CPU-equivalent precision) — well within
`tanh`'s NaN-triggering range on this backend once amplified through
`0.044715*x^3` inside the GELU formula.

**Lesson:** a verification run comparing against a `.double()` PyTorch
reference should pass `JAX_PLATFORMS=cpu` whenever the code path touches
`tanh`/`gelu` (or anything else transcendental) at float64 — TPU float64
support for these ops isn't reliable at this magnitude range, independent
of anything about the port itself. Not a production concern: bf16/fp32
compute never reaches these magnitudes in the same way, and this only
surfaced because float64 verification deliberately runs everything
without the usual intermediate re-normalizations that keep production
activations bounded.

## Connector: an initializer-only adjustment leaked into the forward pass on real weights

`Embeddings1DConnector`'s `learnable_registers` parameter is trained (real
checkpoints carry real values), but the reference's own *fresh-init*
default is `torch.rand(...) * 2 - 1` (uniform on `[-1, 1)`, vs. Flax's
`nn.initializers.uniform`'s default `[0, scale)`). An earlier version of
this port applied the matching `- 1.0` shift unconditionally in the
forward pass (`registers = (self.param(...) - 1.0)`) rather than only
inside the initializer function — which is invisible when testing with
freshly-initialized (never-checkpoint-loaded) parameters, but silently
shifted every *real, trained* register value down by exactly `1.0` once a
checkpoint was loaded via `.apply()`.

Caught only once the connector's real masking/register-substitution path
was exercised in a bit-exact check (correlation `0.12` before the fix,
`0.9999916` after) — earlier checks that never triggered padding/register
substitution (most of an 8-layer smoke test) never touched this parameter
at all and passed cleanly, giving false confidence.

**Lesson:** an initializer's job is to produce a plausible tensor when no
checkpoint is loaded; any deliberate offset it applies must live entirely
inside the initializer closure, never leak into `__call__` where it would
also apply to real loaded weights. Also: a bit-exact check that never
exercises a conditional code path (padding, here) can't catch bugs in it —
construct at least one test input that forces every branch.

## Attention softmax: a hardcoded `float32` cast silently truncates float64 verification runs

`vidax.models.ltx2_5.dit.LTXAttention` computed its softmax logits via
`.astype(jnp.float32)` unconditionally — the standard, correct pattern for
bf16 production compute (numerically stable, matches how attention
backends generally upcast internally), but it silently downcasts a
float64 verification run's activations, which combined with the
connector's large (thousands-magnitude) intermediate values overflowed to
`inf`/`NaN` well before the final re-normalization that would have
rescaled them back down.

Fixed to `jnp.promote_types(q.dtype, jnp.float32)` — promotes bf16/fp32
inputs up to fp32 exactly as before, but leaves float64 inputs at float64
rather than truncating them.

**Lesson:** "upcast to fp32 for softmax stability" should mean *at least*
fp32, expressed as a promotion, not a hardcoded target dtype — a fixed
target is a silent downcast for any caller already running at higher
precision, which specifically breaks verification (the one context this
matters least in production, but most in a bit-exact check).

## Example script: the connector's own mask format isn't the DiT's mask format

`Embeddings1DConnector` returns `(encoded, additive_attention_mask)` where
the mask is already `(B, 1, 1, L)` and already additive (0.0 valid,
large-negative invalid) — but `LTXDiT.pre_process`'s `encoder_attention_
mask` parameter expects a plain `(B, L)` *binary* mask and builds its own
`[:, None, None, :]`-broadcast additive bias from it. An early version of
`examples/generate_ltx2_5.py` fed the connector's already-4D mask straight
through as the DiT's `encoder_attention_mask` argument; the DiT's own
broadcast then inserted two more axes into an already-4D array, producing
a rank-6 tensor that broke a downstream attention `einsum` several calls
later with a confusing "wrong number of indices" error whose traceback
(mangled by JIT tracing) pointed nowhere near the actual bug.

Fixed by not threading a mask into the DiT at all here: the connector's
own register-substitution already makes every position "real" (no padding
left to mask), matching the reference's own
`modality_from_latent_state(..., context_mask=None)`.

**Lesson:** two components that both deal in "attention masks" don't
necessarily share a *format* (shape, or additive-vs-binary convention) —
check the producer's actual return contract against the consumer's actual
parameter contract explicitly, don't assume compatibility from the name
alone. Also: a JIT-mangled traceback pointing at a plausible-looking but
wrong line is worth being skeptical of; reproducing the failure in a small
non-jitted (or separately-jitted) script isolates the real fault line much
faster than trusting the reported one.

## Manually materialized attention, and the fused-forward-pass memory bug offloading fixed as a side effect

`LTXAttention` originally computed self/cross-attention by hand
(`jnp.einsum` + `jax.nn.softmax`) instead of using `vidax.core.attention
.dot_product_attention`, so it never took the real (O(S) memory) TPU flash-
attention path — always materializing the full `(B, H, S, S)` logits
matrix. Fixed by threading a `mesh` field through `LTXAttention`/
`LTXDiTBlock`/`LTXDiT` and switching to `dot_product_attention(q, k, v,
bias=encoder_attention_bias, scale=scale, mesh=self.mesh)`.

This alone barely moved peak memory (54.13GB → 53.68GB at reference
resolution) — isolated measurement showed flash attention itself only
needs ~0.03GB at that token count, so it was never the dominant cost.
**The actual dominant cost**: compiling the whole 48-block forward pass as
one fused `jax.jit` program does not free each block's own AdaLN/FFN
intermediates before the next block starts — temp memory scaled almost
linearly with block count (1 layer: 4.29GB, 4 layers: 8.99GB, 8 layers:
16.89GB, all at the same token count), not the ~O(1) you'd expect if
peak memory were bounded by any single block's own compute. Fixing this
needed **`--offload_dit_weights`** (`examples/generate_ltx2_5.py`,
mirroring the Wan/Cosmos pattern in `docs/weight_offloading.md`) — but for
a reason distinct from every other model that's used it: the DiT's own
weights were never the bottleneck here (~6.6GB/chip at tp=4, comfortably
resident). What offloading buys here is its *side effect*: splitting the
block loop into `--offload_chunk_size`-sized separately-compiled `jax.jit`
programs bounds peak activation memory to one chunk's worth, independent
of `num_layers`, by construction — closing exactly the gap the fused-trace
measurement above exposed. With this plus the flash-attention fix, the
reference's own single-stage default resolution (704×1216×121) went from
OOM at tp=4 to fitting with room to spare (`--offload_chunk_size 8`,
confirmed as the largest divisor of 48 that still fits — 12 OOMs at 22.28GB
required vs. 16.97GB free with Gemma-4 still resident).

**A second, separate OOM** showed up once resolution was pushed further
(241+ frames): DiT sampling completed fine, but VAE decode then OOM'd —
the exact "DiT weights/closures still resident, competing with VAE
decode's own activation memory" pattern already documented for Wan2.1 in
`docs/weight_offloading.md`. Fixed the same way: explicit `del` on every
DiT/Gemma/connector-side reference (params *and* every closure that
captured them — `single_step`/`dit_apply` or `single_step_offloaded`/
`pre_apply`/`post_apply`/`chunk_forward`/`chunk_params_host`) right before
the VAE decode call. CPython's refcounting frees the underlying HBM
immediately once nothing references it — the closures matter as much as
the params themselves; deleting only the params while a still-referenced
closure (defined earlier, never re-assigned) keeps them alive is a real
trap.

**Lesson:** `jax.jit`-fusing an entire deep sequential network is not
free — measure whether per-block temporaries are actually reused across
blocks (grow a truncated-layer-count model and watch whether peak memory
tracks layer count) before assuming a single fused trace is optimal.
Splitting the trace into per-chunk `jax.jit` calls (the offloading
machinery already built for weight streaming) fixes this as a side
effect even when weight streaming itself isn't needed.

## Missing CFG guidance rescale, and destructively downcast AdaLN tables

Two real, independent quality bugs, found while investigating a user
report that dev-checkpoint output looked "very low quality":

**1. Missing guidance rescale.** The real reference's guidance formula
(`ltx_core.components.guiders.MultiModalGuider.calculate`) is not just
plain CFG — for the dev checkpoint's real recipe (`cfg_scale=3.0,
stg_scale=1.0, rescale_scale=0.7`):
```python
pred = cond + (cfg_scale - 1) * (cond - uncond_text) + stg_scale * (cond - uncond_perturbed)
if rescale_scale != 0:
    factor = rescale_scale * (cond.std() / pred.std()) + (1 - rescale_scale)
    pred = pred * factor
```
`examples/generate_ltx2_5.py` only ever implemented the first (plain CFG)
term — algebraically identical to the reference's own CFG-only term,
confirmed correct — but was missing the rescale correction entirely (STG,
the third term, was a known, documented scope cut from the start, same as
LTX-Video's "plain CFG only" precedent; the rescale wasn't a documented
cut, just missing). At `cfg_scale=3.0` with no rescale correction, this is
exactly the textbook recipe for CFG over-saturation — washed-out color,
blown highlights. Fixed by adding `--guidance_rescale` (defaulting to
`0.7` for `--sampler dev`, `0.0`/no-op for `distilled`, which uses no CFG
at all) and applying the rescale formula unconditionally (a true no-op at
`0.0`, avoiding a Python `if` on a traced value under `jax.jit`).

**2. AdaLN modulation tables destructively downcast to bf16.** The real
`ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors` checkpoints
ship every `scale_shift_table`/`prompt_scale_shift_table` (290 tensors
total — every single one of these two names, at every block, plus the
top-level table) in **float32**, not bf16, unlike the rest of the DiT's
weights (confirmed directly from the checkpoint's own per-tensor dtypes,
not assumed: `4059` bf16 tensors, `290` f32 tensors, and the f32 set is
*exactly* these two names). `examples/generate_ltx2_5.py`'s blanket
`cast_to_dtype(dit_params, dit_dtype)` downcast these along with
everything else. The DiT's own AdaLN math already explicitly promotes to
float32 before use (`scale_shift_table[None, None].astype(jnp.float32)
+ ...`) — but that promotion starts from an already bf16-rounded value if
the *stored* parameter was downcast first, which doesn't recover the lost
mantissa bits. These tables directly scale/shift/gate the residual stream
at every one of 48 blocks, so this loss compounds across the whole depth
of the network in a way it wouldn't for an ordinary large matmul weight.
Fixed with `cast_dit_params` (`examples/generate_ltx2_5.py`), a
path-aware cast that leaves any leaf named `scale_shift_table` or
`prompt_scale_shift_table` at float32 regardless of `--dit_dtype`, casting
everything else normally. (Audited the other three checkpoints — Gemma-4,
the VAE, and the connector's own params — for the same pattern: none of
them have any float32-preserved tensors, so this fix is DiT-specific.)

**Lesson:** "every released checkpoint ships as bfloat16" (a claim
repeated in this port's own CLI help text) turned out to be true for
*most* tensors but not a safe blanket assumption — always check actual
per-tensor dtypes in the checkpoint (`safetensors.safe_open(path).keys()`
+ `get_slice(k).get_dtype()` for every key, not just a handful) before
writing a uniform cast, especially for small, non-matmul parameters
(norm scales, modulation tables) that are cheap enough to keep at higher
precision without a real memory/speed cost, and where a mixed-precision
checkpoint author had every reason to do exactly that.

## The periodic "chunking" artifact: real model behavior, but of the *wrong* decoder

After the two fixes above made output sharper (less oversaturated
softness masking fine detail), a periodic artifact became clearly visible:
temporal (period = `temporal_scale` = 8 frames) and spatial (a repeating
lattice). Verified, with real rigor, that this is **not a bug in this
port's conv-decoder VAE**: decoding a perfectly uniform synthetic latent
(constant across time, and separately constant across space) through both
`vidax.models.ltx2_5.vae`'s JAX decoder and the real PyTorch
`ConvVideoDecoder` reference class — loaded with the checkpoint's real
weights, run in float32 to sidestep bf16 having no optimized PyTorch CPU
kernel (a red herring that initially looked like a 3-4x magnitude
mismatch, actually caused by the *comparison script* missing the
checkpoint's top-level, non-`decoder.`-prefixed `per_channel_statistics
.{mean,std}-of-means` keys and silently falling back to identity
normalization) — produces output matching to ~4 decimal places, periodic
artifact included, in both the temporal and spatial cases.

**But that doesn't mean the artifact is expected/acceptable** — it means
it's specific to the checkpoint this port implements
(`ltx-2.5-video-vae-conv-bf16.safetensors`, `_class_name:
"CausalVideoAutoencoder"`), which is *not* the same VAE the official
LTX-2.5 demos use. The `Lightricks/LTX-2.5` HF repo ships a second video
VAE, `vae/ltx-2.5-video-vae-bf16.safetensors` (no `-conv-` suffix, i.e.
the default-named one), `_class_name: "CausalDiffusionVAE"` / decoder
`NADiffusionDecoder` — an entirely different, transformer-based decoder
using 3D neighborhood attention (NATTEN), architecturally unrelated to
the ResNet + pixel-shuffle-upsample conv decoder. The conv decoder was
picked for this port specifically because NATTEN has no JAX/TPU kernel — a
real, correctly-identified blocker at the time — but that scope cut's
actual visual-quality consequence (this periodic artifact, real and
inherent to the conv decoder specifically) wasn't connected until this
investigation. The diffusion decoder was ported in a later session anyway
(see the entries below) once a pure-JAX NATTEN windowing implementation
made that blocker solvable.

**Lesson:** "verified bit-exact against the reference" answers *whether
this port matches the specific reference component it targets* — it does
not by itself answer *whether that reference component is the one that
produces the output you're comparing against* (e.g. official demo videos).
When a user's quality expectation and a bit-exact-verified result
disagree, check whether there's a different reference *artifact* (a
different checkpoint, a different code path) before concluding the
disagreement must be a bug in the verified component.

## Two more real bugs: a resolution-independent sigma schedule fed the real resolution, and a ~7x-wrong feature-extractor rescale

Found while chasing a user report that `dev`-checkpoint output was
generically plausible-looking but semantically disconnected from the
prompt (a recurring, unrelated "vehicle in a field" motif appearing
across multiple different seeds/prompts/resolutions/CFG settings) — two
separate, real, previously-undetected bugs, neither related to CFG,
rescale, or precision (all methodically ruled out first by disabling each
independently and finding the wrong content persisted regardless).

**1. `compute_shifted_sigmas` was given the real target resolution's token
count, but the real reference never passes one for the single-stage
pipeline this port targets.** `ltx_pipelines.ti2vid_one_stage.py` (the
`dev`/one-stage recipe) calls `LTX2Scheduler.execute(steps=
num_inference_steps)` with **no `latent` argument at all** — `execute`'s
own `default_number_of_tokens: int = MAX_SHIFT_ANCHOR` (4096) then
applies unconditionally, making the real single-stage recipe's sigma
shift **resolution-independent**, always calibrated at exactly
`sigma_shift == max_shift`. (A different pipeline, the two-stage HQ one,
*does* pass a real `latent=` — resolution-dependent shifting is a real
reference code path, just not the one `dev`/one-stage uses.) An earlier
version of this port passed the real resolution's `latent_f * latent_h *
latent_w` (13376 at the reference default) instead of the constant 4096,
extrapolating `sigma_shift` from `2.05` to `~5.37` — producing a
pathological schedule where sigma barely moves for the first 27 of 30
steps (`1.0 → 0.54`) before two huge final steps do essentially all the
denoising. Fixed by defaulting `AncestralEulerScheduler`'s `num_tokens` to
`_MAX_SHIFT_ANCHOR` and no longer passing the real token count from
`examples/generate_ltx2_5.py`. Confirmed via a direct comparison: the
un-shifted 30-value schedule now looks like an ordinary gradually-shifted
curve, not the previous near-flat-then-cliff shape. **This alone did not
fix the wrong-content symptom** (confirmed by testing before/after with
the same seed) — it was a real, independent bug worth fixing regardless,
but the search for the actual cause of wrong content continued.

**2. `extract_video_features`'s rescale used the wrong denominator.** The
reference's `FeatureExtractorV2.forward` calls `_rescale_norm(normed,
v_dim, self.embedding_dim)` where `_rescale_norm(x, target_dim,
source_dim) = x * sqrt(target_dim / source_dim)` — and `self.embedding_dim`
is set from `gemma_text_config.hidden_size` (Gemma's own single-layer
width, e.g. `3840`) at `FeatureExtractorV2` construction time
(`ltx_core.text_encoders.gemma.encoders.encoder_configurator`), **not**
`D * num_layers` (the width of the concatenated, all-49-hidden-states
tensor `normed` actually has, e.g. `3840 * 49 = 188160`). This port's
`extract_video_features` used the *concatenated* width (`d * l`, the most
locally-obvious quantity to reach for, since it's literally `normed`'s own
last-axis size) as the rescale denominator instead of the separate,
smaller `embedding_dim` the reference actually uses — computing a scale
factor `sqrt(out_dim / 188160)` instead of the correct `sqrt(out_dim /
3840)`, off by a factor of `sqrt(49) = 7`. This silently fed the
embeddings connector (and from there, every cross-attention call in the
DiT) a conditioning signal ~7x weaker than intended — not wrong-shaped,
not NaN, not crashing, just badly under-scaled, so the model's own strong
generative prior dominated over the (heavily attenuated) prompt signal
almost everywhere: generically plausible video, weakly if at all
connected to the actual prompt text. This is exactly why it wasn't caught
by this port's own bit-exact checks: the truncated-layer-count Gemma used
for verification throughout this port's original development couldn't
exercise the real `video_aggregate_embed` weight at its true width (see
the Status section of `docs/models/ltx2_5.md` — this was explicitly
flagged as a known, never-completed verification gap from the very start
of the port), so this scaling bug had no bit-exact check that could have
caught it; it only ever showed up as a real-generation content-quality
symptom. Fixed by threading `embedding_dim` (`gemma_model.hidden_size`)
through as an explicit parameter rather than deriving it from the wrong
local tensor shape.

**Confirmed fixed end-to-end**: the same prompt/seed that previously
produced an incongruous vehicle motif (regardless of CFG, rescale, or the
schedule fix alone) now produces a video that actually matches the
prompt, consistently across the full clip.

**Lesson:** when wrong output persists across every guidance/CFG/rescale/
precision toggle, stop varying those and check the *conditioning path*
itself — a systematically-scaled (not NaN, not shape-mismatched)
under-signal is invisible to shape/dtype/crash-based checks and can look
exactly like "the model just isn't very good at this prompt" instead of
what it is: a bug. Also: a documented "structurally verified but never
bit-exact checked" gap (see the Status section's own caveats) is exactly
where an undetected bug like this should be expected to hide — treat such
gaps as a prioritized checklist when a symptom doesn't have an obvious
cause elsewhere, not just a footnote.

## Diffusion (NATTEN) VAE decoder: the full compile-time and memory story

Getting `vidax.models.ltx2_5.diffusion_vae` (the transformer/NATTEN-based
decoder that avoids most of the conv decoder's periodic artifact, see the
entries below) from "bit-exact at small scale" to "runs at the reference's
own `1216x704`/121-frame resolution on one v4-8" took five real, separate
fixes, each only exposed once the previous one let a bigger shape run.
Recorded here as one story since the individual fixes only make sense in
sequence — trying to jump straight to the last one without the intermediate
measurements would look like premature optimization.

**0. Why this needed real engineering at all.** `NeighborhoodAttention3D`'s
3D local-window attention has no native JAX/TPU op (`natten`'s own kernel is
CUDA/Triton-only) — the port has to build the windowing itself, and every
step below is about doing that without either (a) materializing the full
local-window Cartesian product at once, which blows up memory by the
*product* of all window axes, or (b) unrolling the resulting loop in Python,
which blows up *compile* time by the same kind of factor.

**1. Naive full materialization: 147GB at a tiny shape.** The first
working version gathered all three window axes at once
(`(B, T, Kt, H, Kh, W, Kw, NH, HD)`, a direct "im2col" translation of
`fallback_na/eager.py`'s semantics) before the attention einsums. OOM'd
(`RESOURCE_EXHAUSTED`, 147.65GB of HLO temporaries vs. 30.75GB available)
on a *tiny* synthetic latent (`(1, 3, 8, 8, 128)`) — not just at
production scale — because the real checkpoint's stage-5 kernel is
`(11, 11, 11)` (not the `(3, 7, 7)` the pre-download scoping doc assumed,
see the entry below), so the window Cartesian product alone multiplies
memory by `11³ = 1331`.

**2. Per-T Python loop: fixes the OOM, hangs on compile at real scale.**
Processing one query-T-slice at a time (an ordinary Python `for` loop —
`T` is a static shape) and gathering only the `Kh × Kw` window per slice
got a small-shape bit-exact check running (`0.9999946` correlation against
the real PyTorch reference). But a real end-to-end benchmark run at the
reference's own resolution then appeared to hang for over an hour at
"Decoding final latents into video frames" — because the Python loop
*unrolled at trace time* into `T` (~17 at this decoder's own default
resolution) near-identical copies of the same subgraph, repeated across
all 24 blocks (16 det + 8 diffusion). Exact same *shape* of problem as
[`docs/hardware_and_sharding.md`'s "The VAE decode 'hang' that wasn't a
hang"](hardware_and_sharding.md#the-vae-decode-hang-that-wasnt-a-hang) (a
genuine `backend_compile_and_load` cost, not a deadlock) — different root
cause (eager per-op compilation there vs. one `jax.jit` with a massively
unrolled loop inside it here), same fix shape: compile the repeated
per-step body once, reuse it.

**3. `jax.lax.scan` + `jax.vmap`, chunked over the flattened `T * H`
index.** Rewrote `neighborhood_attention_3d` around a single `scan` over
`T * H` (one full `W`-row processed per step, gathering only the `Kw`
window per step — `H`x less peak memory than the `T`-only chunked
version, on top of that version's own `T`x reduction vs. fully-naive) —
`scan` compiles the per-step body exactly once regardless of step count
(fixes the hang), and bounds peak memory to one query row's worth (fixes
a *second*, independent OOM: a moderate `(1, 3, 16, 16, 128)` test —
only 4x more spatial area than the tiny shape step 2 was verified
against — still OOM'd at 43.33GB with the `T`-only chunked version, so
`Kh × Kw` materialized per `T`-step was itself already too large). The
`Kt × Kh` gather within each step is `vmap`'d, not scanned — already
bounded to one row's own footprint, so batching avoids unrolling without
paying `scan`'s serialization cost where nothing forced it. Re-verified
bit-exact after the rewrite (same `0.9999945` correlation, confirming a
pure restructuring) and at the previously-OOMing shape (now succeeds).

**4. Real end-to-end run: still OOMs, but only just — and compile-time is
fine.** With fix 3 in place, a real benchmark run (distilled, T2V,
reference resolution) got all the way through compiling and into
execution before OOMing at **36.6GB** (`context()` + `diffuse()`, all 24
blocks, fused into one `jax.jit` trace) — a `~4000x` improvement over step
1's 147GB, and no more hang, but still `~6GB` over budget. Splitting
`decode()` into separately-`.jit`-able `context()`/`diffuse()` methods
(`setup()`-based, not `@nn.compact`, mirroring `WanVAEDecoder`'s own
`pre_process`/`decode_chunk` split in `docs/hardware_and_sharding.md`) so
each stage's temporaries are freed before the next begins only got to
**34.94GB** for `diffuse()` alone (8 blocks, still one fused trace) —
barely better, suggesting the problem wasn't really "24 blocks accumulate"
so much as "stage 5's own blocks are individually expensive" (kernel
`(11,11,11)` vs. det stages' much smaller `(3,7,7)`/`(3,5,5)`). Splitting
*again*, per-block (`diffuse_prepare`/`diffuse_step`/`diffuse_finalize`,
`block_idx` a static arg so each of the 8 blocks gets its own small
compile — same granularity as the DiT's own `--offload_dit_weights`),
still needed **32.53GB** for a *single* stage-5 block alone. Confirms the
per-block-fusion theory was wrong: one block's own attention + MLP,
already row-chunked by fix 3, is close to the whole budget on its own at
this resolution.

**5. A red herring, then the real fix: tensor parallelism.** Given a
single block needed 32.53GB, the SwiGLU MLP (`hidden_dim=1024`, unchunked,
running over the *full* un-chunked `T * H * W` token volume unlike the
now-row-chunked attention) looked like the obvious remaining suspect —
added the reference's own `SwiGLUTileSpec`/`DEFAULT_SWIGLU_TILES`-style
token tiling (`SwiGLU.num_tiles`, default 4, this module's earlier
docstring had dismissed this as "a PyTorch-CUDA memory optimization, XLA
fuses this fine on TPU without it" — wrong at this decoder's real scale).
**Zero effect**: identical `32.53GB`, to two decimal places, before and
after. Isolated profiling (`jax.jit(...).lower(...).compile().
memory_analysis()`) on synthetic inputs at production shape found
`neighborhood_attention_3d` alone needs only `~4.75GB` and a full block
(attention + MLP + norms) only `~12.24GB` unsharded, single-device — both
far under the `32.53GB` the real, mesh-sharded pipeline actually measured,
a real, unresolved discrepancy between isolated single-device profiling
and the real multi-device run that this session didn't fully explain
(worth investigating if this ever needs to be pushed further). Rather than
keep chasing that gap empirically, applied the fix this repo already has a
working precedent for: Megatron-style tensor parallelism
(`vidax.core.sharding.shard_wan_params`, the same mechanism the DiT/
Gemma-4 already use) across the *existing* `tp=4` mesh — added `w_gate`/
`w_up` to `COLUMN_PARALLEL_NAMES` and `w_down` to `ROW_PARALLEL_NAMES`,
and renamed `NeighborhoodAttention3D`'s output `Dense` from `proj` to
`to_out` (reusing an existing `ROW_PARALLEL_NAMES` entry, and avoiding a
same-file collision with this module's *other* `proj`-named Denses in
`LinearPixelShuffleUpsample`/`AdaLNZero`, which must stay replicated).
`examples/generate_ltx2_5.py` now `shard_wan_params`s the diffusion VAE's
params the same way it already does for the DiT — `Encoder`'s own conv
submodule names aren't in the sharding tables, so it stays correctly
replicated with no code changes there. **Result: `14.76GB` peak HBM/chip**
— comfortably under budget, real end-to-end generation confirmed working
at the full reference resolution.

**Lessons:**
- "Correctness first, optimize later" doesn't mean the first correct-
  looking version is safe to leave alone once it stops erroring, or that
  *any* correctness-first implementation is viable to iterate on at all —
  a naive attention-window gather can blow up memory by the *product* of
  every window axis at once (147GB from one un-chunked implementation
  choice), and this bug surfaced in stages, each only visible at a
  *larger* test shape than the previous fix was verified at. Test at more
  than one shape, including one close to the real target, before declaring
  a memory/compile fix done.
- A fix that reduces peak memory 4000x can still be the wrong place to
  stop — "36.6GB, only slightly over budget" turned out to need three more
  rounds (context/diffuse split, per-block split, then giving up on
  MLP-tiling and reaching for tensor parallelism) before it actually fit.
  Track the remaining gap in absolute terms, not just "did it get better."
- A plausible-looking optimization (SwiGLU tiling, an already-real pattern
  in the reference this port is translating) can measure as a complete
  no-op — verify a memory fix actually changed the number, don't assume a
  targeted-looking change worked because it's the kind of thing that
  *should* help.
- When a whole class of per-component fixes stalls out just short of
  fitting, check whether this repo already has a working mechanism for the
  actual constraint (here: real per-chip memory pressure on a model this
  repo already knows how to shard) before continuing to optimize the
  single-device implementation further.

## Diffusion VAE decoder: the real checkpoint's stage-5 kernel and channel widths differ from the scoping doc's assumed defaults

The prior session's scoping doc (`docs/models/ltx2_5_diffusion_decoder_plan.md`)
assumed `_DIFF_STAGE5_KERNEL_DEFAULT = (3, 7, 7)` (the reference class's own
Python default) before a real `ltx-2.5-video-vae-bf16.safetensors` checkpoint
had been downloaded. The real checkpoint's embedded `config.vae.decoder`
overrides this: `stage5_kernel: (11, 11, 11)`, `stage_channels: (2048, 1024,
512, 512, 256)`, `stage_depths: (4, 6, 4, 2, 8)` — all real, checkpoint-driven
values, not the class defaults. Caught only by reading the checkpoint's own
metadata directly (`safetensors.safe_open(path).metadata()["config"]`)
before writing `vidax.models.ltx2_5.configs.DIFFUSION_VAE_CONFIG`, per this
port's own standing discipline (see this file's very first lesson) — a
config-driven implementation that read real per-tensor keys/shapes for the
translator would still have produced a working *loader*, but the standalone
`DIFFUSION_VAE_CONFIG` fixture (used for documentation/quick-construction
elsewhere) would have silently been wrong had it been transcribed from the
reference class defaults instead.

The checkpoint also carries one parameter, `decoder.type_emb` (shape
`(128,)`), that does not appear anywhere in `refs/LTX-2-main`'s own
loader/model code (confirmed via `grep -rn type_emb`) — treated as an
intentionally-unused/vestigial weight (loaded into no submodule, matching
this port's existing precedent of loading-but-ignoring unused audio-branch
weights elsewhere), consistent with the missing/unexpected-keys check
against the real reference's own `load_state_dict(..., strict=False)`
(`missing: 0`, `unexpected: ['type_emb']` — the *only* unexpected key, on
both the JAX and the real PyTorch side).

**Lesson:** a scoping doc written before the real checkpoint is downloaded
is a plan, not a spec — even when it correctly identifies the *architecture*
(confirmed true here: stages/blocks/RoPE/AdaLN all matched), specific
hyperparameter values (kernel sizes, channel widths, depths) still need to
come from the checkpoint's own embedded config once it exists, not from a
reference class's constructor defaults.

## Diffusion VAE decoder: bit-exact verification result

Verified stages 1-4 (context) + one manual stage-5 diffusion step (real
checkpoint weights, `jax_default_matmul_precision` at its ambient default,
float32 activations, CPU) against the real PyTorch `DiffusionVideoDecoder`
(natten unavailable in this environment, so the reference's own eager
`fallback_na` SDPA backend was used on both the loading side and as the
correctness spec being matched — see `fallback_na/eager.py`'s docstring)
at a small synthetic latent (`(1, 128, 3, 8, 8)`) with an explicit,
identical (numpy-generated, not each framework's own RNG) initial diffusion
noise `x_t` fed to both sides via `DiffusionVideoDecoder.decode`'s new
`x_t` override parameter (a verification-only hook, see its docstring):
**correlation `0.9999946`, max abs diff `0.0045`, mean abs diff
`0.00036`** on the final decoded RGB output. Not the `~1e-8`/float64-grade
bit-exactness this port's other components reach (a float64 check on a
model this deep/wide, with the already-slow Python-loop-based windowed
attention above, was judged too slow to iterate on for this session — a
real, documented gap, not a skipped step) — but strong enough, on real
trained weights end-to-end through every architectural piece (NA windowed
attention + RoPE, `LinearPixelShuffleUpsample`, AdaLN-Zero-modulated
`SwiGLU`/attention residuals, context injection), to be confident the port
is structurally correct. A float64/`HIGHEST`-precision recheck is a
reasonable follow-up once the attention primitive above has a faster
implementation to make iterating on it practical.

## Diffusion VAE decoder: the period-8 artifact is checkpoint-inherent here too, not eliminated

After the memory/tiling fixes above got a real end-to-end generation
running (`--vae_variant diffusion`, distilled checkpoint, T2V, reference
resolution), the output looked visually similar to the conv decoder's own
output for the same prompt/seed — expected on its own (same DiT latent, so
same *content*), but prompting the question of whether the conv decoder's
documented periodic artifact ("The periodic 'chunking' artifact: real
model behavior, but of the *wrong* decoder", earlier in this file) was
still present in the new decoder too, defeating the point of porting it.

**Verified with the same methodology as the original finding, extended to
the diffusion decoder.** The original investigation's diagnostic — decode
a temporally-and-spatially-constant synthetic latent, measure mean abs
pixel diff between frames at every lag 1-16 — was re-run against both this
port's `DiffusionVideoDecoder` *and*, for the first time, the real
untouched PyTorch reference `DiffusionVideoDecoder` (same checkpoint,
`fallback_na` eager SDPA backend since `natten` isn't installed in this
environment). Both show a real dip in the lag-8 curve (frames 8 apart are
more similar to each other than frames 1 apart — the period-8 signature):

| | lag-8 dip / avg(lag 1-7) | 
| --- | --- |
| conv decoder (this port) | 0.248 (dip 75% below neighbors — strong) |
| diffusion decoder (this port) | 0.639 (dip 36% below neighbors — mild) |
| diffusion decoder (**real PyTorch reference**) | 0.638 (dip 36% below neighbors — mild) |

The last two rows match to 3 decimal places on a diagnostic with **no
matched noise** between the two runs (each framework drew its own random
initial diffusion `x_t` independently) — about as strong a confirmation as
a stochastic test can give that this port's diffusion decoder reproduces
the real reference's own behavior, artifact included.

**Conclusion: the period-8 dip is real, checkpoint-inherent behavior of
the diffusion decoder too — not a porting bug, and not something either
decoder does independently "wrong" in the same way. Both real reference
decoders share it**, most plausibly because it traces back to something
more fundamental than either decoder's own architecture (LTX-2.5's video
VAE family's shared 8x temporal downsampling structure is the obvious
suspect, though the *encoder* is identical between both VAE checkpoints —
see this port's module docstring — so a shared-encoder origin is
consistent with a shared symptom across architecturally unrelated
decoders). The diffusion decoder does measurably help — its dip is only
about half as deep, proportionally, as the conv decoder's — matching this
port's own "known quality limitation, not eliminated" framing rather than
a claim of a full fix.

Also re-confirmed while investigating: `video_downscale_factors.time`
(the reference's own fixed constant used in `crop_trailing_context_natten_
pad`'s `time_scale` argument) really is `8` (`SpatioTemporalScaleFactors
.default()`, `refs/LTX-2-main/.../ltx_core/types.py`), matching this
port's own `math.prod(s[0][0] for s in self.upsamples)` computation
exactly — a real point of possible divergence that was checked directly
against the reference source rather than assumed, and confirmed not the
cause.

**Lesson:** when a symptom looks the same across two independently-ported,
architecturally-unrelated components, don't assume a shared porting bug
before checking whether it's a shared *checkpoint* property — re-running
the exact diagnostic that found the first instance against the *second*
component's own real, untouched reference (not just this port's version of
it) is what actually distinguishes "we ported both wrong the same way"
from "the checkpoint really does this, in both, for a reason upstream of
either port." A close but not-quite-eliminated result on a "should fix
this artifact" port is itself worth verifying quantitatively (a lag-diff
ratio, not just "does it still look kind of similar") before concluding
either "still broken" or "fixed."

## The real cause: temporal RoPE was never divided by `fps`, aliasing across the whole clip

The user pushed back on the previous entry's conclusion — real generations
still looked visibly low-quality/"chunky", and it seemed implausible that
would be purely a mild, checkpoint-inherent VAE property as measured
above. Told to check the DiT instead of the VAE. That was the right call:
a real, previously-undetected bug, in `vidax.models.ltx2_5.patchifier`,
shared by every LTX-2.5 generation regardless of `--vae_variant` — this is
what both decoders' outputs were actually showing, not (mainly) the mild
VAE-level artifact documented above.

**The bug**: `ltx_core.tools.VideoLatentTools.create_initial_state`
(`refs/LTX-2-main/packages/ltx-core/src/ltx_core/tools.py`) builds RoPE's
position tensor in two steps — `get_pixel_coords(latent_coords,
scale_factors, causal_fix)` (scales latent coords to pixel-space, applies
the causal-VAE first-frame correction), **then**
`positions[:, 0, ...] = positions[:, 0, ...] / self.fps` — a second,
separate step this port's `vidax.models.ltx2_5.patchifier.
latent_to_pixel_coord_bounds` never had, dividing only the *temporal* axis
by the generation's target fps. The real checkpoint's own
`positional_embedding_max_pos = [20, 2048, 2048]` (confirmed from the
checkpoint's embedded `config.transformer`) is calibrated with this
division already applied — the `20` is **20 seconds**, not 20 frames or
pixels. Height/width's `2048` genuinely are pixel units (no analogous
division exists for those axes) — only the temporal axis gets this
extra step, easy to miss without reading `create_initial_state` specifically
(the RoPE math itself, `get_pixel_coords`/`generate_freqs`, all looks
complete and correct in isolation; the missing division lives one call
site up, in the state-construction helper that builds the tensor RoPE
consumes).

**Effect of skipping it**: `create_ltx2_5_rope_freqs` normalizes each
axis's position by `max_pos` before turning it into a rotation angle
(`fractional_positions = midpoint / max_pos`, then `angle = fractional_
position * theta_band`). Without the `/fps` division, a real generation's
temporal `midpoint` reaches into the hundreds (`~113` pixel-frame-index at
this port's reference `121`-frame/`8x`-temporal-scale default) against
`max_pos[0] = 20`, giving `fractional_positions` up to `~5.6` instead of
the intended `[0, 1)` range. Since RoPE's `cos`/`sin` are periodic, this
doesn't just "extrapolate" the frequency curve — it wraps the temporal
phase **multiple full rotations** across the sequence, aliasing distant
frames' rotary phase back onto each other. This is exactly the kind of
bug that produces structured, semi-periodic corruption in self-attention
(nearby-in-phase-but-far-in-time frames spuriously attending to each other
as if adjacent) rather than random noise — matching the "periodic chunks"
symptom far better than the VAE-level artifact above ever did.

**Verified two ways**:
1. **Latent-space, same seed, before vs. after.** Dumped the raw sampled
   latent (before any VAE decode) for the same prompt/seed with and
   without the fix. The two latents are substantially different (mean abs
   diff `0.72`, max `6.4` — not numerical noise). The lag-diff signature
   (mean abs diff between latent frames at increasing time separation)
   changes shape entirely: **before**, `[0.517, 0.579, 0.644, 0.651,
   0.625, 0.585, 0.568, 0.579]` for lags 1-8 — rises then *falls* then
   rises again, a non-monotonic pattern inconsistent with real video
   content (which should drift smoothly, not partially "return" toward an
   earlier state); **after**, `[0.314, 0.484, 0.608, 0.671, 0.711, 0.753,
   0.779, 0.797]` — cleanly monotonically increasing, the shape a normal
   evolving video's latent should have.
2. **Real end-to-end generation, visual inspection.** Regenerated the
   standard benchmark prompt (distilled checkpoint, T2V, reference
   `1216x704`/121-frame resolution) with the fix and inspected frames 0,
   30, 60, 90, 120 directly. Sharp, coherent, naturally-evolving motion
   throughout (the panda climbs higher and turns its head over the clip)
   with no visible chunking/blockiness at any frame — a clear, visible
   improvement over every pre-fix generation this port had produced.

**Fix**: `latent_to_pixel_coord_bounds` gained an `fps: float = 24.0`
parameter, dividing the temporal axis by it (after the causal-fix
correction, matching the reference's own step order) --
`examples/generate_ltx2_5.py` passes `args.fps` through at its one call
site. `benchmarks/run_ltx2_5.py` already defaults `fps=24` (matching the
CLI default), so its existing call site picks up the fix with no changes
needed there.

**A second, smaller bug caught immediately by re-running end-to-end**: the
first version of this fix divided by `fps` without first casting to
float. `latent_coord_bounds` (built by `get_latent_coord_bounds`'s
`jnp.arange`) is integer-typed, and the causal-fix branch above keeps it
integer throughout -- so the new `/ fps` division produced a `float32`
result that `.at[:, 0].set(...)` then silently cast back down to the
array's `int32` dtype, truncating the fractional part (caught via a real
JAX `FutureWarning` -- "cannot safely cast value from dtype=float32 to
dtype=int32" -- on the very first post-fix end-to-end run, not from static
inspection). The reference itself explicitly guards against exactly this:
`get_pixel_coords(...).float()` in `VideoLatentTools.create_initial_state`
casts to float32 right after the causal-fix step, *before* the `/ fps`
division -- a step this port's first pass at the fix missed. Fixed by
adding the equivalent `.astype(jnp.float32)` at the same point. Confirmed
via a standalone unit check (`pixel_coord_bounds` dtype now `float32`, and
temporal values like `0.041666...` -- `1/24` -- survive with full
precision instead of collapsing to `0`) and a clean end-to-end re-run
(warning gone, output still sharp/coherent).

**Lesson, again**: verify a fix by actually running it, not just by
reasoning that the missing operation is now present -- a `.at[].set()`
silently downcasting a float result into an integer-typed array is exactly
the kind of thing that looks correct by inspection (the division *is*
there, on the right axis, in the right order relative to `causal_fix`) but
is silently wrong at runtime, and a framework's own deprecation warning
caught it immediately where a code read did not.

**Scope of the fix / what needs re-verification**: this bug predates this
session's own VAE decoder work — it was present in the DiT/patchifier code
this session inherited, affecting **every** prior LTX-2.5 generation from
this repo regardless of `--vae_variant`, including the benchmark rows
already recorded in `docs/benchmarking.md` and the "sharp, coherent"
end-to-end status claims in `docs/models/ltx2_5.md`'s own Status section
(written before this fix existed, so likely describing output that was
still affected by this bug to some degree, just not flagged as such at the
time). Those should be treated as stale pending a re-run with the fix, not
as continuing evidence of correctness.

**Lesson**: when a symptom looks structured (periodic, "chunky") rather
than like random noise, suspect a real *positional* bug (an aliasing,
wraparound, or unit-mismatch issue) before a general "checkpoint quality"
explanation — periodicity is a strong, specific signature RoPE/positional-
embedding bugs produce, and a smoothly-looking-plausible-but-wrong initial
hypothesis (the VAE decoder's own genuine, but much smaller, artifact) can
absorb an investigation's attention if a superficially-plausible measured
effect exists at all, even when it isn't the dominant cause of the actual
symptom being chased. Also: a config value's *units* are exactly the kind
of thing that "reads correctly" in isolation (`max_pos: [20, 2048, 2048]`
looks like an unremarkable calibration constant) but is silently wrong
without knowing about a call site one level removed from where it's
consumed -- read the *caller* that constructs a value, not just the
function that consumes it, when a config parameter's meaning depends on
what unit its consumer expects.
