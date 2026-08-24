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
