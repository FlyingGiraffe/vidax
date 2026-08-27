# LTX-2.5 Debugging Lessons

Findings from porting LTX-2.5's DiT/VAE/embeddings-connector/Gemma-4 to
JAX/Flax that generalize beyond this one port — memory/compile behavior of
fusing a deep block-stacked network, a mixed-precision checkpoint
convention, and failure modes that show up as generation *quality* issues
rather than crashes. See [`docs/models/ltx2_5.md`](../models/ltx2_5.md) for
the full port and its current status.

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
needed **`--offload_dit_weights`** (mirroring the Wan/Cosmos pattern in
`docs/weight_offloading.md`) — but for a reason distinct from every other
model that's used it: the DiT's own weights were never the bottleneck here
(~6.6GB/chip at tp=4, comfortably resident). What offloading buys here is
its *side effect*: splitting the block loop into `--offload_chunk_size`-
sized separately-compiled `jax.jit` programs bounds peak activation memory
to one chunk's worth, independent of `num_layers`, by construction —
closing exactly the gap the fused-trace measurement above exposed. With
this plus the flash-attention fix, the reference's own single-stage default
resolution (704×1216×121) went from OOM at tp=4 to fitting with room to
spare (`--offload_chunk_size 8`, confirmed as the largest divisor of 48
that still fits — 12 OOMs at 22.28GB required vs. 16.97GB free with Gemma-4
still resident).

**A second, separate OOM** showed up once resolution was pushed further
(241+ frames): DiT sampling completed fine, but VAE decode then OOM'd —
the exact "DiT weights/closures still resident, competing with VAE
decode's own activation memory" pattern already documented for Wan2.1 in
`docs/weight_offloading.md`. Fixed the same way: explicit `del` on every
DiT/Gemma/connector-side reference (params *and* every closure that
captured them) right before the VAE decode call. CPython's refcounting
frees the underlying HBM immediately once nothing references it — the
closures matter as much as the params themselves; deleting only the params
while a still-referenced closure (defined earlier, never re-assigned) keeps
them alive is a real trap.

**Lesson:** `jax.jit`-fusing an entire deep sequential network is not
free — measure whether per-block temporaries are actually reused across
blocks (grow a truncated-layer-count model and watch whether peak memory
tracks layer count) before assuming a single fused trace is optimal.
Splitting the trace into per-chunk `jax.jit` calls (the offloading
machinery already built for weight streaming) fixes this as a side
effect even when weight streaming itself isn't needed.

## Missing CFG guidance rescale, and destructively downcast AdaLN tables

Two real, independent quality bugs, found while investigating a report that
`dev`-checkpoint output looked "very low quality":

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
*exactly* these two names). A blanket `cast_to_dtype(dit_params,
dit_dtype)` downcast these along with everything else. The DiT's own AdaLN
math already explicitly promotes to float32 before use
(`scale_shift_table[None, None].astype(jnp.float32) + ...`) — but that
promotion starts from an already bf16-rounded value if the *stored*
parameter was downcast first, which doesn't recover the lost mantissa
bits. These tables directly scale/shift/gate the residual stream at every
one of 48 blocks, so this loss compounds across the whole depth of the
network in a way it wouldn't for an ordinary large matmul weight. Fixed
with `cast_dit_params` (`examples/generate_ltx2_5.py`), a path-aware cast
that leaves any leaf named `scale_shift_table` or `prompt_scale_shift_table`
at float32 regardless of `--dit_dtype`, casting everything else normally.
(Audited the other three checkpoints — Gemma-4, the VAE, and the
connector's own params — for the same pattern: none of them have any
float32-preserved tensors, so this fix is DiT-specific.)

**Lesson:** "every released checkpoint ships as bfloat16" turned out to be
true for *most* tensors but not a safe blanket assumption — always check
actual per-tensor dtypes in the checkpoint (`safetensors.safe_open(path)
.keys()` + `get_slice(k).get_dtype()` for every key, not just a handful)
before writing a uniform cast, especially for small, non-matmul parameters
(norm scales, modulation tables) that are cheap enough to keep at higher
precision without a real memory/speed cost, and where a mixed-precision
checkpoint author had every reason to do exactly that — the same class of
lesson as
[Wan2.1's fp32-residual-stream story](wan2_1_debugging.md), just
scoped to a handful of small tensors instead of the whole network.

## A ~7x-wrong feature-extractor rescale silently starved every prompt's conditioning signal

Found while chasing a report that `dev`-checkpoint output was generically
plausible-looking but semantically disconnected from the prompt (a
recurring, unrelated "vehicle in a field" motif appearing across multiple
different seeds/prompts/resolutions/CFG settings) — methodically ruled out
CFG, rescale, and precision first (disabling each independently, the wrong
content persisted regardless), then found: `extract_video_features`'s
rescale used the wrong denominator. The reference's `FeatureExtractorV2
.forward` calls `_rescale_norm(normed, v_dim, self.embedding_dim)` where
`_rescale_norm(x, target_dim, source_dim) = x * sqrt(target_dim /
source_dim)`, and `self.embedding_dim` is Gemma's own single-layer width
(e.g. `3840`) — **not** the width of the concatenated, all-49-hidden-states
tensor `normed` actually has (`3840 * 49 = 188160`). This port used the
concatenated width as the rescale denominator instead, computing a scale
factor `sqrt(out_dim / 188160)` instead of the correct `sqrt(out_dim /
3840)` — off by a factor of `sqrt(49) = 7`.

This silently fed the embeddings connector (and from there, every
cross-attention call in the DiT) a conditioning signal ~7x weaker than
intended — not wrong-shaped, not NaN, not crashing, just badly
under-scaled, so the model's own strong generative prior dominated over
the (heavily attenuated) prompt signal almost everywhere: generically
plausible video, weakly if at all connected to the actual prompt text.
Confirmed fixed end-to-end: the same prompt/seed that previously produced
an incongruous vehicle motif now produces a video that actually matches
the prompt, consistently across the full clip.

**Lesson:** when wrong output persists across every guidance/CFG/rescale/
precision toggle, stop varying those and check the *conditioning path*
itself — a systematically-scaled (not NaN, not shape-mismatched)
under-signal is invisible to shape/dtype/crash-based checks and can look
exactly like "the model just isn't very good at this prompt" instead of
what it is: a bug. This is a useful failure mode to keep in mind generally
for text-conditioned video/image generation: a weak or missing prompt
signal doesn't crash, it just produces plausible-but-generic output, which
is easy to misread as a model-quality ceiling rather than a bug.

## Diffusion (NATTEN) VAE decoder: the full compile-time and memory story

Getting `vidax.models.ltx2_5.diffusion_vae` (the transformer/NATTEN-based
decoder mentioned above) from "bit-exact at small scale" to "runs at the
reference's own `1216x704`/121-frame resolution on one v4-8" took five
real, separate fixes, each only exposed once the previous one let a bigger
shape run. Recorded here as one story since the individual fixes only make
sense in sequence — trying to jump straight to the last one without the
intermediate measurements would look like premature optimization.

**0. Why this needed real engineering at all.** `NeighborhoodAttention3D`'s
3D local-window attention has no native JAX/TPU op (`natten`'s own kernel is
CUDA/Triton-only) — the port has to build the windowing itself, and every
step below is about doing that without either (a) materializing the full
local-window Cartesian product at once, which blows up memory by the
*product* of all window axes, or (b) unrolling the resulting loop in Python,
which blows up *compile* time by the same kind of factor.

**1. Naive full materialization: 147GB at a tiny shape.** The first
working version gathered all three window axes at once before the
attention einsums. OOM'd (`RESOURCE_EXHAUSTED`, 147.65GB of HLO
temporaries vs. 30.75GB available) on a *tiny* synthetic latent
(`(1, 3, 8, 8, 128)`) — not just at production scale — because the real
checkpoint's stage-5 kernel is `(11, 11, 11)`, so the window Cartesian
product alone multiplies memory by `11³ = 1331`.

**2. Per-T Python loop: fixes the OOM, hangs on compile at real scale.**
Processing one query-T-slice at a time (an ordinary Python `for` loop —
`T` is a static shape) and gathering only the `Kh × Kw` window per slice
got a small-shape bit-exact check running. But a real end-to-end benchmark
run at the reference's own resolution then appeared to hang for over an
hour — because the Python loop *unrolled at trace time* into `T` (~17 at
this decoder's own default resolution) near-identical copies of the same
subgraph, repeated across all 24 blocks. Exact same *shape* of problem as
[`docs/hardware_and_sharding.md`'s "The VAE decode 'hang' that wasn't a
hang"](../hardware_and_sharding.md#the-vae-decode-hang-that-wasnt-a-hang) (a
genuine compile cost, not a deadlock) — different root cause (eager
per-op compilation there vs. one `jax.jit` with a massively unrolled loop
inside it here), same fix shape: compile the repeated per-step body once,
reuse it.

**3. `jax.lax.scan` + `jax.vmap`, chunked over the flattened `T * H`
index.** Rewrote `neighborhood_attention_3d` around a single `scan` over
`T * H` (one full `W`-row processed per step, gathering only the `Kw`
window per step). `scan` compiles the per-step body exactly once regardless
of step count (fixes the hang), and bounds peak memory to one query row's
worth (fixes a *second*, independent OOM at a moderate test shape only 4x
more spatial area than step 2 was verified against, where `Kh × Kw`
materialized per `T`-step was itself already too large). The `Kt × Kh`
gather within each step is `vmap`'d, not scanned — already bounded to one
row's own footprint, so batching avoids unrolling without paying `scan`'s
serialization cost where nothing forced it.

**4. Real end-to-end run: still OOMs, but only just — and compile-time is
fine.** With fix 3 in place, a real benchmark run got all the way through
compiling and into execution before OOMing at **36.6GB** (all 24 blocks,
fused into one `jax.jit` trace) — a `~4000x` improvement over step 1's
147GB, and no more hang, but still `~6GB` over budget. Splitting `decode()`
into separately-jit-able `context()`/`diffuse()` methods so each stage's
temporaries are freed before the next begins only got to **34.94GB** for
`diffuse()` alone — barely better, suggesting the problem wasn't really
"24 blocks accumulate" so much as "stage 5's own blocks are individually
expensive" (kernel `(11,11,11)` vs. det stages' much smaller
`(3,7,7)`/`(3,5,5)`). Splitting *again*, per-block, still needed **32.53GB**
for a *single* stage-5 block alone — confirming the per-block-fusion
theory was wrong: one block's own attention + MLP, already row-chunked by
fix 3, is close to the whole budget on its own at this resolution.

**5. A red herring, then the real fix: tensor parallelism.** Given a
single block needed 32.53GB, the SwiGLU MLP (unchunked, running over the
*full* un-chunked token volume unlike the now-row-chunked attention)
looked like the obvious remaining suspect — added the reference's own
token-tiling scheme for it. **Zero effect**: identical `32.53GB`, to two
decimal places, before and after. Isolated profiling on synthetic inputs
at production shape found the attention primitive alone needs only
`~4.75GB` and a full block only `~12.24GB` unsharded, single-device — both
far under the `32.53GB` the real, mesh-sharded pipeline actually measured,
a real, unresolved discrepancy between isolated single-device profiling
and the real multi-device run. Rather than keep chasing that gap
empirically, applied the fix this repo already has a working precedent
for: Megatron-style tensor parallelism (`vidax.core.sharding
.shard_wan_params`, the same mechanism the DiT/Gemma-4 already use) across
the existing `tp=4` mesh. **Result: `14.76GB` peak HBM/chip** — comfortably
under budget, real end-to-end generation confirmed working at the full
reference resolution.

**Lessons:**
- "Correctness first, optimize later" doesn't mean the first correct-
  looking version is safe to leave alone once it stops erroring — a naive
  attention-window gather can blow up memory by the *product* of every
  window axis at once (147GB from one un-chunked implementation choice),
  and this bug surfaced in stages, each only visible at a *larger* test
  shape than the previous fix was verified at. Test at more than one
  shape, including one close to the real target, before declaring a
  memory/compile fix done.
- A fix that reduces peak memory 4000x can still be the wrong place to
  stop — "36.6GB, only slightly over budget" turned out to need three more
  rounds before it actually fit. Track the remaining gap in absolute
  terms, not just "did it get better."
- A plausible-looking optimization (token tiling, an already-real pattern
  in the reference this port is translating) can measure as a complete
  no-op — verify a memory fix actually changed the number, don't assume a
  targeted-looking change worked because it's the kind of thing that
  *should* help.
- When a whole class of per-component fixes stalls out just short of
  fitting, check whether this repo already has a working mechanism for the
  actual constraint (here: real per-chip memory pressure on a model this
  repo already knows how to shard) before continuing to optimize the
  single-device implementation further.
