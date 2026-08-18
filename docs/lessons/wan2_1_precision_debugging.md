# Wan2.1: the fp32/bf16 precision bug behind corrupted large-scale I2V output

Wan2.1 I2V-14B produced severely corrupted output (hazy, flat, low-detail —
not noise, not a crash) specifically at large token counts (native 720P,
81 frames), while smaller configs (480P, fewer frames) at the same
resolution class worked fine. Diagnosing this took an unusually long,
deliberately rigorous investigation, because the corruption's dependence on
*scale* (rather than being reproducible at any size) ruled out most
single-shot ablations and repeatedly pointed toward "numerical noise" as a
tempting but wrong conclusion. See
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
general sharding/JIT conventions this investigation also touched (Megatron
TP vs. sequence parallelism, flash-attention int overflow, JIT compilation
cache) — this doc covers only the precision bug itself.

## Symptom

- Native 720P, 81 frames (the model's actual training config): severely
  corrupted output — hazy, flat, low-detail, visually obviously wrong.
- Smaller configs (480P, fewer frames): correct, sharp, coherent output.
- The corruption's dependence on token count (not on any particular
  resolution or frame count individually) was the key clue, and took
  substantial elimination work to isolate: sharding scheme (Megatron TP vs.
  sequence parallelism), 16-bit integer overflow inside the Pallas flash-
  attention kernel's row/col index arrays, RoPE frequency scaling, causal
  time masking, JIT compilation cache staleness, and package-shadowing were
  all considered and directly ruled out (by source inspection or targeted
  ablation) before the real cause was found.

## Root cause

The PyTorch reference (`refs/Wan2.1-main`) wraps `WanAttentionBlock.forward`'s
residual updates in `amp.autocast(dtype=torch.float32)`, and never casts
back down to bf16 afterward — ordinary type promotion then keeps the
residual stream in float32 for virtually the entire 40-layer network, from
partway through block 0 onward. `amp.autocast(dtype=torch.bfloat16)`
elsewhere in the reference only transiently downcasts *activations* for
specific registered ops (matmul/conv) — it never touches the stored
`nn.Parameter` dtype. Wan2.1's released checkpoints are natively float32 on
disk (confirmed directly via `safetensors`/`torch.load` inspection of the
T2V-1.3B, T2V-14B, and I2V-14B-720P DiT weights — not by reading code) and
loaded as such; "bf16 compute" in the reference is purely a per-op autocast
artifact on top of permanently-float32 weights, never a rounding of the
weights themselves.

This repo's implementation instead treated Wan2.1 like every other bf16-
weights model in the repo: round the checkpoint to bf16 at load, run the
whole block (including the residual stream) in bf16 throughout. That is a
reasonable default and works fine for most model/resolution combinations —
the corruption specifically needs the *combination* of low per-element
precision *and* a very long chain of additions (40 blocks x several
residual adds per block x tens of thousands of tokens) for accumulated
bf16 rounding error to become visually significant. Short chains (fewer
tokens) never accumulate enough error to see; the long chain at native 720P
does. This is exactly the class of bug the model's original architecture
research should have caught by reading the autocast wrapping directly, but
didn't — a documented "we use bf16 for weights and compute" assumption
that was never checked against the reference's actual dtype handling.

## Why this took three separate fixes, not one

Testing an initial single fix repeatedly produced **byte-identical output**
to the broken baseline (same mean/std/max, same MP4 file size) despite real
source edits — the most confusing part of the investigation. Stale JIT
cache (cleared `~/.cache/vidax/jax/` entirely, reproduced identically),
package shadowing (confirmed via `python3 -c "import vidax;
print(vidax.__file__)"` that only one install existed, correctly pointing
at the edited source), and stale output files (checked mtimes — genuinely
fresh, different timestamps, identical content) were all ruled out. The
real explanation was simpler and more mundane: the fix needed all three of
the following bugs fixed *together* before any visible change would occur,
since fixing any one or two alone left a re-quantization point that erased
the others' effect.

### Bug 1 — DiT weights rounded to bf16 at checkpoint load

`cast_to_dtype(dit_params, dtype)` in both example scripts unconditionally
rounded the native-fp32 checkpoint weights down to bf16 at load time,
regardless of what the reference actually does. Fixed by adding a new
`--dit_dtype` CLI flag (default `float32`, independent of the general
`--dtype` flag which still defaults to `bfloat16` for T5/VAE/CLIP — those
really are bf16-native or cheap to run in bf16), applied only to
`dit_params`. `--dit_dtype bfloat16` remains fully supported as an explicit
opt-in, preserving the ability to trade this precision fix away for memory
when it isn't needed (see "Memory cost" below) — the fix is a default
change, not a removed option.

### Bug 2 — `compute_dtype` didn't track `dit_dtype`

With Bug 1 fixed alone, output didn't change at all. A direct sanity test
confirmed `nn.Dense` given an explicit bf16-cast input plus fp32 params
*does* still produce fp32-promoted output (Flax/JAX promotion rules work as
expected) — so the fp32 weights alone should have helped. A runtime
trace-time `print` inside `WanDiT.__call__` revealed why they didn't:
`compute_dtype` (the dtype sub-layer matmuls actually run in) was being
derived from `latents.dtype` — itself controlled by the general `--dtype`
flag, still bf16 — regardless of what `--dit_dtype` said. Every activation
was being explicitly `.astype(bf16)`'d before each Dense call, which forces
bf16-level rounding on the activation *itself* before the (correctly
fp32-promoted) matmul ever runs — the fp32 weights were real, but every
input to them was pre-quantized away. Fixed by adding an explicit
`compute_dtype` field to `WanDiT` (`src/vidax/models/wan/wan2_1/dit.py`),
threaded from the calling script's `dit_dtype`, decoupled entirely from the
general pipeline dtype.

### Bug 3 — latents/output re-quantized every sampling step

With Bugs 1+2 fixed together, output *still* didn't change. Diagnosed by
reproducing the one config already known to work from earlier ad hoc
testing (`--dtype float32 --dit_dtype float32`, i.e. no bf16 anywhere) —
that config succeeded (only OOM'd later during VAE decode, because the VAE
was also unnecessarily cast to fp32, a separate and expected memory cost,
not a correctness bug). This isolated the missing variable to
`latents.dtype`: `WanDiT.__call__`'s final `x = x.astype(input_dtype)`
re-quantizes the DiT's output velocity back down to bf16 at the end of
every single denoising step, regardless of what precision the block
internals just computed at — so even a perfectly fp32-internal DiT was
being fed a freshly bf16-rounded input at the start of every step. Fixed
by removing the `WanDiTBlock`'s downcast at the end of each residual add
(the block now expects float32 `x` on entry and returns float32, matching
the reference's own persistently-float32 residual stream) and by changing
`latents`/`y` (I2V conditioning) construction in the example scripts to use
`dit_dtype` instead of the general `dtype` — since `y` gets concatenated
directly onto `latents` before any Dense layer and must match.

## The actual fix, structurally

`WanDiTBlock` (`src/vidax/models/wan/wan2_1/dit.py`):
- `compute_dtype: jnp.dtype = jnp.bfloat16` field, decoupled from the
  residual stream's dtype.
- `__call__` expects `x` already float32 on entry, returns it float32 (no
  final `.astype(x.dtype)` downcast after residual adds).
- Each sub-layer (self-attention, cross-attention, FFN): normalize/gate in
  float32, `.astype(self.compute_dtype)` immediately before the actual
  attention/matmul call, then `.astype(jnp.float32)` immediately after,
  before the residual add — i.e. bf16 only for the matmul itself, float32
  for everything that accumulates across blocks. This matches the
  reference's autocast boundaries exactly: transient bf16 for specific
  ops, permanent float32 for the residual stream.

`WanDiT`:
- `compute_dtype: Optional[jnp.dtype] = None` field; `x = x.astype(jnp.float32)`
  before the block loop; `_effective_compute_dtype = self.compute_dtype if
  self.compute_dtype is not None else input_dtype`, passed to every block.

Example scripts (`generate_wan2_1_t2v.py`, `generate_wan2_1_i2v.py`):
- New `--dit_dtype` flag (default `float32`), applied to `dit_params`,
  `latents`, and (I2V) the conditioning tensor `y` — decoupled from
  `--dtype` (default `bfloat16`, still governs T5/VAE/CLIP).

`benchmarks/run_wan2_1.py` needed `dit_dtype="float32"` added explicitly to
both its T2V and I2V `argparse.Namespace(...)` constructions — it builds
`Namespace` objects by hand rather than parsing CLI args, so it doesn't
automatically inherit new argparse defaults from the example scripts.

## Verification

Confirmed on real production code (not a synthetic harness), both scales:
native 720P/81 frames (previously corrupted — std ~60-63 after the fix,
visually sharp and coherent) and native 480P (no regression from the
previous, already-correct behavior). A `--dit_dtype bfloat16`
backward-compatibility smoke test (the pre-fix behavior, now opt-in) also
passed cleanly, confirming the option is preserved, not removed.

## Memory cost and current limits

Keeping DiT weights at float32 instead of bf16 roughly doubles their
resident weight memory (the actual multiplier depends on which parameters;
see [`docs/benchmarking.md`](../benchmarking.md) for measured per-model
numbers). On this repo's 4-chip machine, this is the deciding factor behind
a real, measured limit: Wan2.1 I2V-14B at native 720P now genuinely OOMs
(`RESOURCE_EXHAUSTED: Attempting to allocate 21.02M... There are 8.89M
free`) during the initial VAE-encode of the conditioning image, before
generation even begins — confirmed by directly running the benchmark, not
estimated. Tensor parallelism and sequence parallelism were both considered
as a way around this and directly ruled out for this specific problem:
weight memory is only reduced by the (already-maxed) tensor-parallel axis,
so trading tensor-parallel width for sequence-parallel width would *reduce*
available weight sharding while token-activation memory was never the
actual bottleneck here — it would make the OOM worse, not better. [Per-layer weight offloading](../weight_offloading.md) is a genuinely
different approach (reduce *resident* weight memory per step, not shard it
further across chips this machine doesn't have) that fixes this exact
problem — implemented and confirmed working (`--offload_dit_weights`), see
that doc and [`docs/benchmarking.md`](../benchmarking.md) for the measured
result.
