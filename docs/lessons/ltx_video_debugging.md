# LTX-Video Debugging Lessons

Bugs found while porting LTX-Video 0.9.8's DiT/VAE/T5-encoder to JAX/Flax —
unlike this repo's other model ports, correctness here was checked via
direct numerical comparison against the actual reference PyTorch
implementation (not just real-checkpoint end-to-end runs), which caught two
real bugs *before* any end-to-end generation was attempted at all. See
[`docs/models/ltx_video.md`](../models/ltx_video.md) for the full port.

## Verification methodology: a throwaway conda env for bit-exact checks

The reference imports `diffusers` for several base classes (`AdaLayerNormSingle`,
`RMSNorm`, `PixArtAlphaTextProjection`, ...) that this repo doesn't otherwise
depend on, and pinning `diffusers`/`transformers` into the main dev
environment wasn't wanted (keeps that environment clean for everything
else). A separate `conda create -n ltx-verify python=3.11` environment
with `torch==2.3.1`/`diffusers==0.26.3`/`transformers==4.38.2` (plus a
matching `huggingface_hub==0.20.3` pin — newer `huggingface_hub` versions
removed an import `diffusers==0.26.3` still uses) runs the *actual,
unmodified* reference code against the real downloaded checkpoint, dumping
intermediate/final tensors to `.npy` files; the JAX side (in the normal
dev environment) loads the same checkpoint through the port and compares.

One real gotcha in the comparison itself: at JAX's *default* matmul
precision, the two implementations' outputs only agreed to ~2 decimal
places (correlation ~0.9999, max diff ~0.02) even though both were
numerically correct — JAX defaults to a lower-precision matmul algorithm
on this backend for speed. Setting
`jax.config.update("jax_default_matmul_precision", "highest")` before
comparing collapsed the gap to `~3e-5` max diff (correlation
`0.999999999984`). Worth knowing before concluding a port is wrong from a
cross-framework comparison that doesn't force this.

## Bugs that only surfaced against the real reference implementation

### VAE `patchify`/`unpatchify`: width/height merge order

The reference has *two* different pixel-(un)shuffle channel-merge
conventions in the same file
(`causal_video_autoencoder.py`)/`pixel_shuffle.py`): `PixelShuffleND`
(used by `SpaceToDepthDownsample`/`DepthToSpaceUpsample`, the VAE's
internal down/upsample blocks) merges the channel axis in natural
`(c, p_time, p_height, p_width)` order — but the *top-level*
`patchify`/`unpatchify` functions (applied once, right before `conv_in`/
right after `conv_out`) use `rearrange(x, "b c (f p) (h q) (w r) -> b (c p
r q) f h w")` — note `r` (the *width* subpixel factor) comes *before* `q`
(height) in the merge group, despite `q`/height appearing first in the
un-merged pattern. The first version of this port's `_patchify`/
`_unpatchify` reused the same `_merge_subpixel`/`_split_subpixel` helper
written for `PixelShuffleND` (natural height-before-width order) for both
call sites — same *output shape* either way (both are `patch_size x
patch_size` pixel-unshuffles), so nothing crashed and nothing looked
obviously wrong from a shapes-only review. Only the bit-exact numerical
comparison caught it (encoder-output correlation ~0.90 against the
reference instead of ~1.0). Fixed by giving `_patchify`/`_unpatchify` their
own, separately-derived transpose, verified against a brute-force
nested-loop transcription of the reference's exact einops pattern before
trusting it (both the width-before-height merge order and its exact
inverse) — see `vidax.models.ltx_video.vae`'s `_patchify`/`_unpatchify`
docstrings.

**Lesson:** two functions in the same reference file that look like they
should share one helper aren't guaranteed to use the same element order —
check the literal einops/rearrange pattern per call site, not just "this
looks like the same kind of operation as that other one," especially when
a mismatch produces a same-shaped, plausible-looking (just numerically
wrong) result rather than a crash.

### VAE decoder: `causal_decoder=False` means *symmetric* temporal padding, not causal

Every released 0.9.8 checkpoint's VAE config sets `causal_decoder: false`.
The reference's `CausalConv3d.forward(x, causal=True)` branches on this
flag: `causal=True` (the encoder, always) front-pads only, replicating the
first frame `time_kernel_size - 1` times; `causal=False` (the decoder, for
every released checkpoint) pads *symmetrically* instead — replicating the
first frame on the front and the *last* frame on the back, each by
`(time_kernel_size - 1) // 2`. This port's first version hardcoded the
encoder's front-only-causal padding into the one shared `causal_conv3d`
helper and used it everywhere, including every decoder conv — the encoder
came out bit-exact (it really is always causal), but the entire decoder
was numerically wrong throughout (correlation dropped to ~0.32 against the
reference, essentially decorrelated) despite producing correctly-shaped
output that would have looked like a plausible, if low-quality, video.
Fixed by threading a `causal: bool` argument through `causal_conv3d` and
every block that calls it (`ResnetBlock3D`, `UNetMidBlock3D`,
`SpaceToDepthDownsample`, `DepthToSpaceUpsample`), with `Decoder` fixing it
to `False` (`Encoder` keeps the default `True`) — see
`vidax.models.ltx_video.vae.causal_conv3d`'s docstring.

**Lesson:** a config field that reads as "of course this model's [X] is
[Y]" (a *causal* video VAE's decoder being *non*-causal) is exactly the
kind of assumption worth checking against the literal reference code
before porting, not inferring from the model's name/category. This was
also the single largest-magnitude bug found in this port — worth
prioritizing decoder-path checks specifically when a VAE's encode comes
out correct but decode doesn't.

## Not a bug: `jnp.clip`'s `a_min`/`a_max` kwargs

`vidax.models.ltx_video.patchifier.latent_to_pixel_coords`'s
`causal_temporal_positioning` correction (used by the 13B checkpoints,
`False`/unused by 2B — so unexercised by this port's initial 2B-only
smoke tests) called `jnp.clip(x, a_min=0)`, which errors on the installed
JAX version (`jax==0.11.0`) — that alias was removed; only the positional/
`min=`/`max=` spelling works now. Not a numerical-correctness bug, just an
API-version mismatch, but a good reminder that a feature only exercised by
a subset of configs (here, one checkpoint's config value) can hide even a
trivial bug until that subset is actually run — the 2B-only smoke tests
that passed cleanly gave no signal here at all.
