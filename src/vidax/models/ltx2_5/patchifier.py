"""Trivial reshape-based patchify/unpatchify and latent<->pixel coordinate
bounds for LTX-2.5's RoPE, a structural port of the reference's
`VideoLatentPatchifier`/`get_pixel_coords`
(`refs/LTX-2-main/packages/ltx-core/src/ltx_core/components/patchifiers.py`).

Every released checkpoint uses `patch_size=1` for the DiT's own patchifier
(`patchify_proj`'s input width equals the VAE's raw `latent_channels`, no
further spatial merging) -- same as LTX-Video, see
`vidax.models.ltx_video.patchifier`'s docstring. The one real difference
from that module: LTX-2.5's RoPE consumes `[start, end)` bounds per token
per axis (`use_middle_indices_grid=True`, evaluated at the midpoint -- see
`vidax.models.ltx2_5.rope`), not a single corner coordinate, so
`get_latent_coord_bounds`/`latent_to_pixel_coord_bounds` carry a trailing
size-2 axis throughout instead of collapsing to one point.
"""
import jax.numpy as jnp


def patchify(latents: jnp.ndarray) -> jnp.ndarray:
    """(B, F, H, W, C) -> (B, F*H*W, C), row-major (f, h, w) token order."""
    b, f, h, w, c = latents.shape
    return latents.reshape(b, f * h * w, c)


def unpatchify(tokens: jnp.ndarray, f: int, h: int, w: int) -> jnp.ndarray:
    """Inverse of `patchify`: (B, F*H*W, C) -> (B, F, H, W, C)."""
    b, n, c = tokens.shape
    return tokens.reshape(b, f, h, w, c)


def get_latent_coord_bounds(f: int, h: int, w: int, batch_size: int) -> jnp.ndarray:
    """(B, 3, F*H*W, 2) integer latent-space [start, end) bounds per token
    per axis (f, h, w) -- `patch_size=1` throughout, so `end = start + 1`
    -- in the same row-major order `patchify` flattens tokens in.
    """
    ff, hh, ww = jnp.meshgrid(jnp.arange(f), jnp.arange(h), jnp.arange(w), indexing="ij")
    starts = jnp.stack([ff, hh, ww], axis=0).reshape(3, -1)  # (3, F*H*W)
    bounds = jnp.stack([starts, starts + 1], axis=-1)  # (3, F*H*W, 2)
    return jnp.broadcast_to(bounds[None], (batch_size, 3, f * h * w, 2))


def latent_to_pixel_coord_bounds(
    latent_coord_bounds: jnp.ndarray,
    temporal_scale: int,
    spatial_scale: int,
    causal_fix: bool = False,
    fps: float = 24.0,
) -> jnp.ndarray:
    """Scales latent-space `[start, end)` bounds to pixel-space for RoPE.
    `causal_fix` (the checkpoint's own `causal_temporal_positioning` config)
    corrects the temporal axis (both start and end bounds) the same way as
    `vidax.models.ltx_video.patchifier.latent_to_pixel_coords` -- the VAE
    encoder's causal convolutions give the first latent frame a temporal
    receptive field of 1 (not `temporal_scale`) the way every later frame
    has, so its pixel-space timestamp needs the same `+1 - temporal_scale,
    clamped at 0` correction (`ltx_core.components.patchifiers.
    get_pixel_coords`'s `causal_fix` branch).

    `fps`: the real reference (`ltx_core.tools.VideoLatentTools.
    create_initial_state`) divides the temporal axis by the generation's
    target fps *after* the causal-fix correction above (`positions[:, 0,
    ...] = positions[:, 0, ...] / self.fps`) -- `positional_embedding_
    max_pos`'s temporal entry (`20` in the real checkpoint) is calibrated
    in **seconds**, not frame-index units. Skipping this (an earlier
    version of this port did) leaves the temporal RoPE fractional position
    far outside `[0, 1)` at real clip lengths (e.g. `~113/20 ≈ 5.6` for a
    121-frame/24fps clip after `temporal_scale=8` pixel scaling) -- since
    RoPE's `cos`/`sin` are periodic, this wraps the temporal phase multiple
    full rotations across the sequence instead of the intended single
    sweep, a real bug (not a checkpoint-inherent property) responsible for
    periodic/aliased self-attention structure across the whole clip.
    """
    scale_factors = jnp.asarray(
        (temporal_scale, spatial_scale, spatial_scale), dtype=latent_coord_bounds.dtype)
    pixel_coord_bounds = latent_coord_bounds * scale_factors[None, :, None, None]
    if causal_fix:
        pixel_coord_bounds = pixel_coord_bounds.at[:, 0].set(
            jnp.clip(pixel_coord_bounds[:, 0] + 1 - temporal_scale, min=0))
    # The reference casts to float32 here too (`get_pixel_coords(...)
    # .float()` in `VideoLatentTools.create_initial_state`) -- `latent_
    # coord_bounds` (and everything derived from it above) is integer-typed
    # (`get_latent_coord_bounds`'s `jnp.arange`), so without this cast the
    # `/ fps` division below gets silently truncated back to the array's
    # int dtype by `.at[].set()` (a real, separate bug from the missing
    # division itself -- caught via a JAX `FutureWarning` on exactly this
    # unsafe int-cast during a real end-to-end run after the division was
    # first added, not from static inspection).
    pixel_coord_bounds = pixel_coord_bounds.astype(jnp.float32)
    pixel_coord_bounds = pixel_coord_bounds.at[:, 0].set(pixel_coord_bounds[:, 0] / fps)
    return pixel_coord_bounds
