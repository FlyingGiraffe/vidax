"""Trivial reshape-based patchify/unpatchify and latent<->pixel coordinate
scaling for LTX-Video's RoPE, a structural port of the reference's
`SymmetricPatchifier`/`get_latent_coords`/`latent_to_pixel_coords`
(`refs/LTX-Video-main/ltx_video/models/transformers/symmetric_patchifier.py`,
`.../models/autoencoders/vae_encode.py`).

Every released checkpoint uses `patch_size=1` for the transformer's own
patchifier (confirmed via the checkpoint-embedded config's `patchify_proj`
input width equalling the VAE's raw `latent_channels`, with no further
spatial merging) -- the DiT's actual spatial/temporal compression all
happens in the VAE (`vidax.models.ltx_video.vae`), not here. With
`patch_size=1`, `patchify`/`unpatchify` are pure reshapes, and the token
order they produce is identical to `get_latent_coords`'s row-major
`(f, h, w)` meshgrid flatten -- both implemented as one plain `reshape`
below rather than as two independently-must-agree implementations.
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


def get_latent_coords(f: int, h: int, w: int, batch_size: int) -> jnp.ndarray:
    """(B, 3, F*H*W) integer latent-space (f, h, w) corner coordinates per
    token, in the same row-major order `patchify` flattens tokens in.
    """
    ff, hh, ww = jnp.meshgrid(jnp.arange(f), jnp.arange(h), jnp.arange(w), indexing="ij")
    coords = jnp.stack([ff, hh, ww], axis=0).reshape(3, -1)  # (3, F*H*W)
    return jnp.broadcast_to(coords[None], (batch_size, 3, f * h * w))


def latent_to_pixel_coords(
    latent_coords: jnp.ndarray,
    temporal_scale: int,
    spatial_scale: int,
    causal_fix: bool = False,
) -> jnp.ndarray:
    """Scales latent-space coordinates to pixel-space for RoPE
    (`vidax.models.ltx_video.rope`'s `positional_embedding_max_pos` is
    itself in pixel units). `causal_fix` (the checkpoint's own
    `causal_temporal_positioning` config -- `True` for the 13B variants,
    `False` for 2B, per each's embedded metadata) corrects the first
    latent frame's temporal scale to account for the VAE encoder's causal
    convolutions not having a "before frame 0" receptive field the way
    every later frame does.
    """
    scale_factors = jnp.asarray(
        (temporal_scale, spatial_scale, spatial_scale), dtype=latent_coords.dtype)
    pixel_coords = latent_coords * scale_factors[None, :, None]
    if causal_fix:
        pixel_coords = pixel_coords.at[:, 0].set(
            jnp.clip(pixel_coords[:, 0] + 1 - temporal_scale, min=0))
    return pixel_coords
