"""LTX-Video's 3-axis fractional-pixel-coordinate RoPE.

A structural port of `Transformer3DModel.precompute_freqs_cis`/
`Attention.apply_rotary_emb` from `refs/LTX-Video-main/ltx_video/models/
transformers/transformer3d.py` and `attention.py`. Deliberately not shared
with `vidax.core.rope3d` (Wan2.1's RoPE): the two differ structurally, not
just in constants --

- Wan2.1 splits the head dim into three axis-sized chunks
  (`c - 2*(c//3)`/`c//3`/`c//3`) and computes frequencies per head.
- LTX computes ``cos``/``sin`` once over the model's full ``inner_dim``
  (*not* per-head) from a `dim // 6`-band exponential frequency schedule,
  with each band contributing one value per axis (f, h, w) before the
  cos/sin repeat-interleave -- and applies it to the *un-split* query/key
  projection, before it's reshaped into per-head form. The frequencies are
  driven by fractional *pixel-space* coordinates (each latent axis divided
  by `positional_embedding_max_pos[axis]`), not raw integer positions.

Forcing either convention to share code with the other would risk a subtly
wrong rotation that only shows up as degraded (not obviously broken) output
-- see `docs/lessons/ltx_video_debugging.md` if issues turn up here.
"""
import math
from typing import Sequence, Tuple

import jax.numpy as jnp


def create_ltx_rope_freqs(
    latent_coords: jnp.ndarray,
    dim: int,
    theta: float,
    max_pos: Sequence[int],
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes (cos, sin) RoPE tables for LTX's self-attention.

    Args:
        latent_coords: (B, 3, N) integer/float latent-space (f, h, w)
            corner coordinates per token, already scaled to pixel space
            (see `vidax.models.ltx_video.patchifier.latent_to_pixel_coords`)
            -- the reference calls this `indices_grid`.
        dim: the model's full `inner_dim` (num_attention_heads *
            attention_head_dim) -- RoPE here spans the whole projection,
            not one head's slice of it.
        theta: `positional_embedding_theta` (10000.0 for every released
            checkpoint).
        max_pos: `positional_embedding_max_pos`, one value per axis
            (f, h, w) -- `[20, 2048, 2048]` for every released checkpoint.

    Returns:
        (cos, sin), each (B, N, dim), ready for `apply_rope` on the
        un-split (B, N, dim) query/key projection.
    """
    max_pos = jnp.asarray(max_pos, dtype=jnp.float32)
    # (B, N, 3): fractional position in [0, 1) per axis.
    fractional_positions = jnp.transpose(latent_coords, (0, 2, 1)).astype(jnp.float32) / max_pos

    num_bands = dim // 6
    # theta ** linspace(log_theta(1), log_theta(theta), num_bands) == theta ** linspace(0, 1, num_bands)
    indices = theta ** jnp.linspace(0.0, 1.0, num_bands, dtype=jnp.float32)
    indices = indices * (math.pi / 2)

    # (B, N, 3, num_bands) -> transpose -> (B, N, num_bands, 3) -> flatten -> (B, N, num_bands * 3)
    freqs = indices[None, None, None, :] * (fractional_positions[..., None] * 2 - 1)
    freqs = jnp.transpose(freqs, (0, 1, 3, 2))
    b, n = freqs.shape[0], freqs.shape[1]
    freqs = freqs.reshape(b, n, num_bands * 3)

    cos_freq = jnp.repeat(jnp.cos(freqs), 2, axis=-1)
    sin_freq = jnp.repeat(jnp.sin(freqs), 2, axis=-1)
    remainder = dim % 6
    if remainder != 0:
        cos_freq = jnp.concatenate([jnp.ones((b, n, remainder), dtype=cos_freq.dtype), cos_freq], axis=-1)
        sin_freq = jnp.concatenate([jnp.zeros((b, n, remainder), dtype=sin_freq.dtype), sin_freq], axis=-1)
    # The reference computes freqs_cis in float32 throughout, then downcasts
    # only at the very end (`cos_freq.to(self.dtype)`) -- the rotation
    # itself (`apply_rope`) then runs in that already-downcast dtype, not
    # float32. Match both precisely: cast here, not inside `apply_rope`.
    return cos_freq.astype(dtype), sin_freq.astype(dtype)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Applies LTX's rotate-half-pairwise RoPE to an un-split (..., dim)
    tensor (query or key, before the per-head reshape) -- matches
    `Attention.apply_rotary_emb`'s `rearrange(x, "... (d r) -> ... d r",
    r=2)` pairing exactly (consecutive-pair, not split-half). Runs in
    `x`'s own dtype (no forced float32), matching the reference exactly --
    `cos`/`sin` are expected to already be in that dtype.
    """
    *batch, d = x.shape
    x_pairs = x.reshape(*batch, d // 2, 2)
    t1, t2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated = jnp.stack([-t2, t1], axis=-1).reshape(*batch, d)
    return x * cos + rotated * sin
