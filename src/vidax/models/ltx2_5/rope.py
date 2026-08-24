"""LTX-2.5's "split" RoPE.

A structural port of `ltx_core.model.transformer.rope` (`generate_freqs`/
`split_freqs_cis`/`apply_split_rotary_emb`) from
`refs/LTX-2-main/packages/ltx-core/src/ltx_core/model/transformer/rope.py`.

Two things differ from `vidax.models.ltx_video.rope` (LTX-Video's RoPE),
despite both models sharing the same `dim // (2 * n_pos_dims)`-band
exponential frequency schedule over fractional pixel-space coordinates:

- **Rotation pairing.** LTX-Video's checkpoints use "interleaved" rotation
  (`rearrange(x, "... (d r) -> ... d r", r=2)`, consecutive-pair rotate,
  `cos`/`sin` built via `repeat_interleave(2)`). LTX-2.5 checkpoints use
  "split" rotation (`rearrange(x, "... (d r) -> ... d r", d=2)`, i.e. a
  plain `x.reshape(..., 2, dim // 2)` -- first-half/second-half rotate, the
  same convention as Llama/GPT-NeoX's `rotate_half`), applied *per attention
  head* rather than to the model's full un-split `inner_dim` at once.
- **Position sampling.** LTX-2.5 uses `use_middle_indices_grid=True`: each
  patch carries a `[start, end)` bound per axis and RoPE is evaluated at the
  *midpoint* of that range, not at a single start-corner coordinate like
  LTX-Video.

Deliberately not shared with `vidax.models.ltx_video.rope`: forcing either
convention to reuse the other's `apply_rope` would produce a silently wrong
(same-shape, plausible-looking) rotation -- see
`docs/lessons/ltx_video_debugging.md` for a real bug of exactly this shape.
"""
import math
from typing import Sequence, Tuple

import jax.numpy as jnp
import numpy as np


def create_ltx2_5_rope_freqs(
    latent_coords: jnp.ndarray,
    dim: int,
    theta: float,
    max_pos: Sequence[int],
    num_attention_heads: int,
    dtype: jnp.dtype = jnp.float32,
    double_precision: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes per-head (cos, sin) RoPE tables for LTX-2.5's self-attention.

    Args:
        latent_coords: (B, n_pos_dims, N, 2) fractional [start, end) pixel-
            space patch bounds per token per axis (n_pos_dims=3 for video:
            f, h, w) -- see `patchifier.latent_to_pixel_coords`. RoPE is
            evaluated at each patch's midpoint (`use_middle_indices_grid`).
        dim: the model's full `inner_dim` (num_attention_heads *
            attention_head_dim).
        theta: `positional_embedding_theta` (10000.0 for released
            checkpoints).
        max_pos: `positional_embedding_max_pos`, one value per axis.
        num_attention_heads: used to reshape the flat per-token frequency
            vector into one slice per head (`d_head // 2` each).
        double_precision: matches `frequencies_precision: "float64"` in a
            checkpoint's embedded config (`double_precision_rope` /
            `generate_freq_grid_np` in the reference, vs. the float32
            `generate_freq_grid_pytorch` path) -- the exponential frequency
            band table (`theta ** linspace(...)`) is computed in numpy
            float64 rather than JAX float32 when set. JAX arrays stay
            float32/bf16 throughout (this never needs `jax_enable_x64`):
            only this small, static `(num_bands,)` table is computed with
            plain numpy before entering JAX.

    Returns:
        (cos, sin), each (B, num_attention_heads, N, d_head // 2), ready for
        `apply_rope` on a (B, N, num_attention_heads, d_head)-shaped query/
        key (post per-head split).
    """
    start, end = latent_coords[..., 0], latent_coords[..., 1]
    midpoint = (start + end) / 2.0  # (B, n_pos_dims, N)

    n_pos_dims = midpoint.shape[1]
    max_pos = jnp.asarray(max_pos, dtype=jnp.float32)
    # (B, N, n_pos_dims): fractional position in [0, 1) per axis.
    fractional_positions = jnp.transpose(midpoint, (0, 2, 1)).astype(jnp.float32) / max_pos

    num_bands = dim // (2 * n_pos_dims)
    if double_precision:
        indices_np = theta ** np.linspace(0.0, 1.0, num_bands, dtype=np.float64)
        indices = jnp.asarray(indices_np * (math.pi / 2), dtype=jnp.float32)
    else:
        indices = theta ** jnp.linspace(0.0, 1.0, num_bands, dtype=jnp.float32)
        indices = indices * (math.pi / 2)

    # (B, N, n_pos_dims, num_bands) -> transpose -> (B, N, num_bands, n_pos_dims) -> flatten.
    freqs = indices[None, None, None, :] * (fractional_positions[..., None] * 2 - 1)
    freqs = jnp.transpose(freqs, (0, 1, 3, 2))
    b, n = freqs.shape[0], freqs.shape[1]
    freqs = freqs.reshape(b, n, num_bands * n_pos_dims)

    expected_freqs = dim // 2
    pad_size = expected_freqs - freqs.shape[-1]

    cos_freq = jnp.cos(freqs)
    sin_freq = jnp.sin(freqs)
    if pad_size != 0:
        cos_freq = jnp.concatenate(
            [jnp.ones((b, n, pad_size), dtype=cos_freq.dtype), cos_freq], axis=-1
        )
        sin_freq = jnp.concatenate(
            [jnp.zeros((b, n, pad_size), dtype=sin_freq.dtype), sin_freq], axis=-1
        )

    # Reshape to per-head slices and move heads next to batch: (B, H, N, d_head // 2).
    cos_freq = cos_freq.reshape(b, n, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, n, num_attention_heads, -1)
    cos_freq = jnp.swapaxes(cos_freq, 1, 2)
    sin_freq = jnp.swapaxes(sin_freq, 1, 2)

    return cos_freq.astype(dtype), sin_freq.astype(dtype)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Applies LTX-2.5's split (rotate-half) RoPE to a per-head query/key.

    Args:
        x: (B, H, N, d_head) -- query or key, already split into heads.
        cos, sin: (B, H, N, d_head // 2), from `create_ltx2_5_rope_freqs`.

    Matches `apply_split_rotary_emb`: `x` is split into first-half/second-
    half along the last axis (`x1, x2 = x[..., :d/2], x[..., d/2:]`, *not*
    consecutive-pair interleaving), rotated as
    `out1 = x1*cos - x2*sin`, `out2 = x2*cos + x1*sin`.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1)
