"""3D axial RoPE for Cosmos-Predict2.5's DiT (`VideoRopePosition3DEmb` in the
reference, cosmos-predict2.5-main/cosmos_predict2/_src/predict2/networks/
minimal_v4_dit.py).

Structurally the same idea as Wan's 3D RoPE (`vidax.core.rope3d`) -- split
each head's channels into T/H/W frequency groups, index each group by its
own axis position, broadcast over the (T, H, W) grid -- but two concrete
details differ, so this can't reuse that module directly:

  1. Channel split: Wan splits `head_dim // 2` frequency *pairs* unevenly
     (`c - 2*(c//3)` to T, `c//3` each to H/W). Cosmos splits the full
     `head_dim` directly into `dim_h = dim_w = head_dim // 6 * 2` and
     `dim_t = head_dim - 2*dim_h`, each further halved into frequency pairs.
  2. Rotation convention: Wan rotates *interleaved* adjacent pairs
     `(x[0::2], x[1::2])` (`torch.view_as_complex` style). Cosmos uses
     TransformerEngine's `apply_rotary_pos_emb`, which rotates the two
     *halves* of the channel axis (`rotate_half`: `(x[..., :d/2], x[...,
     d/2:])`) -- the standard GPT-NeoX/Llama convention. The per-axis
     frequency table is therefore concatenated with itself once (not
     interleaved) before taking cos/sin, to align with that split.

Also supports Cosmos's per-axis NTK frequency scaling (`rope_{h,w,t}_
extrapolation_ratio`), used to let a model trained at one resolution/frame
count extrapolate its RoPE table to a larger one at inference -- the 2B
pretrained checkpoint was trained with `h/w_extrapolation_ratio=3.0`,
`t_extrapolation_ratio=1.0` (i.e. only spatial extrapolation is scaled).
FPS-conditioned temporal frequency modulation (`rope_enable_fps_modulation`)
exists in the reference but is disabled for this checkpoint, so it's not
implemented here.
"""
from typing import Tuple

import jax.numpy as jnp


def _axis_freqs(seq_len: int, dim: int, extrapolation_ratio: float, theta: float) -> jnp.ndarray:
    """Per-position rotation angles for one axis, shape (seq_len, dim // 2).

    `extrapolation_ratio` NTK-rescales the frequency base: matches the
    reference's `h_ntk_factor = ratio ** (dim / (dim - 2))`, `theta *= factor`.
    """
    ntk_factor = extrapolation_ratio ** (dim / (dim - 2))
    scaled_theta = theta * ntk_factor
    freq_range = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / jnp.power(scaled_theta, freq_range)
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    return jnp.outer(positions, inv_freq)  # (seq_len, dim // 2)


def create_cosmos_rope3d_freqs(
    t: int, h: int, w: int,
    head_dim: int,
    theta: float = 10000.0,
    h_extrapolation_ratio: float = 1.0,
    w_extrapolation_ratio: float = 1.0,
    t_extrapolation_ratio: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds per-position (cos, sin) rotation angles for a (T, H, W) grid,
    ready for `apply_cosmos_rope3d`'s rotate-half convention.

    Returns:
        (cos, sin), each of shape (1, t * h * w, 1, head_dim), broadcastable
        against a (B, S, num_heads, head_dim) tensor.
    """
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h

    freqs_t = _axis_freqs(t, dim_t, t_extrapolation_ratio, theta)  # (t, dim_t // 2)
    freqs_h = _axis_freqs(h, dim_h, h_extrapolation_ratio, theta)  # (h, dim_h // 2)
    freqs_w = _axis_freqs(w, dim_w, w_extrapolation_ratio, theta)  # (w, dim_w // 2)

    gt = jnp.broadcast_to(freqs_t[:, None, None, :], (t, h, w, dim_t // 2))
    gh = jnp.broadcast_to(freqs_h[None, :, None, :], (t, h, w, dim_h // 2))
    gw = jnp.broadcast_to(freqs_w[None, None, :, :], (t, h, w, dim_w // 2))
    half = jnp.concatenate([gt, gh, gw], axis=-1)  # (t, h, w, head_dim // 2)

    # rotate-half convention: the angle table is duplicated (not
    # interleaved) across the two halves of the channel axis.
    angles = jnp.concatenate([half, half], axis=-1).reshape(t * h * w, head_dim)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    return cos, sin


def apply_cosmos_rope3d(x: jnp.ndarray, freqs: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """Applies rotate-half RoPE, matching TransformerEngine's
    `apply_rotary_pos_emb` (the reference's attention backend).

    Args:
        x: (B, S, num_heads, head_dim).
        freqs: (cos, sin), each broadcastable to (1, S, 1, head_dim), as
            returned by `create_cosmos_rope3d_freqs`.
    """
    cos, sin = freqs
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated_half = jnp.concatenate([-x2, x1], axis=-1)
    out = x * cos + rotated_half * sin
    return out.astype(orig_dtype)
