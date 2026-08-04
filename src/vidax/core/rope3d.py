"""3D Rotary Position Embeddings and sinusoidal timestep embeddings for Wan2.1.

Matches the reference PyTorch implementation exactly:
  - RoPE rotates adjacent (even, odd) coordinate pairs (as in
    ``torch.view_as_complex``), not the "rotate-half" (GPT-NeoX) convention.
  - The per-head rotary frequencies are split unevenly across the temporal,
    height, and width axes: of the ``head_dim // 2`` frequency pairs,
    ``c - 2 * (c // 3)`` are assigned to T and ``c // 3`` each to H and W,
    where ``c = head_dim // 2``.
"""
import jax.numpy as jnp
from typing import Tuple


def sinusoidal_embedding_1d(dim: int, position: jnp.ndarray) -> jnp.ndarray:
    """1D sinusoidal embedding for diffusion timesteps.

    Args:
        dim: Embedding dimension (must be even).
        position: Timesteps, shape (B,).

    Returns:
        Embedding of shape (B, dim), as [cos(freqs), sin(freqs)].
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(jnp.float32)
    freqs = jnp.power(10000.0, -jnp.arange(half, dtype=jnp.float32) / half)
    sinusoid = jnp.outer(position, freqs)
    return jnp.concatenate([jnp.cos(sinusoid), jnp.sin(sinusoid)], axis=1)


def _axis_freqs(seq_len: int, dim: int, theta: float = 10000.0) -> jnp.ndarray:
    """Per-position complex rotation angles for one axis, as (cos, sin).

    Returns arrays of shape (seq_len, dim // 2).
    """
    inv_freq = 1.0 / jnp.power(
        theta, jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    angles = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)
    return jnp.cos(angles), jnp.sin(angles)


def create_rope3d_freqs(
    t: int, h: int, w: int,
    head_dim: int,
    theta: float = 10000.0,
    max_seq_len: int = 1024,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds per-position (cos, sin) rotation angles for a (T, H, W) video grid.

    Mirrors Wan2.1's ``rope_params`` + ``rope_apply``: head_dim // 2 frequency
    pairs are split into three unequal groups for the T, H, and W axes, each
    axis's positions index into its own frequency bank, and the resulting
    per-axis angles are broadcast over the grid and concatenated.

    Args:
        t, h, w: Patchified grid sizes along time, height, width.
        head_dim: Per-head channel dimension (must be divisible by 2).
        theta: RoPE base frequency.
        max_seq_len: Maximum index precomputed per axis (matches the
            reference implementation's fixed buffer size of 1024).

    Returns:
        (cos, sin), each of shape (1, t * h * w, 1, head_dim // 2), ready to
        broadcast against a (B, S, num_heads, head_dim) tensor reshaped into
        (B, S, num_heads, head_dim // 2, 2) pairs.
    """
    c = head_dim // 2
    dim_t = c - 2 * (c // 3)
    dim_h = c // 3
    dim_w = c // 3

    cos_t, sin_t = _axis_freqs(max_seq_len, 2 * dim_t, theta)
    cos_h, sin_h = _axis_freqs(max_seq_len, 2 * dim_h, theta)
    cos_w, sin_w = _axis_freqs(max_seq_len, 2 * dim_w, theta)

    def broadcast_grid(freqs_t, freqs_h, freqs_w):
        gt = jnp.broadcast_to(freqs_t[:t, None, None, :], (t, h, w, dim_t))
        gh = jnp.broadcast_to(freqs_h[None, :h, None, :], (t, h, w, dim_h))
        gw = jnp.broadcast_to(freqs_w[None, None, :w, :], (t, h, w, dim_w))
        return jnp.concatenate([gt, gh, gw], axis=-1).reshape(t * h * w, c)

    cos = broadcast_grid(cos_t, cos_h, cos_w)
    sin = broadcast_grid(sin_t, sin_h, sin_w)
    return cos[None, :, None, :], sin[None, :, None, :]


def apply_rope3d(
    x: jnp.ndarray,
    freqs: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Applies 3D rotary position embeddings via pairwise complex rotation.

    Args:
        x: Tensor of shape (B, S, num_heads, head_dim).
        freqs: (cos, sin), each broadcastable to (1, S, 1, head_dim // 2), as
            returned by ``create_rope3d_freqs``.

    Returns:
        Rotated tensor of the same shape as x.
    """
    cos, sin = freqs
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    out = jnp.stack([out1, out2], axis=-1).reshape(x.shape)
    return out.astype(orig_dtype)
