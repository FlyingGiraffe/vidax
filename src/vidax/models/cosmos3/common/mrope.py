"""Cosmos3's interleaved 3D mRoPE (`Cosmos3VLTextRotaryEmbedding` in the
reference, refs/diffusers-cosmos3/transformer_cosmos3.py:129-161, and the
position-id builders `get_3d_mrope_ids_text_tokens`/`get_3d_mrope_ids_vae_tokens`
in refs/diffusers-cosmos3/pipeline_cosmos3_omni.py:60-127).

Differs from both Wan's 3D RoPE (`vidax.core.rope3d`) and Cosmos-Predict2.5's
3D RoPE (`vidax.models.cosmos.common.rope`) in how the three (T, H, W) axes'
frequency tables are combined into one `head_dim`-wide table:

  - Wan/Cosmos2.5: *block* layout -- the first `dim_t` channels use T's
    positions, the next `dim_h` use H's, the next `dim_w` use W's (each
    block's own, independently-scaled `inv_freq` sub-table).
  - Cosmos3: *interleaved* layout -- a single shared `inv_freq` table (over
    the full `head_dim // 2` channels) is evaluated at all three axes'
    position ids, then the first `min(rope_axes_dim[1], rope_axes_dim[2]) *
    3` channels are rearranged into repeating `(T, H, W)` triples (channel
    `3k`: T's value at inv_freq index `3k`; `3k+1`: H's value at that same
    index; `3k+2`: W's value at that same index) -- i.e. per triple, the
    three axes share the *same* underlying frequency, just evaluated at each
    axis's own position. The remaining tail channels (`rope_axes_dim[0] -
    rope_axes_dim[1]`, here `24 - 20 = 4`) stay purely T-indexed. See
    `apply_interleaved_mrope` for the exact index arithmetic.

Uses the standard GPT-NeoX/Llama "rotate_half" convention (halves of the
channel axis rotated together), same as Cosmos-Predict2.5's RoPE and Reason1's
own text RoPE -- the angle table is duplicated (not interleaved) across the
two halves before taking cos/sin.
"""
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp


def _inv_freq(head_dim: int, theta: float) -> jnp.ndarray:
    """Shared frequency table over `head_dim // 2` channels, one shared table
    for all three (T, H, W) axes (`Cosmos3VLTextRotaryEmbedding.__init__`)."""
    return 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))


def get_mrope_ids_text_tokens(num_tokens: int, temporal_offset: float = 0.0) -> jnp.ndarray:
    """(3, num_tokens) position ids for a text segment: all three axes share
    the same sequential positions, matching `get_3d_mrope_ids_text_tokens`.
    """
    ids = jnp.arange(num_tokens, dtype=jnp.float32) + temporal_offset
    return jnp.broadcast_to(ids[None, :], (3, num_tokens))


def get_mrope_ids_vision_tokens(
    grid_t: int, grid_h: int, grid_w: int,
    temporal_offset: float = 0.0,
    fps: Optional[float] = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
) -> jnp.ndarray:
    """(3, grid_t*grid_h*grid_w) position ids for a patchified vision segment,
    flattened in (t, h, w) row-major order -- matching `x`'s own patch-token
    flatten order elsewhere in this port -- so index `i` here lines up with
    token `i` of the vision segment. Matches `get_3d_mrope_ids_vae_tokens`
    (`reset_spatial_indices=True`, the checkpoint's own default).

    `fps` FPS-modulates the temporal axis (only when `grid_t > 1`, i.e. never
    for single-frame/image mode) so a fixed number of *pixel-time* seconds
    maps to the same temporal position regardless of how many latent frames
    that clip was encoded into -- irrelevant when generating at the model's
    trained `base_fps`, where it's a no-op (`tps == base_tps`), but kept
    exact since callers may pass a different `fps`.
    """
    fps_modulation = fps is not None and grid_t > 1
    if fps_modulation:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_idx = jnp.arange(grid_t, dtype=jnp.float32)
        t_scalar = frame_idx / tps * base_tps + temporal_offset
    else:
        t_scalar = jnp.arange(grid_t, dtype=jnp.float32) + temporal_offset

    t_index = jnp.repeat(t_scalar, grid_h * grid_w)
    h_index = jnp.tile(jnp.repeat(jnp.arange(grid_h, dtype=jnp.float32), grid_w), grid_t)
    w_index = jnp.tile(jnp.arange(grid_w, dtype=jnp.float32), grid_t * grid_h)
    return jnp.stack([t_index, h_index, w_index], axis=0)


def apply_interleaved_mrope(freqs: jnp.ndarray, rope_axes_dim: Sequence[int]) -> jnp.ndarray:
    """Reorganize the per-axis `(3, ..., head_dim // 2)` frequency tensor into
    the interleaved `[T,H,W,T,H,W,...,T,T,...]` layout described in this
    module's docstring. `freqs[a, ..., c] = position_ids[a] * inv_freq[c]`
    for axis `a` (0=T, 1=H, 2=W); this picks, for each output channel `c`,
    which axis's value to keep.
    """
    dim_h, dim_w = rope_axes_dim[1], rope_axes_dim[2]
    freqs_t = freqs[0]
    for axis_idx, offset, dim in ((1, 1, dim_h), (2, 2, dim_w)):
        length = dim * 3
        idx = jnp.arange(offset, length, 3)
        freqs_t = freqs_t.at[..., idx].set(freqs[axis_idx][..., idx])
    return freqs_t


def compute_cosmos3_mrope_cos_sin(
    position_ids: jnp.ndarray,  # (3, ..., N)
    head_dim: int,
    theta: float,
    rope_axes_dim: Sequence[int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds `(cos, sin)`, each `(..., N, head_dim)`, ready for `apply_mrope`.

    The position-id -> angle matmul is done in float32 throughout (the
    reference explicitly disables autocast for this step: under bf16, integer
    positions past 256 collapse onto the same value, silently corrupting the
    rotary table for any sequence longer than 256 tokens -- exactly the kind
    of bug that produces bounded-looking but wrong output without erroring).
    """
    inv_freq = _inv_freq(head_dim, theta)  # (head_dim // 2,)
    position_ids = position_ids.astype(jnp.float32)
    freqs = position_ids[..., None] * inv_freq  # (3, ..., N, head_dim // 2)
    freqs = apply_interleaved_mrope(freqs, rope_axes_dim)  # (..., N, head_dim // 2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (..., N, head_dim)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_mrope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """x: (B, N, num_heads, head_dim). cos/sin: (B, N, head_dim), float32."""
    orig_dtype = x.dtype
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    out = x.astype(jnp.float32) * cos + _rotate_half(x.astype(jnp.float32)) * sin
    return out.astype(orig_dtype)
