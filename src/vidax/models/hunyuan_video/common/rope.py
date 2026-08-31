"""3D axial RoPE for HunyuanVideo / HunyuanVideo-1.5.

Reference: ``hyvideo/models/transformers/modules/posemb_layers.py``
(``get_nd_rotary_pos_embed`` / ``get_1d_rotary_pos_embed`` / ``rotate_half``
/ ``apply_rotary_emb``, identical file shared verbatim between the 1.0 and
1.5 reference repos).

**Important correction vs. this module's own early scoping notes**: the
reference's ``rotate_half`` is *not* the GPT-NeoX "rotate-half" (split the
head dim into two contiguous halves) convention despite the name. It
reshapes the last dim into ``(D//2, 2)`` pairs (``x.reshape(*shape[:-1], -1,
2)``) and rotates each *adjacent* pair -- the same interleaved-pair
convention ``vidax.core.rope3d`` already implements for Wan. Concretely,
with ``x1 = x[..., 0::2]``, ``x2 = x[..., 1::2]``:

  reference: ``xq_out = xq*cos + rotate_half(xq)*sin``, where
  ``rotate_half(xq)`` interleaves ``[-x2, x1, -x2', x1', ...]``, giving
  ``out_even = x1*cos - x2*sin``, ``out_odd = x2*cos + x1*sin``.

This is exactly ``vidax.core.rope3d.apply_rope3d``'s formula (also
``out1 = x1*cos - x2*sin``, ``out2 = x2*cos + x1*sin``), so that function is
reused directly here -- only the per-axis frequency construction differs
(HunyuanVideo splits the head dim unevenly across T/H/W via
``rope_dim_list`` and integer 0-based grid positions per patchified axis,
vs. Wan's ``c - 2*(c//3)`` split). The reference's ``get_1d_rotary_pos_embed``
also returns full-head-dim cos/sin via ``repeat_interleave(2, dim=1)`` (each
frequency duplicated into its own adjacent pair) -- ``vidax.core.rope3d``'s
half-width ``cos``/``sin`` (one value per pair, consumed by
``apply_rope3d``'s ``x1*cos``/``x2*cos``) is the same value per pair, just
packed at half width; mathematically identical, so ``_axis_freqs`` (built
for exactly this per-axis "half-width cos/sin" shape) is reused unmodified
for each axis.

RoPE is applied to image (patchified latent) tokens only -- text/glyph/
vision tokens carry no positional rotation (see ``dit_layers.py``).
"""
from typing import Sequence, Tuple

import jax.numpy as jnp

from vidax.core.rope3d import _axis_freqs, apply_rope3d  # noqa: F401  (re-exported)

__all__ = ["create_hunyuan_rope3d_freqs", "apply_rope3d"]


def create_hunyuan_rope3d_freqs(
    t: int, h: int, w: int,
    rope_dim_list: Sequence[int] = (16, 56, 56),
    theta: float = 256.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds per-position (cos, sin) rotation angles for a (T, H, W) grid.

    Mirrors ``get_nd_rotary_pos_embed(rope_dim_list, (t, h, w), theta=256.0,
    use_real=True)`` in the reference: each axis gets its own slice of
    ``rope_dim_list[i]`` head-dim channels (summing to ``head_dim``), with
    plain 0-based integer positions along that axis (the reference's
    ``get_meshgrid_nd`` default-start path, ``linspace(0, n, n+1)[:n]`` ==
    ``arange(n)``), broadcast over the other two axes and flattened in
    (T, H, W) row-major order -- matching ``torch.meshgrid(..., indexing=
    "ij")`` followed by ``grid[i].reshape(-1)``, i.e. the same token order
    patchify/unpatchify already use.

    Args:
        t, h, w: Patchified grid sizes along time, height, width.
        rope_dim_list: Per-axis (T, H, W) share of ``head_dim``; must sum to
            the attention head_dim. HunyuanVideo-1.5's default is
            ``(16, 56, 56)`` (head_dim=128).
        theta: RoPE base frequency (HunyuanVideo default 256.0, notably
            lower than the usual 10000.0 LLM default).

    Returns:
        (cos, sin), each of shape (1, t*h*w, 1, head_dim // 2), ready to
        broadcast against a (B, S, num_heads, head_dim) tensor via
        ``apply_rope3d``.
    """
    dim_t, dim_h, dim_w = (d // 2 for d in rope_dim_list)

    cos_t, sin_t = _axis_freqs(t, rope_dim_list[0], theta)
    cos_h, sin_h = _axis_freqs(h, rope_dim_list[1], theta)
    cos_w, sin_w = _axis_freqs(w, rope_dim_list[2], theta)

    def broadcast_grid(freqs_t, freqs_h, freqs_w):
        gt = jnp.broadcast_to(freqs_t[:, None, None, :], (t, h, w, dim_t))
        gh = jnp.broadcast_to(freqs_h[None, :, None, :], (t, h, w, dim_h))
        gw = jnp.broadcast_to(freqs_w[None, None, :, :], (t, h, w, dim_w))
        return jnp.concatenate([gt, gh, gw], axis=-1).reshape(t * h * w, dim_t + dim_h + dim_w)

    cos = broadcast_grid(cos_t, cos_h, cos_w)
    sin = broadcast_grid(sin_t, sin_h, sin_w)
    return cos[None, :, None, :], sin[None, :, None, :]
