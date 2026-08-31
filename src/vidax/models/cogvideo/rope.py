"""CogVideoX's 3D rotary positional embedding (Flax/JAX).

A structural port of diffusers' `get_3d_rotary_pos_embed` /
`get_1d_rotary_pos_embed` / `apply_rotary_emb` (from
`diffusers/models/embeddings.py`) plus `get_resize_crop_region_for_grid`
(from `diffusers/pipelines/cogvideo/pipeline_cogvideox.py`), and the 3D
sinusoidal positional embedding (`get_3d_sincos_pos_embed`) used by the
non-rotary CogVideoX-2b / the I2V checkpoints instead of RoPE.

Deliberately self-contained (not shared with `vidax.core.rope3d`, which is
Wan2.1's three-way head-split convention, nor with `ltx_video.rope`): the
CogVideoX RoPE table is built **per head** over `attention_head_dim`, split
`t : h : w = D/4 : 3D/8 : 3D/8`, with a "consecutive-pair" rotate
(`x.reshape(..., -1, 2)` then `[-x_imag, x_real]`) -- diffusers'
`apply_rotary_emb(..., use_real_unbind_dim=-1)`. The table is applied to the
**visual tokens only**; the 226 text tokens that share the joint attention
sequence are left un-rotated (see diffusers `CogVideoXAttnProcessor2_0`).

Everything here is static per (resolution, frame-count) and cheap, so it's
computed once on host in numpy and passed into the model as constants
(mirrors how `examples/generate_ltx_video.py` precomputes its RoPE inputs).
"""
import math
from typing import Optional, Tuple

import numpy as np


def get_resize_crop_region_for_grid(src, tgt_width: int, tgt_height: int):
    """diffusers' `get_resize_crop_region_for_grid` -- the `crops_coords`
    passed to `get_3d_rotary_pos_embed` for the CogVideoX 1.0 (`linspace`)
    grid. `src = (grid_height, grid_width)`.
    """
    tw, th = tgt_width, tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def _get_1d_rotary_pos_embed(dim: int, pos: np.ndarray, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """diffusers `get_1d_rotary_pos_embed(dim, pos, theta, use_real=True,
    repeat_interleave_real=True)` -- returns `(cos, sin)`, each `(len(pos), dim)`.
    """
    assert dim % 2 == 0
    pos = np.asarray(pos, dtype=np.float32)
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))  # (dim/2,)
    freqs = np.outer(pos, freqs)  # (S, dim/2)
    cos = np.repeat(np.cos(freqs), 2, axis=1).astype(np.float32)  # (S, dim)
    sin = np.repeat(np.sin(freqs), 2, axis=1).astype(np.float32)
    return cos, sin


def _combine_time_height_width(freqs_t, freqs_h, freqs_w, T, gh, gw):
    # freqs_t: (T, dim_t), freqs_h: (gh, dim_h), freqs_w: (gw, dim_w)
    ft = np.broadcast_to(freqs_t[:, None, None, :], (T, gh, gw, freqs_t.shape[-1]))
    fh = np.broadcast_to(freqs_h[None, :, None, :], (T, gh, gw, freqs_h.shape[-1]))
    fw = np.broadcast_to(freqs_w[None, None, :, :], (T, gh, gw, freqs_w.shape[-1]))
    out = np.concatenate([ft, fh, fw], axis=-1)  # (T, gh, gw, dim_t+dim_h+dim_w)
    return out.reshape(T * gh * gw, -1)


def get_3d_rotary_pos_embed(
    embed_dim: int,
    crops_coords,
    grid_size: Tuple[int, int],
    temporal_size: int,
    theta: float = 10000.0,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """diffusers `get_3d_rotary_pos_embed` (`use_real=True`).

    Returns `(cos, sin)`, each `(temporal_size * grid_h * grid_w, embed_dim)`
    -- `embed_dim` is `attention_head_dim` (64).
    """
    grid_h, grid_w = grid_size
    if grid_type == "linspace":
        start, stop = crops_coords
        gh = np.linspace(start[0], stop[0] * (grid_h - 1) / grid_h, grid_h, dtype=np.float32)
        gw = np.linspace(start[1], stop[1] * (grid_w - 1) / grid_w, grid_w, dtype=np.float32)
        gt = np.linspace(
            0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, dtype=np.float32)
    elif grid_type == "slice":
        max_h, max_w = max_size
        gh = np.arange(max_h, dtype=np.float32)
        gw = np.arange(max_w, dtype=np.float32)
        gt = np.arange(temporal_size, dtype=np.float32)
    else:
        raise ValueError(f"Invalid `grid_type`: {grid_type!r}")

    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    t_cos, t_sin = _get_1d_rotary_pos_embed(dim_t, gt, theta=theta)
    h_cos, h_sin = _get_1d_rotary_pos_embed(dim_h, gh, theta=theta)
    w_cos, w_sin = _get_1d_rotary_pos_embed(dim_w, gw, theta=theta)

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_h], h_sin[:grid_h]
        w_cos, w_sin = w_cos[:grid_w], w_sin[:grid_w]

    cos = _combine_time_height_width(t_cos, h_cos, w_cos, temporal_size, grid_h, grid_w)
    sin = _combine_time_height_width(t_sin, h_sin, w_sin, temporal_size, grid_h, grid_w)
    return cos.astype(np.float32), sin.astype(np.float32)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_latent_frames: int,
    *,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: Optional[int] = None,
    attention_head_dim: int = 64,
    sample_height: int = 60,
    sample_width: int = 90,
    theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror of the diffusers pipeline's `_prepare_rotary_positional_embeddings`.
    `height` / `width` are in *pixels*, `num_latent_frames` is `latents.size(1)`
    (already padded up to a multiple of `patch_size_t` for 1.5). Returns
    `(cos, sin)`, each `(num_visual_tokens, attention_head_dim)`.
    """
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = sample_width // patch_size
    base_size_height = sample_height // patch_size

    if patch_size_t is None:
        crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height)
        return get_3d_rotary_pos_embed(
            attention_head_dim, crops_coords, (grid_height, grid_width),
            temporal_size=num_latent_frames, theta=theta, grid_type="linspace")
    base_num_frames = (num_latent_frames + patch_size_t - 1) // patch_size_t
    return get_3d_rotary_pos_embed(
        attention_head_dim, None, (grid_height, grid_width),
        temporal_size=base_num_frames, theta=theta, grid_type="slice",
        max_size=(base_size_height, base_size_width))


def apply_rotary_emb(x, cos, sin):
    """diffusers `apply_rotary_emb(x, (cos, sin), use_real=True,
    use_real_unbind_dim=-1)` for `x` of shape `(B, S, H, D)` (vidax's
    heads-third layout) and `cos`/`sin` of shape `(S, D)`.

    `x_real, x_imag = x.reshape(..., D/2, 2).unbind(-1)`;
    `x_rotated = stack([-x_imag, x_real], -1).flatten` ; `x*cos + x_rotated*sin`,
    computed in float32 then cast back (matches the reference).
    """
    import jax.numpy as jnp
    orig_dtype = x.dtype
    xf = x.astype(jnp.float32)
    cos = cos.astype(jnp.float32)[None, :, None, :]
    sin = sin.astype(jnp.float32)[None, :, None, :]
    b, s, h, d = xf.shape
    xr = xf.reshape(b, s, h, d // 2, 2)
    x_real = xr[..., 0]
    x_imag = xr[..., 1]
    x_rot = jnp.stack([-x_imag, x_real], axis=-1).reshape(b, s, h, d)
    return (xf * cos + x_rot * sin).astype(orig_dtype)


# --------------------------------------------------------------------------
# Non-rotary path: 3D sinusoidal positional embedding (CogVideoX-2b, I2V).
# Port of diffusers `get_3d_sincos_pos_embed` (output_type="pt" branch) +
# `get_1d_sincos_pos_embed_from_grid` / `get_2d_sincos_pos_embed_from_grid`.
# --------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000.0 ** omega
    pos = pos.reshape(-1).astype(np.float64)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb  # (M, embed_dim)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, embed_dim)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Tuple[int, int],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.875,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    """diffusers `get_3d_sincos_pos_embed` -- `spatial_size = (width, height)`
    (post-patch grid). Returns `(temporal_size * H * W, embed_dim)`.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = _get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)  # (H*W, 3D/4)

    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = _get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)  # (T, D/4)

    pos_embed_spatial = np.repeat(pos_embed_spatial[None], temporal_size, axis=0)  # (T, H*W, 3D/4)
    pos_embed_temporal = np.repeat(
        pos_embed_temporal[:, None], spatial_size[0] * spatial_size[1], axis=1)  # (T, H*W, D/4)

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # (T, H*W, D)
    return pos_embed.reshape(temporal_size * spatial_size[0] * spatial_size[1], embed_dim).astype(np.float32)
