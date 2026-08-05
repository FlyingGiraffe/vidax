"""Building blocks shared by every Wan causal-VAE version (Wan2.1, Wan2.2, ...).

These are ports of the reference PyTorch ``CausalConv3d``/``RMS_norm``/
``ResidualBlock``/``AttentionBlock``/``Resample`` classes, which are
byte-for-byte identical (module-docstring/formatting aside) between
Wan2.1-main/wan/modules/vae.py and Wan2.2-main/wan/modules/vae2_2.py. Each
VAE version's own module (``wan2_1/vae.py``, ``wan2_2/vae.py``) wires these
into its version-specific ``Encoder3d``/``Decoder3d`` (which do differ:
Wan2.2 adds ``AvgDown3D``/``DupUp3D``/patchify for its higher-compression
latent space).
"""
from typing import List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from vidax.core.attention import RMSNorm, dot_product_attention

CACHE_T = 2


def causal_conv3d(
    x: jnp.ndarray, features: int, kernel_size: Sequence[int], name: str,
    cache: Optional[jnp.ndarray] = None,
    strides: Sequence[int] = (1, 1, 1),
    padding: Sequence[int] = (0, 0, 0),
) -> jnp.ndarray:
    """A 3D conv, causally zero-padded on time (front-only) given explicit
    (pad_t, pad_h, pad_w), matching the reference's ``CausalConv3d``.

    Must be called from within an enclosing ``nn.compact`` scope (it creates
    a bare ``nn.Conv`` at that scope under ``name``, so the parameter tree
    stays flat: ``.../{name}/kernel`` rather than an extra nesting level).

    When ``cache`` (up to ``CACHE_T`` previous frames) is supplied, it is
    prepended along time before padding, and the amount of zero-padding is
    reduced accordingly — this is how causal continuity across chunks is
    achieved without recomputing earlier frames.
    """
    pad_t, pad_h, pad_w = padding
    front_pad_t = 2 * pad_t
    if cache is not None and front_pad_t > 0:
        x = jnp.concatenate([cache, x], axis=1)
        front_pad_t -= cache.shape[1]
    x = jnp.pad(x, ((0, 0), (front_pad_t, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    return nn.Conv(features, tuple(kernel_size), strides=tuple(strides),
                    padding="VALID", name=name)(x)


def cached_causal_conv3d(
    x: jnp.ndarray, features: int, kernel_size: Sequence[int], name: str,
    cache_list: Optional[List], idx_ref: Optional[List[int]],
    padding: Sequence[int] = (1, 1, 1),
) -> jnp.ndarray:
    """causal_conv3d call with the CACHE_T edge-frame caching used by the
    plain 3x3x3 convs throughout the encoder/decoder (residual blocks, head,
    etc).
    """
    if cache_list is None:
        return causal_conv3d(x, features, kernel_size, name, padding=padding)
    idx = idx_ref[0]
    cache_x = x[:, -CACHE_T:]
    if cache_x.shape[1] < 2 and cache_list[idx] is not None:
        cache_x = jnp.concatenate([cache_list[idx][:, -1:], cache_x], axis=1)
    out = causal_conv3d(x, features, kernel_size, name, cache=cache_list[idx], padding=padding)
    cache_list[idx] = cache_x
    idx_ref[0] += 1
    return out


class ResidualBlock(nn.Module):
    in_dim: int
    out_dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        shortcut = x
        if self.in_dim != self.out_dim:
            shortcut = causal_conv3d(
                x, self.out_dim, (1, 1, 1), "shortcut", padding=(0, 0, 0))

        h = RMSNorm(self.in_dim, eps=self.eps, name="norm1")(x)
        h = nn.silu(h)
        h = cached_causal_conv3d(h, self.out_dim, (3, 3, 3), "conv1", cache_list, idx_ref)
        h = RMSNorm(self.out_dim, eps=self.eps, name="norm2")(h)
        h = nn.silu(h)
        h = cached_causal_conv3d(h, self.out_dim, (3, 3, 3), "conv2", cache_list, idx_ref)
        return shortcut + h


class AttentionBlock(nn.Module):
    """Single-head causal self-attention over the spatial grid of each frame."""
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, t, h, w, c = x.shape
        identity = x
        y = RMSNorm(self.dim, eps=self.eps, name="norm")(x)
        y = y.reshape(b * t, h, w, c)
        qkv = nn.Conv(self.dim * 3, (1, 1), name="to_qkv")(y)
        qkv = qkv.reshape(b * t, h * w, 3, c)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        q = q[:, :, None, :]
        k = k[:, :, None, :]
        v = v[:, :, None, :]
        out = dot_product_attention(q, k, v).reshape(b * t, h, w, c)
        out = nn.Conv(self.dim, (1, 1), name="proj",
                       kernel_init=nn.initializers.zeros)(out)
        return identity + out.reshape(b, t, h, w, c)


class Resample(nn.Module):
    """Spatial (2d) or spatio-temporal (3d) 2x resampling block.

    ``halve_upsample_channels`` controls the upsample path's output width:
    Wan2.1's ``Resample`` projects to ``dim // 2`` there (True, the
    default), while Wan2.2's otherwise-identical ``Resample`` keeps the full
    ``dim`` (the reference's own upsample conv literally has a commented-out
    ``dim // 2`` version of the same line right next to the ``dim, dim`` one
    it now uses) -- everything else about the two versions' ``Resample`` is
    byte-for-byte identical.
    """
    dim: int
    mode: str  # 'none' | 'upsample2d' | 'upsample3d' | 'downsample2d' | 'downsample3d'
    eps: float = 1e-6
    halve_upsample_channels: bool = True

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        b, t, h, w, c = x.shape

        if self.mode == "upsample3d" and cache_list is not None:
            idx = idx_ref[0]
            if cache_list[idx] is None:
                # First-ever call for this cache slot (the very first decoded
                # chunk): matches the reference's "Rep" placeholder -- skip
                # the temporal upsample entirely this chunk (no real history
                # yet). Stored as a real, zero-length-along-time array
                # (rather than the reference's sentinel string) so cache_list
                # stays a plain pytree of arrays throughout -- None/an
                # ordinary array are both JIT-compatible pytree structures,
                # a Python string is not, which matters for jit-ing the
                # per-chunk decode step (see `vidax.models.wan.wan2_2.vae`'s
                # `decode_chunk`).
                cache_list[idx] = jnp.zeros((b, 0, h, w, c), dtype=x.dtype)
                idx_ref[0] += 1
            else:
                cache_x = x[:, -CACHE_T:]
                is_rep = cache_list[idx].shape[1] == 0
                if cache_x.shape[1] < 2 and not is_rep:
                    cache_x = jnp.concatenate([cache_list[idx][:, -1:], cache_x], axis=1)
                if cache_x.shape[1] < 2 and is_rep:
                    cache_x = jnp.concatenate([jnp.zeros_like(cache_x), cache_x], axis=1)

                time_cache = None if is_rep else cache_list[idx]
                x = causal_conv3d(x, self.dim * 2, (3, 1, 1), "time_conv",
                                   cache=time_cache, padding=(1, 0, 0))
                cache_list[idx] = cache_x
                idx_ref[0] += 1

                x = x.reshape(b, t, h, w, 2, c)
                x = jnp.transpose(x, (0, 1, 4, 2, 3, 5))
                x = x.reshape(b, t * 2, h, w, c)

        t_cur = x.shape[1]
        y = x.reshape(b * t_cur, h, w, c)
        if self.mode in ("upsample2d", "upsample3d"):
            y = jax.image.resize(y, (b * t_cur, h * 2, w * 2, c), method="nearest")
            out_dim = self.dim // 2 if self.halve_upsample_channels else self.dim
            y = nn.Conv(out_dim, (3, 3), padding=((1, 1), (1, 1)),
                        name="resample_1")(y)
        elif self.mode in ("downsample2d", "downsample3d"):
            y = jnp.pad(y, ((0, 0), (0, 1), (0, 1), (0, 0)))
            y = nn.Conv(self.dim, (3, 3), strides=(2, 2), padding="VALID",
                        name="resample_1")(y)
        h2, w2, c2 = y.shape[1], y.shape[2], y.shape[-1]
        x = y.reshape(b, t_cur, h2, w2, c2)

        if self.mode == "downsample3d" and cache_list is not None:
            idx = idx_ref[0]
            if cache_list[idx] is None:
                cache_list[idx] = x
                idx_ref[0] += 1
            else:
                cache_x = x[:, -1:]
                x = jnp.concatenate([cache_list[idx][:, -1:], x], axis=1)
                x = causal_conv3d(x, self.dim, (3, 1, 1), "time_conv",
                                   strides=(2, 1, 1), padding=(0, 0, 0))
                cache_list[idx] = cache_x
                idx_ref[0] += 1
        return x
