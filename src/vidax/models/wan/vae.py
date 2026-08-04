"""Wan2.1 3D causal VAE, encoder and decoder (Flax/JAX).

A structural port of the reference PyTorch ``Encoder3d``/``Decoder3d``/
``WanVAE`` from Wan2.1-main/wan/modules/vae.py. The encoder
(``WanVAEEncoder``) is only needed for I2V's image-conditioning path (it
encodes the conditioning frame into latent space); T2V generation only
needs ``WanVAEDecoder``.

Wan2.1's causal VAE performs temporal up/downsampling via a frame-chunked
"streaming" algorithm rather than a single strided convolution over the
whole clip: latent frames are decoded one at a time, and each
temporal-upsampling ``CausalConv3d`` reads a small cache of edge frames left
behind by the previous chunk. This is not just a memory optimization — the
reference model's *first* frame is causally special-cased (no temporal
upsampling contribution), so a naive whole-tensor forward pass produces
different output. This implementation reproduces that chunked algorithm
exactly (see ``WanVAEDecoder.__call__``), threading a plain Python list of
cached edge-frame tensors through the decoder the same way the reference's
``feat_cache`` does.

Callers should invoke this module's `.apply()` directly, *without* wrapping
the call in `jax.jit`: the chunk loop lives inside `__call__`, so jit-ing it
traces/unrolls all ~20 chunks into a single HLO program, and their
intermediate activations can end up needing to coexist in that one
program's memory footprint instead of being freed between chunks -- this is
what caused whole-video decode to OOM at full resolution even after DiT
sampling (a much bigger model) succeeded. Running eagerly is simple and
correct at some dispatch-overhead cost; recovering that speed would mean
either jit-ing a single-chunk step function and calling it from a Python
loop (mirroring the DiT sampling loop's `single_step` pattern in
`examples/generate_wan2_1.py`) or extending the decoder to
`vidax.core.sharding`'s tensor-parallel scheme (not currently done -- the
VAE's convolutions aren't attention-shaped, so it would need its own
channel-parallel sharding rules rather than reusing `shard_wan_params`),
neither of which is implemented here yet.
"""
from typing import List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from vidax.core.attention import RMSNorm, dot_product_attention

CACHE_T = 2

# Per-channel latent mean/std used to normalize/denormalize Wan2.1's 16-channel
# latent space (Wan2.1-main/wan/modules/vae.py: WanVAE.__init__).
VAE_LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
VAE_LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


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


def _cached_causal_conv3d(
    x: jnp.ndarray, features: int, kernel_size: Sequence[int], name: str,
    cache_list: Optional[List], idx_ref: Optional[List[int]],
    padding: Sequence[int] = (1, 1, 1),
) -> jnp.ndarray:
    """causal_conv3d call with the CACHE_T edge-frame caching used by the
    plain 3x3x3 convs throughout the decoder (residual blocks, head, etc).
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
        h = _cached_causal_conv3d(h, self.out_dim, (3, 3, 3), "conv1", cache_list, idx_ref)
        h = RMSNorm(self.out_dim, eps=self.eps, name="norm2")(h)
        h = nn.silu(h)
        h = _cached_causal_conv3d(h, self.out_dim, (3, 3, 3), "conv2", cache_list, idx_ref)
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
    """Spatial (2d) or spatio-temporal (3d) 2x resampling block."""
    dim: int
    mode: str  # 'none' | 'upsample2d' | 'upsample3d' | 'downsample2d' | 'downsample3d'
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        b, t, h, w, c = x.shape

        if self.mode == "upsample3d" and cache_list is not None:
            idx = idx_ref[0]
            if cache_list[idx] is None:
                cache_list[idx] = "Rep"
                idx_ref[0] += 1
            else:
                cache_x = x[:, -CACHE_T:]
                if cache_x.shape[1] < 2 and cache_list[idx] not in (None, "Rep"):
                    cache_x = jnp.concatenate([cache_list[idx][:, -1:], cache_x], axis=1)
                if cache_x.shape[1] < 2 and cache_list[idx] == "Rep":
                    cache_x = jnp.concatenate([jnp.zeros_like(cache_x), cache_x], axis=1)

                time_cache = None if cache_list[idx] == "Rep" else cache_list[idx]
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
            y = nn.Conv(self.dim // 2, (3, 3), padding=((1, 1), (1, 1)),
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


class Decoder3d(nn.Module):
    """Mirrors the reference ``Decoder3d``: conv1 -> middle -> upsamples -> head."""
    dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_upsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        dims = [self.dim * u for u in (self.dim_mult[-1],) + tuple(self.dim_mult[::-1])]
        scale = 1.0 / 2 ** (len(self.dim_mult) - 2)

        x = _cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

        x = ResidualBlock(dims[0], dims[0], self.eps, name="middle_0")(x, cache_list, idx_ref)
        x = AttentionBlock(dims[0], self.eps, name="middle_1")(x)
        x = ResidualBlock(dims[0], dims[0], self.eps, name="middle_2")(x, cache_list, idx_ref)

        layer_idx = 0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            for _ in range(self.num_res_blocks + 1):
                x = ResidualBlock(in_dim, out_dim, self.eps,
                                   name=f"upsamples_{layer_idx}")(x, cache_list, idx_ref)
                layer_idx += 1
                if scale in self.attn_scales:
                    x = AttentionBlock(out_dim, self.eps, name=f"upsamples_{layer_idx}")(x)
                    layer_idx += 1
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                mode = "upsample3d" if self.temperal_upsample[i] else "upsample2d"
                x = Resample(out_dim, mode, self.eps,
                             name=f"upsamples_{layer_idx}")(x, cache_list, idx_ref)
                layer_idx += 1
                scale *= 2.0

        x = RMSNorm(dims[-1], eps=self.eps, name="head_0")(x)
        x = nn.silu(x)
        x = _cached_causal_conv3d(x, 3, (3, 3, 3), "head_2", cache_list, idx_ref)
        return x


def _count_causal_convs(decoder: Decoder3d) -> int:
    """Number of CausalConv3d calls per forward pass, for cache-list sizing.

    conv1 + (middle_0, middle_2 residual blocks: 2 convs each) + per upsample
    stage residual blocks (2 convs each) + per upsample3d Resample (1
    time_conv) + per downsample3d Resample (1 time_conv, unused in decode)
    + head's final conv.
    """
    count = 1  # conv1
    count += 2 + 2  # middle_0, middle_2 (middle_1 is attention, no conv)
    for i, dim_mult in enumerate(decoder.dim_mult):
        for _ in range(decoder.num_res_blocks + 1):
            count += 2
        if i != len(decoder.dim_mult) - 1 and decoder.temperal_upsample[i]:
            count += 1
    count += 1  # head_2
    return count


class WanVAEDecoder(nn.Module):
    """Top-level Wan2.1 VAE decoder: latent normalization + conv2 + Decoder3d."""
    dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_upsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: (B, T_latent, H_latent, W_latent, z_dim) latents.

        Returns:
            (B, T_latent upsampled to frames, H*8, W*8, 3) RGB frames in [-1, 1].
        """
        # The reference computes this as `z / scale[1] + scale[0]` where
        # `scale = [mean, 1/std]` -- i.e. `z / (1/std) + mean == z * std +
        # mean`. Written directly (`z * std + mean`) to avoid the trap of
        # transcribing `z / scale[1]` as `z / std`.
        mean = jnp.asarray(VAE_LATENT_MEAN, dtype=z.dtype)
        std = jnp.asarray(VAE_LATENT_STD, dtype=z.dtype)
        z = z * std + mean

        z = causal_conv3d(z, self.z_dim, (1, 1, 1), "conv2", padding=(0, 0, 0))

        decoder = Decoder3d(
            self.dim, self.z_dim, self.dim_mult, self.num_res_blocks,
            self.attn_scales, self.temperal_upsample, self.eps, name="decoder")

        num_convs = _count_causal_convs(decoder)
        cache_list = [None] * num_convs

        # The reference decodes one latent frame at a time, threading a cache
        # of edge frames between iterations to reproduce causal streaming
        # behavior exactly (see module docstring).
        outputs = []
        for i in range(z.shape[1]):
            idx_ref = [0]
            outputs.append(decoder(z[:, i:i + 1], cache_list, idx_ref))
        return jnp.concatenate(outputs, axis=1)


class Encoder3d(nn.Module):
    """Mirrors the reference ``Encoder3d``: conv1 -> downsamples -> middle -> head."""
    dim: int = 96
    z_dim: int = 32  # note: this is the *head's* output width (2*z_dim, for mu/log_var).
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_downsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        dims = [self.dim * u for u in (1,) + tuple(self.dim_mult)]
        scale = 1.0

        x = _cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

        layer_idx = 0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(self.num_res_blocks):
                x = ResidualBlock(in_dim, out_dim, self.eps,
                                   name=f"downsamples_{layer_idx}")(x, cache_list, idx_ref)
                layer_idx += 1
                if scale in self.attn_scales:
                    x = AttentionBlock(out_dim, self.eps, name=f"downsamples_{layer_idx}")(x)
                    layer_idx += 1
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                mode = "downsample3d" if self.temperal_downsample[i] else "downsample2d"
                x = Resample(out_dim, mode, self.eps,
                             name=f"downsamples_{layer_idx}")(x, cache_list, idx_ref)
                layer_idx += 1
                scale /= 2.0

        out_dim = dims[-1]
        x = ResidualBlock(out_dim, out_dim, self.eps, name="middle_0")(x, cache_list, idx_ref)
        x = AttentionBlock(out_dim, self.eps, name="middle_1")(x)
        x = ResidualBlock(out_dim, out_dim, self.eps, name="middle_2")(x, cache_list, idx_ref)

        x = RMSNorm(out_dim, eps=self.eps, name="head_0")(x)
        x = nn.silu(x)
        x = _cached_causal_conv3d(x, self.z_dim, (3, 3, 3), "head_2", cache_list, idx_ref)
        return x


def _count_causal_convs_encoder(encoder: Encoder3d) -> int:
    """Number of CausalConv3d calls per forward pass, for cache-list sizing
    (see `_count_causal_convs`'s docstring for the decoder's mirror-image
    version of this).
    """
    count = 1  # conv1
    for i in range(len(encoder.dim_mult)):
        count += 2 * encoder.num_res_blocks
        if i != len(encoder.dim_mult) - 1 and encoder.temperal_downsample[i]:
            count += 1  # downsample3d's time_conv
    count += 2 + 2  # middle_0, middle_2 (middle_1 is attention, no conv)
    count += 1  # head_2
    return count


class WanVAEEncoder(nn.Module):
    """Top-level Wan2.1 VAE encoder: Encoder3d -> conv1 (mu/log_var) -> normalize.

    Returns the normalized latent *mean* only (deterministic, no
    reparameterization sampling) -- this matches the reference's actual
    usage for I2V conditioning (`WanVAE.encode(...)`  never samples; it's
    only `WanVAE_.sample()`, used for VAE training/eval, that does).
    """
    dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_downsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, T, H, W, 3) RGB video in [-1, 1], T = 1 + 4k.

        Returns:
            (B, T_latent, H/8, W/8, z_dim) normalized latent means, where
            T_latent = 1 + (T - 1) // 4.
        """
        encoder = Encoder3d(
            self.dim, self.z_dim * 2, self.dim_mult, self.num_res_blocks,
            self.attn_scales, self.temperal_downsample, self.eps, name="encoder")

        num_convs = _count_causal_convs_encoder(encoder)
        cache_list = [None] * num_convs

        # Matches the reference's `WanVAE_.encode`: the first chunk is a
        # single frame, then chunks of 4 frames -- the same "1 + 4k" causal
        # grouping the decoder inverts (see WanVAEDecoder's docstring).
        t = x.shape[1]
        bounds = [(0, 1)] + [(1 + 4 * i, 1 + 4 * (i + 1)) for i in range((t - 1) // 4)]

        outputs = []
        for start, end in bounds:
            idx_ref = [0]
            outputs.append(encoder(x[:, start:end], cache_list, idx_ref))
        out = jnp.concatenate(outputs, axis=1)

        moments = causal_conv3d(out, self.z_dim * 2, (1, 1, 1), "conv1", padding=(0, 0, 0))
        mu, _log_var = jnp.split(moments, 2, axis=-1)

        mean = jnp.asarray(VAE_LATENT_MEAN, dtype=mu.dtype)
        std = jnp.asarray(VAE_LATENT_STD, dtype=mu.dtype)
        return (mu - mean) / std
