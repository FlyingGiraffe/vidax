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

`WanVAEDecoder.decode_chunk` (wrapped in `jax.jit`, called from a plain
Python loop over latent frames -- see `examples/generate_wan2_1_t2v.py`) is
the production-path entry point, *not* calling this module's `.apply()`
directly on the whole video (`__call__`, kept only as a simple, fully-eager
convenience path e.g. for this repo's own small-scale tests). Calling the
whole decoder eagerly, one op at a time -- what `__call__`'s Python chunk
loop does -- means every individual op inside `Decoder3d` triggers its own
separate XLA compilation; this is tolerable at Wan2.1's default resolution
(384 channels at the widest, small enough that eager compilation stays
fast), but does not scale to higher resolutions, and stops being tolerable
entirely for Wan2.2's much wider decoder (1024 channels) at its one
supported resolution -- see `vidax.models.wan.wan2_2.vae.WanVAEDecoder`'s
`decode_chunk` docstring, which hit this for real. `decode_chunk` compiles
the *whole* per-frame computation as one fused program instead, at most
twice total (the cache state's pytree structure stabilizes after the first
frame here, since Wan2.1's `Decoder3d` has no `DupUp3D`-style `first_chunk`
distinction the way Wan2.2's does) rather than once per op per frame.

Separately: jit-ing the *whole* per-chunk loop in one call (i.e. `jax.jit`
around `__call__` itself, unrolling every one of the ~20 chunks into a
single HLO program) is a different, additional problem from the above --
each chunk's intermediate activations would need to coexist in that one
program's memory footprint instead of being freed between chunks, which is
what caused whole-video decode to OOM at full resolution even after DiT
sampling (a much bigger model) succeeded. `decode_chunk` avoids this too,
since only *one* chunk's computation is ever inside a given jit call.
"""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from vidax.core.attention import RMSNorm
from vidax.models.wan.common.vae_layers import (
    AttentionBlock, ResidualBlock, Resample, cached_causal_conv3d,
)

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

        x = cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

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
        x = cached_causal_conv3d(x, 3, (3, 3, 3), "head_2", cache_list, idx_ref)
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
    """Top-level Wan2.1 VAE decoder: latent normalization + conv2 + Decoder3d.

    Uses `setup()` (not `@nn.compact`) specifically so `decode_chunk` can be
    called on its own, independent of `__call__` -- see that method's
    docstring for why large-resolution callers should use it (wrapped in
    `jax.jit`) instead of calling this module directly.
    """
    dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_upsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    def setup(self):
        # `conv2` is kernel-1 (no causal padding/cache needed at all -- see
        # `pre_process`), so it's defined directly rather than through the
        # `causal_conv3d` helper (which can only create its inline `nn.Conv`
        # from a `@nn.compact` method, not one of a `setup()`-based module
        # like this one).
        self.conv2 = nn.Conv(self.z_dim, (1, 1, 1), padding="VALID", name="conv2")
        self.decoder = Decoder3d(
            self.dim, self.z_dim, self.dim_mult, self.num_res_blocks,
            self.attn_scales, self.temperal_upsample, self.eps, name="decoder")

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: (B, T_latent, H_latent, W_latent, z_dim) latents.

        Returns:
            (B, T_latent upsampled to frames, H*8, W*8, 3) RGB frames in [-1, 1].
        """
        x = self.pre_process(z)
        cache_list = [None] * _count_causal_convs(self.decoder)

        # The reference decodes one latent frame at a time, threading a cache
        # of edge frames between iterations to reproduce causal streaming
        # behavior exactly (see module docstring).
        outputs = []
        for i in range(x.shape[1]):
            out_chunk, cache_list = self.decode_chunk(x[:, i:i + 1], cache_list)
            outputs.append(out_chunk)
        return jnp.concatenate(outputs, axis=1)

    def pre_process(self, z: jnp.ndarray) -> jnp.ndarray:
        """Denormalizes and runs `conv2` on the *full* (unchunked) latent
        tensor -- safe and cheap to do all at once (kernel-1, no temporal
        receptive field to worry about across chunk boundaries).

        The reference computes denormalization as `z / scale[1] + scale[0]`
        where `scale = [mean, 1/std]` -- i.e. `z / (1/std) + mean == z * std
        + mean`. Written directly (`z * std + mean`) to avoid the trap of
        transcribing `z / scale[1]` as `z / std`.
        """
        mean = jnp.asarray(VAE_LATENT_MEAN, dtype=z.dtype)
        std = jnp.asarray(VAE_LATENT_STD, dtype=z.dtype)
        z = z * std + mean
        return self.conv2(z)

    def decode_chunk(self, x_chunk: jnp.ndarray, cache_list: list):
        """Decodes one latent frame (`x_chunk`, from `pre_process`'s output),
        given and returning the running cache state.

        This -- not `__call__` -- is what production callers should use,
        wrapped in `jax.jit` (e.g. `jax.jit(lambda params, x, c: model.apply(
        params, x, c, method=model.decode_chunk))`), called from a plain
        Python loop over latent frames (see `examples/generate_wan2_1_t2v.py`
        and `vidax.models.wan.wan2_2.vae.WanVAEDecoder.decode_chunk`'s
        identically-motivated docstring for the full reasoning).

        Args:
            x_chunk: (B, 1, H, W, dim) -- one latent frame, post-`pre_process`.
            cache_list: Running cache state -- `[None] * _count_causal_convs(
                self.decoder)` for the very first call, whatever this method
                last returned otherwise.

        Returns:
            (out_chunk, new_cache_list).
        """
        idx_ref = [0]
        out_chunk = self.decoder(x_chunk, cache_list, idx_ref)
        return out_chunk, cache_list


class Encoder3d(nn.Module):
    """Mirrors the reference ``Encoder3d``: conv1 -> downsamples -> middle -> head."""
    dim: int = 96
    z_dim: int = 32  # note: this is the *head's* output width (2*z_dim, for mu/log_var).
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_downsample: Sequence[bool] = (False, True, True)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        dims = [self.dim * u for u in (1,) + tuple(self.dim_mult)]
        scale = 1.0

        x = cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

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
        x = cached_causal_conv3d(x, self.z_dim, (3, 3, 3), "head_2", cache_list, idx_ref)
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

    Uses `setup()` (not `@nn.compact`) so `encode_chunk` can be called on
    its own, independent of `__call__` -- same reasoning as
    `WanVAEDecoder.decode_chunk`'s docstring (this repo's own i2v pipeline
    only ever encodes a single conditioning frame today, so this mattered
    far less in practice than the decoder's ~20-chunk loop did, but the
    same inefficiency is there for anyone encoding a longer clip, or at
    higher resolution).
    """
    dim: int = 96
    z_dim: int = 16
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Sequence[float] = ()
    temperal_downsample: Sequence[bool] = (False, True, True)
    eps: float = 1e-6

    def setup(self):
        # `conv1` is kernel-1 (no causal padding/cache needed at all -- see
        # `post_process`), so it's defined directly rather than through the
        # `causal_conv3d` helper (which can only create its inline `nn.Conv`
        # from a `@nn.compact` method, not one of a `setup()`-based module
        # like this one).
        self.conv1 = nn.Conv(self.z_dim * 2, (1, 1, 1), padding="VALID", name="conv1")
        self.encoder = Encoder3d(
            self.dim, self.z_dim * 2, self.dim_mult, self.num_res_blocks,
            self.attn_scales, self.temperal_downsample, self.eps, name="encoder")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, T, H, W, 3) RGB video in [-1, 1], T = 1 + 4k.

        Returns:
            (B, T_latent, H/8, W/8, z_dim) normalized latent means, where
            T_latent = 1 + (T - 1) // 4.
        """
        # Matches the reference's `WanVAE_.encode`: the first chunk is a
        # single frame, then chunks of 4 frames -- the same "1 + 4k" causal
        # grouping the decoder inverts (see WanVAEDecoder's docstring).
        t = x.shape[1]
        bounds = [(0, 1)] + [(1 + 4 * i, 1 + 4 * (i + 1)) for i in range((t - 1) // 4)]

        cache_list = [None] * _count_causal_convs_encoder(self.encoder)
        outputs = []
        for start, end in bounds:
            out_chunk, cache_list = self.encode_chunk(x[:, start:end], cache_list)
            outputs.append(out_chunk)
        out = jnp.concatenate(outputs, axis=1)
        return self.post_process(out)

    def encode_chunk(self, x_chunk: jnp.ndarray, cache_list: list):
        """Encodes one chunk (the first call gets 1 frame, every subsequent
        call gets 4 -- see `__call__`'s `bounds`), given and returning the
        running cache state.

        This -- not `__call__` -- is what production callers encoding more
        than a single frame should use, wrapped in `jax.jit` and called from
        a plain Python loop -- see `WanVAEDecoder.decode_chunk`'s docstring
        for the full reasoning (identical here, just for the encoder).

        Args:
            x_chunk: (B, 1 or 4, H, W, 3) -- one chunk of RGB video.
            cache_list: Running cache state -- `[None] *
                _count_causal_convs_encoder(self.encoder)` for the very
                first call, whatever this method last returned otherwise.

        Returns:
            (out_chunk, new_cache_list).
        """
        idx_ref = [0]
        out_chunk = self.encoder(x_chunk, cache_list, idx_ref)
        return out_chunk, cache_list

    def post_process(self, out: jnp.ndarray) -> jnp.ndarray:
        """Runs `conv1` (mu/log_var projection) on the *full* (unchunked,
        already-concatenated) encoder output, then normalizes -- safe and
        cheap to do all at once (kernel-1, no temporal receptive field).
        """
        moments = self.conv1(out)
        mu, _log_var = jnp.split(moments, 2, axis=-1)

        mean = jnp.asarray(VAE_LATENT_MEAN, dtype=mu.dtype)
        std = jnp.asarray(VAE_LATENT_STD, dtype=mu.dtype)
        return (mu - mean) / std
