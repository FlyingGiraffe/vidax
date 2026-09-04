"""Wan2.2 3D causal VAE, encoder and decoder (Flax/JAX).

A structural port of the reference PyTorch ``Encoder3d``/``Decoder3d``/
``WanVAE_`` from Wan2.2-main/wan/modules/vae2_2.py. Wan2.2 ships a genuinely
new VAE architecture (not just a config change from Wan2.1's): pixels are
first rearranged 2x2 spatially into extra channels (``patchify``, undone by
``unpatchify`` on the way out) before ever reaching the encoder/decoder, and
each resolution stage adds a parameter-free "average-pool"/"repeat-and-
reshuffle" shortcut path (``avg_down3d``/``dup_up3d``) alongside the usual
residual-block main path (`Down_ResidualBlock`/`Up_ResidualBlock` in the
reference). `causal_conv3d`/`cached_causal_conv3d`/`ResidualBlock`/
`AttentionBlock`/`Resample` themselves are unchanged from Wan2.1 (verified
byte-for-byte identical against the reference source, module-docstring/
formatting aside) and are imported from
``vidax.models.wan.common.vae_layers``; only the new per-stage wiring and
the patchify/AvgDown3D/DupUp3D pieces are new here.

`avg_down3d`/`dup_up3d`/`patchify`/`unpatchify` are transpositions of the
reference's channel-first (`B,C,T,H,W`) einops/reshape sequences into this
codebase's channels-last (`B,T,H,W,C`) layout; each was checked for exact
numerical equivalence against a direct numpy re-implementation of the
reference's own reshape/transpose sequence (not just shape-compatibility)
before being written here.

Callers should invoke this module's `.apply()` directly, *without* wrapping
in `jax.jit` -- same reasoning as `vidax.models.wan.wan2_1.vae`'s module
docstring (the chunked encode/decode loop must run eagerly to avoid OOM from
`jax.jit` unrolling every chunk into one HLO program).
"""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from vidax.core.attention import RMSNorm
from vidax.models.wan.common.vae_layers import (
    AttentionBlock, ResidualBlock, Resample, cached_causal_conv3d,
)

# Fixed 2x2 spatial pixel-unshuffle applied before the encoder / after the
# decoder (Wan2.2-main/wan/modules/vae2_2.py: `WanVAE_.encode`/`.decode` both
# hardcode `patch_size=2`, it is not a configurable architecture parameter).
PATCH_SIZE = 2

# Per-channel latent mean/std used to normalize/denormalize Wan2.2's
# 48-channel latent space (Wan2.2-main/wan/modules/vae2_2.py: Wan2_2_VAE.__init__).
VAE_LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
VAE_LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


def patchify(x: jnp.ndarray, patch_size: int = PATCH_SIZE) -> jnp.ndarray:
    """(B, T, H, W, C) -> (B, T, H/p, W/p, C*p*p), matching the reference's
    ``rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=p, r=p)``.
    """
    if patch_size == 1:
        return x
    b, t, h, w, c = x.shape
    ph, pw = h // patch_size, w // patch_size
    x = x.reshape(b, t, ph, patch_size, pw, patch_size, c)
    x = jnp.transpose(x, (0, 1, 2, 4, 6, 5, 3))  # -> (b, t, ph, pw, c, r, q)
    return x.reshape(b, t, ph, pw, c * patch_size * patch_size)


def unpatchify(x: jnp.ndarray, patch_size: int = PATCH_SIZE) -> jnp.ndarray:
    """Inverse of `patchify`, matching the reference's
    ``rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=p, r=p)``.
    """
    if patch_size == 1:
        return x
    b, t, ph, pw, cc = x.shape
    c = cc // (patch_size * patch_size)
    x = x.reshape(b, t, ph, pw, c, patch_size, patch_size)  # (..., c, r, q)
    x = jnp.transpose(x, (0, 1, 2, 6, 3, 5, 4))  # -> (b, t, ph, q, pw, r, c)
    return x.reshape(b, t, ph * patch_size, pw * patch_size, c)


def avg_down3d(x: jnp.ndarray, out_channels: int, factor_t: int, factor_s: int = 1) -> jnp.ndarray:
    """Parameter-free downsampling shortcut: reshape into (factor_t * factor_s^2)
    groups per output channel and average-pool over each group. Matches the
    reference's ``AvgDown3D``.
    """
    b, t, h, w, c = x.shape
    pad_t = (factor_t - t % factor_t) % factor_t
    if pad_t:
        x = jnp.pad(x, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)))
    t2 = x.shape[1]
    factor = factor_t * factor_s * factor_s
    x = x.reshape(b, t2 // factor_t, factor_t, h // factor_s, factor_s, w // factor_s, factor_s, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
    x = x.reshape(b, t2 // factor_t, h // factor_s, w // factor_s, c * factor)
    group_size = c * factor // out_channels
    x = x.reshape(b, t2 // factor_t, h // factor_s, w // factor_s, out_channels, group_size)
    return x.mean(axis=-1)


def dup_up3d(
    x: jnp.ndarray, out_channels: int, factor_t: int, factor_s: int = 1,
    first_chunk: bool = False,
) -> jnp.ndarray:
    """Parameter-free upsampling shortcut: repeat channels and reshuffle into
    the spatiotemporal grid. Matches the reference's ``DupUp3D``.
    """
    b, t, h, w, c = x.shape
    factor = factor_t * factor_s * factor_s
    repeats = out_channels * factor // c
    x = jnp.repeat(x, repeats, axis=-1)
    x = x.reshape(b, t, h, w, out_channels, factor_t, factor_s, factor_s)
    x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    x = x.reshape(b, t * factor_t, h * factor_s, w * factor_s, out_channels)
    if first_chunk:
        x = x[:, factor_t - 1:]
    return x


class DownResidualBlock(nn.Module):
    """Mirrors the reference ``Down_ResidualBlock``: `mult` ResidualBlocks
    (+ optional Resample) as the main path, `avg_down3d` as a parallel,
    parameter-free shortcut, summed at the end.
    """
    in_dim: int
    out_dim: int
    mult: int
    temperal_downsample: bool = False
    down_flag: bool = False
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        x_copy = x
        cur_in = self.in_dim
        for i in range(self.mult):
            x = ResidualBlock(cur_in, self.out_dim, self.eps,
                               name=f"downsamples_{i}")(x, cache_list, idx_ref)
            cur_in = self.out_dim
        if self.down_flag:
            mode = "downsample3d" if self.temperal_downsample else "downsample2d"
            x = Resample(self.out_dim, mode, self.eps,
                         name=f"downsamples_{self.mult}")(x, cache_list, idx_ref)

        factor_t = 2 if self.temperal_downsample else 1
        factor_s = 2 if self.down_flag else 1
        shortcut = avg_down3d(x_copy, self.out_dim, factor_t, factor_s)
        return x + shortcut


class UpResidualBlock(nn.Module):
    """Mirrors the reference ``Up_ResidualBlock``: `mult` ResidualBlocks
    (+ optional Resample) as the main path, `dup_up3d` as a parallel,
    parameter-free shortcut (only when `up_flag`), summed at the end.
    """
    in_dim: int
    out_dim: int
    mult: int
    temperal_upsample: bool = False
    up_flag: bool = False
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None, first_chunk: bool = False):
        x_main = x
        cur_in = self.in_dim
        for i in range(self.mult):
            x_main = ResidualBlock(cur_in, self.out_dim, self.eps,
                                    name=f"upsamples_{i}")(x_main, cache_list, idx_ref)
            cur_in = self.out_dim
        if not self.up_flag:
            return x_main

        mode = "upsample3d" if self.temperal_upsample else "upsample2d"
        x_main = Resample(self.out_dim, mode, self.eps, halve_upsample_channels=False,
                           name=f"upsamples_{self.mult}")(x_main, cache_list, idx_ref)
        factor_t = 2 if self.temperal_upsample else 1
        shortcut = dup_up3d(x, self.out_dim, factor_t, 2, first_chunk=first_chunk)
        return x_main + shortcut


class Encoder3d(nn.Module):
    """Mirrors the reference ``Encoder3d``: conv1 -> downsamples -> middle -> head."""
    dim: int = 160
    z_dim: int = 96  # note: this is the *head's* output width (2*48, for mu/log_var).
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: Sequence[bool] = (False, True, True)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None):
        dims = [self.dim * u for u in (1,) + tuple(self.dim_mult)]

        x = cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = self.temperal_downsample[i] if i < len(self.temperal_downsample) else False
            down_flag = i != len(self.dim_mult) - 1
            x = DownResidualBlock(
                in_dim, out_dim, self.num_res_blocks, t_down, down_flag, self.eps,
                name=f"downsamples_{i}")(x, cache_list, idx_ref)

        out_dim = dims[-1]
        x = ResidualBlock(out_dim, out_dim, self.eps, name="middle_0")(x, cache_list, idx_ref)
        x = AttentionBlock(out_dim, self.eps, name="middle_1")(x)
        x = ResidualBlock(out_dim, out_dim, self.eps, name="middle_2")(x, cache_list, idx_ref)

        x = RMSNorm(out_dim, eps=self.eps, name="head_0")(x)
        x = nn.silu(x)
        x = cached_causal_conv3d(x, self.z_dim, (3, 3, 3), "head_2", cache_list, idx_ref)
        return x


def _count_causal_convs_encoder(encoder: Encoder3d) -> int:
    """Number of CausalConv3d calls per forward pass, for cache-list sizing.

    Each ResidualBlock issues 2 (conv1, conv2); Resample only issues one
    (its `time_conv`) when its mode is the temporal variant
    (downsample3d/upsample3d) -- the spatial `resample_1` conv is a plain,
    uncached `nn.Conv`, not a `causal_conv3d` call. See
    `vidax.models.wan.wan2_1.vae`'s identically-structured count functions.
    """
    count = 1  # conv1
    for i in range(len(encoder.dim_mult)):
        count += 2 * encoder.num_res_blocks  # this stage's DownResidualBlock ResidualBlocks
        down_flag = i != len(encoder.dim_mult) - 1
        t_down = encoder.temperal_downsample[i] if i < len(encoder.temperal_downsample) else False
        if down_flag and t_down:
            count += 1  # the stage's Resample (downsample3d's time_conv)
    count += 2 + 2  # middle_0, middle_2 (middle_1 is attention, no conv)
    count += 1  # head_2
    return count


class Decoder3d(nn.Module):
    """Mirrors the reference ``Decoder3d``: conv1 -> middle -> upsamples -> head."""
    dim: int = 256
    z_dim: int = 48
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_upsample: Sequence[bool] = (True, True, False)
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cache_list=None, idx_ref=None, first_chunk: bool = False):
        dims = [self.dim * u for u in (self.dim_mult[-1],) + tuple(self.dim_mult[::-1])]

        x = cached_causal_conv3d(x, dims[0], (3, 3, 3), "conv1", cache_list, idx_ref)

        x = ResidualBlock(dims[0], dims[0], self.eps, name="middle_0")(x, cache_list, idx_ref)
        x = AttentionBlock(dims[0], self.eps, name="middle_1")(x)
        x = ResidualBlock(dims[0], dims[0], self.eps, name="middle_2")(x, cache_list, idx_ref)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = self.temperal_upsample[i] if i < len(self.temperal_upsample) else False
            up_flag = i != len(self.dim_mult) - 1
            x = UpResidualBlock(
                in_dim, out_dim, self.num_res_blocks + 1, t_up, up_flag, self.eps,
                name=f"upsamples_{i}")(x, cache_list, idx_ref, first_chunk)

        x = RMSNorm(dims[-1], eps=self.eps, name="head_0")(x)
        x = nn.silu(x)
        x = cached_causal_conv3d(x, 12, (3, 3, 3), "head_2", cache_list, idx_ref)
        return x


def _count_causal_convs(decoder: Decoder3d) -> int:
    """Number of CausalConv3d calls per forward pass, for cache-list sizing
    (see `_count_causal_convs_encoder`'s docstring for why Resample only
    sometimes contributes).
    """
    count = 1  # conv1
    count += 2 + 2  # middle_0, middle_2 (middle_1 is attention, no conv)
    for i, _ in enumerate(decoder.dim_mult):
        count += 2 * (decoder.num_res_blocks + 1)  # this stage's UpResidualBlock ResidualBlocks
        up_flag = i != len(decoder.dim_mult) - 1
        t_up = decoder.temperal_upsample[i] if i < len(decoder.temperal_upsample) else False
        if up_flag and t_up:
            count += 1  # the stage's Resample (upsample3d's time_conv)
    count += 1  # head_2
    return count


class WanVAEDecoder(nn.Module):
    """Top-level Wan2.2 VAE decoder: latent normalization + conv2 + Decoder3d + unpatchify.

    Uses `setup()` (not `@nn.compact`) specifically so `decode_chunk` can be
    called on its own, independent of `__call__` -- see that method's
    docstring for why large-resolution callers should use it (wrapped in
    `jax.jit`) instead of calling this module directly. `__call__` still
    works as a simple, fully-eager convenience path (e.g. for the small-scale
    tests/init calls elsewhere in this repo): it just calls `pre_process`
    then `decode_chunk` in a plain Python loop itself.
    """
    dim: int = 256
    z_dim: int = 48
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
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
            self.temperal_upsample, self.eps, name="decoder")

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: (B, T_latent, H_latent, W_latent, z_dim) latents.

        Returns:
            (B, T_latent upsampled to frames, H*16, W*16, 3) RGB frames in [-1, 1].
        """
        x = self.pre_process(z)
        cache_list = [None] * _count_causal_convs(self.decoder)

        outputs = []
        for i in range(x.shape[1]):
            out_chunk, cache_list = self.decode_chunk(x[:, i:i + 1], cache_list, first_chunk=(i == 0))
            outputs.append(out_chunk)
        out = jnp.concatenate(outputs, axis=1)
        return unpatchify(out, PATCH_SIZE)

    def pre_process(self, z: jnp.ndarray) -> jnp.ndarray:
        """Denormalizes and runs `conv2` on the *full* (unchunked) latent
        tensor -- safe and cheap to do all at once (kernel-1, no temporal
        receptive field to worry about across chunk boundaries), matching
        the reference's own `WanVAE_.decode`, which does the same before its
        per-frame loop.
        """
        mean = jnp.asarray(VAE_LATENT_MEAN, dtype=z.dtype)
        std = jnp.asarray(VAE_LATENT_STD, dtype=z.dtype)
        z = z * std + mean
        return self.conv2(z)

    def decode_chunk(self, x_chunk: jnp.ndarray, cache_list: list, first_chunk: bool = False):
        """Decodes one latent frame (`x_chunk`, from `pre_process`'s output),
        given and returning the running cache state.

        This -- not `__call__` -- is what production callers should use,
        wrapped in `jax.jit` with `first_chunk` passed as a static argument
        (e.g. `jax.jit(lambda params, x, c, fc: model.apply(params, x, c,
        fc, method=model.decode_chunk), static_argnums=(3,))`), called from
        a plain Python loop over latent frames (see
        `examples/generate_wan2_2_ti2v.py`). Calling `Decoder3d` (a deep,
        1024-channel-wide network at TI2V-5B's full resolution) eagerly,
        frame by frame, means every individual op inside it -- not just the
        whole per-frame step -- triggers its own separate, slow XLA
        compilation; jit-ing this method instead compiles the *whole*
        per-chunk computation as one fused program, once per distinct
        `cache_list`/`first_chunk` structure (in practice: once for the
        first chunk, once for the second, and once more reused for every
        remaining chunk, since `cache_list`'s structure stabilizes from the
        third chunk on -- see `cache_list`'s handling in
        `vidax.models.wan.common.vae_layers.Resample` for why the first two
        chunks are special).

        Args:
            x_chunk: (B, 1, H, W, dim) -- one latent frame, post-`pre_process`.
            cache_list: Running cache state -- `[None] * _count_causal_convs(
                self.decoder)` for the very first call, whatever this method
                last returned otherwise.
            first_chunk: Whether this is the first latent frame overall
                (affects `Up_ResidualBlock`'s temporal-upsample shortcut,
                `DupUp3D` -- see `vidax.models.wan.wan2_2.vae.UpResidualBlock`).

        Returns:
            (out_chunk, new_cache_list).
        """
        idx_ref = [0]
        out_chunk = self.decoder(x_chunk, cache_list, idx_ref, first_chunk=first_chunk)
        return out_chunk, cache_list


class WanVAEEncoder(nn.Module):
    """Top-level Wan2.2 VAE encoder: patchify + Encoder3d -> conv1 (mu/log_var) -> normalize.

    Returns the normalized latent *mean* only (deterministic, no
    reparameterization sampling), matching `vidax.models.wan.wan2_1.vae.WanVAEEncoder`.

    Uses `setup()` (not `@nn.compact`) so `encode_chunk` can be called on
    its own, independent of `__call__` -- same reasoning as
    `WanVAEDecoder.decode_chunk`'s docstring.
    """
    dim: int = 160
    z_dim: int = 48
    dim_mult: Sequence[int] = (1, 2, 4, 4)
    num_res_blocks: int = 2
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
            self.temperal_downsample, self.eps, name="encoder")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, T, H, W, 3) RGB video in [-1, 1], T = 1 + 4k.

        Returns:
            (B, T_latent, H/16, W/16, z_dim) normalized latent means, where
            T_latent = 1 + (T - 1) // 4.
        """
        x = self.pre_process(x)

        t = x.shape[1]
        bounds = [(0, 1)] + [(1 + 4 * i, 1 + 4 * (i + 1)) for i in range((t - 1) // 4)]

        cache_list = [None] * _count_causal_convs_encoder(self.encoder)
        outputs = []
        for start, end in bounds:
            out_chunk, cache_list = self.encode_chunk(x[:, start:end], cache_list)
            outputs.append(out_chunk)
        out = jnp.concatenate(outputs, axis=1)
        return self.post_process(out)

    def pre_process(self, x: jnp.ndarray) -> jnp.ndarray:
        """2x2 pixel-patchify, applied once to the *full* (unchunked) input
        video -- safe and cheap to do all at once (a pure spatial reshuffle,
        no temporal receptive field to worry about across chunk boundaries).
        """
        return patchify(x, PATCH_SIZE)

    def encode_chunk(self, x_chunk: jnp.ndarray, cache_list: list):
        """Encodes one chunk (the first call gets 1 frame, every subsequent
        call gets 4 -- see `__call__`'s `bounds`), given and returning the
        running cache state.

        This -- not `__call__` -- is what production callers encoding more
        than a single frame should use, wrapped in `jax.jit` and called from
        a plain Python loop -- see `WanVAEDecoder.decode_chunk`'s docstring
        for the full reasoning (identical here, just for the encoder).

        Args:
            x_chunk: (B, 1 or 4, H, W, 12) -- one chunk, post-`pre_process`.
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
