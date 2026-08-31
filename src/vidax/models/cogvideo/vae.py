"""CogVideoX 3D causal VAE (Flax/JAX), encoder + decoder, channels-last.

A structural port of `AutoencoderKLCogVideoX` and its building blocks
(`CogVideoXEncoder3D` / `CogVideoXDecoder3D` / `CogVideoXResnetBlock3D` /
`CogVideoXDownBlock3D` / `CogVideoXMidBlock3D` / `CogVideoXUpBlock3D` /
`CogVideoXCausalConv3d` / `CogVideoXSpatialNorm3D` / `CogVideoXDownsample3D`
/ `CogVideoXUpsample3D`) from
`diffusers/models/autoencoders/autoencoder_kl_cogvideox.py` (+
`diffusers/models/{downsampling,upsampling}.py`).

Spatial compression 8x, temporal 4x, `latent_channels=16`; identical
architecture for every released CogVideoX checkpoint (only the pipeline-side
`scaling_factor` / `invert_scale_latents` differ -- those live in
`configs.py`, not here).

Two things make this its own file rather than a reuse of Wan's causal VAE:

- **Padding/streaming.** `CogVideoXCausalConv3d` left-pads the temporal axis
  by `kernel_t - 1` frames -- replicating the first frame on the first
  chunk, or prepending the previous chunk's tail from a `conv_cache` -- and
  never pads the right. `encode`/`decode` walk the video in fixed temporal
  chunks (8 sample frames / 2 latent frames) carrying that cache between
  chunks, exactly as the reference's `_encode`/`_decode` loops do. This must
  be replicated faithfully: the temporal `avg_pool` / nearest-`interpolate`
  in `CogVideoXDownsample3D` / `CogVideoXUpsample3D` (with their odd-frame
  "keep frame 0, pool/interp the rest" special-case) are *not* cache-aware,
  so chunked and whole-clip results genuinely differ at chunk boundaries --
  matching diffusers means matching its chunking.
- **Decoder GroupNorm is conditioned on the latent.** `CogVideoXSpatialNorm3D`
  replaces the decoder's GroupNorm with `GroupNorm(f) * conv_y(zq) +
  conv_b(zq)`, where `zq` is the (nearest-upsampled) latent -- so `zq` is
  threaded through every decoder block.

Run `encode` / `decode` eagerly (no `jax.jit` wrapping the chunk loop), same
rationale as `vidax.models.wan.wan2_2.vae`'s docstring.
"""
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

NUM_GROUPS = 32
NORM_EPS = 1e-6
NUM_SAMPLE_FRAMES_BATCH_SIZE = 8
NUM_LATENT_FRAMES_BATCH_SIZE = 2
BLOCK_OUT_CHANNELS = (128, 256, 256, 512)
LAYERS_PER_BLOCK = 3
LATENT_CHANNELS = 16
TEMPORAL_COMPRESS_LEVEL = 2  # log2(temporal_compression_ratio=4)


# --------------------------------------------------------------------------
# nearest-neighbour resize matching torch F.interpolate(mode="nearest")
# --------------------------------------------------------------------------

def _nearest_idx(in_size: int, out_size: int) -> jnp.ndarray:
    # torch nearest: out[i] = in[floor(i * in_size / out_size)]
    import numpy as np
    return jnp.asarray(np.floor(np.arange(out_size) * in_size / out_size).astype(np.int32))


def _interp_nearest(x: jnp.ndarray, axes_scales) -> jnp.ndarray:
    """`x` channels-last; `axes_scales` = list of (axis, out_size) OR
    (axis, ("scale", factor)). Gathers along each named axis, torch-style."""
    for axis, spec in axes_scales:
        in_size = x.shape[axis]
        out_size = in_size * spec[1] if isinstance(spec, tuple) else spec
        idx = _nearest_idx(in_size, out_size)
        x = jnp.take(x, idx, axis=axis)
    return x


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """`CogVideoXCausalConv3d` (pad_mode="first"): temporal-causal left pad of
    `kt-1` frames (first-frame replication or `conv_cache` tail), symmetric
    spatial pad; valid conv. Returns `(out, new_cache)` -- the new cache is
    the last `kt-1` frames of the (padded) input, matching the reference.
    """
    features: int
    kernel_size: tuple = (3, 3, 3)

    @nn.compact
    def __call__(self, x, conv_cache=None):
        kt, kh, kw = self.kernel_size
        ph, pw = (kh - 1) // 2, (kw - 1) // 2
        if kt > 1:
            if conv_cache is not None:
                pad = conv_cache
            else:
                pad = jnp.broadcast_to(x[:, :1], (x.shape[0], kt - 1) + x.shape[2:])
            x = jnp.concatenate([pad, x], axis=1)
        new_cache = x[:, -(kt - 1):] if kt > 1 else None
        y = nn.Conv(
            self.features, self.kernel_size, strides=(1, 1, 1),
            padding=[(0, 0), (ph, ph), (pw, pw)], name="conv")(x)
        return y, new_cache


class GroupNorm(nn.Module):
    """`nn.GroupNorm(32, eps=1e-6)` in float32; inner norm named `"norm"` so
    the translator path is `<parent>/norm/{scale,bias}`."""
    channels: int

    @nn.compact
    def __call__(self, x):
        return nn.GroupNorm(
            num_groups=NUM_GROUPS, epsilon=NORM_EPS, use_scale=True, use_bias=True,
            dtype=jnp.float32, param_dtype=jnp.float32, name="norm")(
            x.astype(jnp.float32)).astype(x.dtype)


class SpatialNorm3D(nn.Module):
    """`CogVideoXSpatialNorm3D`: `GroupNorm(f) * conv_y(zq_up) + conv_b(zq_up)`,
    `zq` nearest-upsampled to `f`'s spatial/temporal size (with the reference's
    odd-frame "frame 0 separate" special case).
    """
    f_channels: int

    @nn.compact
    def __call__(self, f, zq, cache=None):
        cache = cache or {}
        Tf = f.shape[1]
        if Tf > 1 and Tf % 2 == 1:
            z_first = _interp_nearest(zq[:, :1], [(1, 1), (2, f.shape[2]), (3, f.shape[3])])
            z_rest = _interp_nearest(
                zq[:, 1:], [(1, Tf - 1), (2, f.shape[2]), (3, f.shape[3])])
            zq_up = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            zq_up = _interp_nearest(zq, [(1, Tf), (2, f.shape[2]), (3, f.shape[3])])
        cy, c_y_cache = CausalConv3d(self.f_channels, (1, 1, 1), name="conv_y")(
            zq_up, cache.get("conv_y"))
        cb, c_b_cache = CausalConv3d(self.f_channels, (1, 1, 1), name="conv_b")(
            zq_up, cache.get("conv_b"))
        out = GroupNorm(self.f_channels, name="norm_layer")(f) * cy + cb
        return out, {"conv_y": c_y_cache, "conv_b": c_b_cache}


class ResnetBlock3D(nn.Module):
    in_channels: int
    out_channels: int
    spatial_norm: bool = False

    @nn.compact
    def __call__(self, x, zq=None, cache=None):
        cache = cache or {}
        new_cache = {}

        def norm(name, ch, h, cname):
            if self.spatial_norm:
                return SpatialNorm3D(ch, name=name)(h, zq, cache.get(cname))
            return GroupNorm(ch, name=name)(h), None

        h, new_cache["norm1"] = norm("norm1", self.in_channels, x, "norm1")
        h = nn.silu(h)
        h, new_cache["conv1"] = CausalConv3d(self.out_channels, name="conv1")(h, cache.get("conv1"))

        h, new_cache["norm2"] = norm("norm2", self.out_channels, h, "norm2")
        h = nn.silu(h)
        h, new_cache["conv2"] = CausalConv3d(self.out_channels, name="conv2")(h, cache.get("conv2"))

        if self.in_channels != self.out_channels:
            x = nn.Conv(self.out_channels, (1, 1, 1), strides=(1, 1, 1), padding="VALID",
                        name="conv_shortcut")(x)
        return x + h, new_cache


class Downsample3D(nn.Module):
    out_channels: int
    compress_time: bool = False

    @nn.compact
    def __call__(self, x):
        # x: (B, T, H, W, C)
        if self.compress_time:
            T = x.shape[1]
            if T % 2 == 1:
                x_first, x_rest = x[:, :1], x[:, 1:]
                if x_rest.shape[1] > 0:
                    x_rest = _avg_pool_time(x_rest)
                x = jnp.concatenate([x_first, x_rest], axis=1)
            else:
                x = _avg_pool_time(x)
        b, t, h, w, c = x.shape
        x = jnp.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)])
        x = x.reshape(b * t, h + 1, w + 1, c)
        x = nn.Conv(self.out_channels, (3, 3), strides=(2, 2), padding="VALID", name="conv")(x)
        return x.reshape(b, t, x.shape[1], x.shape[2], self.out_channels)


class Upsample3D(nn.Module):
    out_channels: int
    compress_time: bool = False

    @nn.compact
    def __call__(self, x):
        b, t, h, w, c = x.shape
        if self.compress_time:
            if t > 1 and t % 2 == 1:
                x_first = _interp_nearest(x[:, 0], [(1, ("scale", 2)), (2, ("scale", 2))])[:, None]
                x_rest = _interp_nearest(
                    x[:, 1:], [(1, ("scale", 2)), (2, ("scale", 2)), (3, ("scale", 2))])
                x = jnp.concatenate([x_first, x_rest], axis=1)
            elif t > 1:
                x = _interp_nearest(x, [(1, ("scale", 2)), (2, ("scale", 2)), (3, ("scale", 2))])
            else:
                x = _interp_nearest(x[:, 0], [(1, ("scale", 2)), (2, ("scale", 2))])[:, None]
        else:
            x = _interp_nearest(x, [(2, ("scale", 2)), (3, ("scale", 2))])
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = nn.Conv(self.out_channels, (3, 3), strides=(1, 1), padding="SAME", name="conv")(x)
        return x.reshape(b, t, h, w, self.out_channels)


def _avg_pool_time(x):
    # x: (B, T, H, W, C) with T even -> (B, T//2, H, W, C), avg over frame pairs.
    b, t, h, w, c = x.shape
    return x.reshape(b, t // 2, 2, h, w, c).mean(axis=2)


# --------------------------------------------------------------------------
# encoder / decoder blocks
# --------------------------------------------------------------------------

class DownBlock3D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int
    add_downsample: bool
    compress_time: bool

    @nn.compact
    def __call__(self, x, cache=None):
        cache = cache or {}
        new_cache = {}
        for i in range(self.num_layers):
            ic = self.in_channels if i == 0 else self.out_channels
            x, new_cache[f"resnets_{i}"] = ResnetBlock3D(
                ic, self.out_channels, name=f"resnets_{i}")(x, None, cache.get(f"resnets_{i}"))
        if self.add_downsample:
            x = Downsample3D(self.out_channels, self.compress_time, name="downsamplers_0")(x)
        return x, new_cache


class MidBlock3D(nn.Module):
    channels: int
    num_layers: int = 2
    spatial_norm: bool = False

    @nn.compact
    def __call__(self, x, zq=None, cache=None):
        cache = cache or {}
        new_cache = {}
        for i in range(self.num_layers):
            x, new_cache[f"resnets_{i}"] = ResnetBlock3D(
                self.channels, self.channels, self.spatial_norm, name=f"resnets_{i}")(
                x, zq, cache.get(f"resnets_{i}"))
        return x, new_cache


class UpBlock3D(nn.Module):
    in_channels: int
    out_channels: int
    num_layers: int
    add_upsample: bool
    compress_time: bool

    @nn.compact
    def __call__(self, x, zq, cache=None):
        cache = cache or {}
        new_cache = {}
        for i in range(self.num_layers):
            ic = self.in_channels if i == 0 else self.out_channels
            x, new_cache[f"resnets_{i}"] = ResnetBlock3D(
                ic, self.out_channels, spatial_norm=True, name=f"resnets_{i}")(
                x, zq, cache.get(f"resnets_{i}"))
        if self.add_upsample:
            x = Upsample3D(self.out_channels, self.compress_time, name="upsamplers_0")(x)
        return x, new_cache


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x, cache=None):
        cache = cache or {}
        new_cache = {}
        x, new_cache["conv_in"] = CausalConv3d(BLOCK_OUT_CHANNELS[0], name="conv_in")(
            x, cache.get("conv_in"))
        out_ch = BLOCK_OUT_CHANNELS[0]
        for i, bc in enumerate(BLOCK_OUT_CHANNELS):
            in_ch, out_ch = out_ch, bc
            x, new_cache[f"down_blocks_{i}"] = DownBlock3D(
                in_ch, out_ch, LAYERS_PER_BLOCK,
                add_downsample=(i != len(BLOCK_OUT_CHANNELS) - 1),
                compress_time=(i < TEMPORAL_COMPRESS_LEVEL),
                name=f"down_blocks_{i}")(x, cache.get(f"down_blocks_{i}"))
        x, new_cache["mid_block"] = MidBlock3D(BLOCK_OUT_CHANNELS[-1], name="mid_block")(
            x, None, cache.get("mid_block"))
        x = nn.silu(GroupNorm(BLOCK_OUT_CHANNELS[-1], name="norm_out")(x))
        x, new_cache["conv_out"] = CausalConv3d(2 * LATENT_CHANNELS, name="conv_out")(
            x, cache.get("conv_out"))
        return x, new_cache


class Decoder(nn.Module):
    out_channels: int = 3

    @nn.compact
    def __call__(self, z, cache=None):
        cache = cache or {}
        new_cache = {}
        zq = z
        rev = list(reversed(BLOCK_OUT_CHANNELS))
        x, new_cache["conv_in"] = CausalConv3d(rev[0], name="conv_in")(z, cache.get("conv_in"))
        x, new_cache["mid_block"] = MidBlock3D(rev[0], spatial_norm=True, name="mid_block")(
            x, zq, cache.get("mid_block"))
        out_ch = rev[0]
        for i, bc in enumerate(rev):
            in_ch, out_ch = out_ch, bc
            x, new_cache[f"up_blocks_{i}"] = UpBlock3D(
                in_ch, out_ch, LAYERS_PER_BLOCK + 1,
                add_upsample=(i != len(rev) - 1),
                compress_time=(i < TEMPORAL_COMPRESS_LEVEL),
                name=f"up_blocks_{i}")(x, zq, cache.get(f"up_blocks_{i}"))
        x, new_cache["norm_out"] = SpatialNorm3D(rev[-1], name="norm_out")(
            x, zq, cache.get("norm_out"))
        x = nn.silu(x)
        x, new_cache["conv_out"] = CausalConv3d(self.out_channels, name="conv_out")(
            x, cache.get("conv_out"))
        return x, new_cache


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------

class CogVideoXVAE(nn.Module):
    """`AutoencoderKLCogVideoX`. Channels-last `(B, T, H, W, C)` everywhere.

    `encode(x)` -> `(mean, logvar)` each `(B, T_lat, H/8, W/8, 16)` (caller
    samples the diagonal Gaussian). `decode(z)` -> `(B, T_pix, H, W, 3)` in
    [-1, 1]. Both walk the clip in temporal chunks with a carried conv cache,
    matching the reference's `_encode` / `_decode`.

    `decode` also mirrors diffusers' **spatial tiling** (`tiled_decode` +
    `blend_v`/`blend_h`): at the reference 480x720 the un-tiled decode's
    512-channel 3D-conv feature maps OOM a TPU v4 chip, so when the latent
    exceeds `tile_latent_min_*` the frame is decoded in overlapping
    240x360-pixel tiles blended back together. Verified to match diffusers'
    `enable_tiling()` output.
    """
    out_channels: int = 3
    sample_height: int = 480
    sample_width: int = 720
    tile_overlap_factor_height: float = 1.0 / 6.0
    tile_overlap_factor_width: float = 1.0 / 5.0
    enable_tiling: bool = True

    def setup(self):
        self.encoder = Encoder(name="encoder")
        self.decoder = Decoder(self.out_channels, name="decoder")
        # diffusers: tile_sample_min_* = sample_* // 2; latent = / 2**(num_blocks-1) = /8.
        self.tile_sample_min_height = self.sample_height // 2
        self.tile_sample_min_width = self.sample_width // 2
        self.tile_latent_min_height = self.tile_sample_min_height // (2 ** (len(BLOCK_OUT_CHANNELS) - 1))
        self.tile_latent_min_width = self.tile_sample_min_width // (2 ** (len(BLOCK_OUT_CHANNELS) - 1))

    def __call__(self, x):
        return self.encode(x)

    def _decode_chunks(self, z):
        """Temporal-chunk loop over the (already spatially-tiled or whole)
        latent -- the reference's `_decode` inner loop."""
        num_frames = z.shape[1]
        fbs = NUM_LATENT_FRAMES_BATCH_SIZE
        num_batches = max(num_frames // fbs, 1)
        remaining = num_frames % fbs
        cache = None
        outs = []
        for i in range(num_batches):
            start = fbs * i + (0 if i == 0 else remaining)
            end = fbs * (i + 1) + remaining
            y, cache = self.decoder(z[:, start:end], cache)
            outs.append(y)
        return jnp.concatenate(outs, axis=1)

    @staticmethod
    def _blend_v(a, b, extent):
        # H axis (2) for channels-last. w = y/extent over the top `extent` rows.
        extent = min(a.shape[2], b.shape[2], extent)
        if extent <= 0:
            return b
        w = (jnp.arange(extent, dtype=b.dtype) / extent).reshape(1, 1, extent, 1, 1)
        blended = a[:, :, -extent:] * (1 - w) + b[:, :, :extent] * w
        return jnp.concatenate([blended, b[:, :, extent:]], axis=2)

    @staticmethod
    def _blend_h(a, b, extent):
        extent = min(a.shape[3], b.shape[3], extent)
        if extent <= 0:
            return b
        w = (jnp.arange(extent, dtype=b.dtype) / extent).reshape(1, 1, 1, extent, 1)
        blended = a[:, :, :, -extent:] * (1 - w) + b[:, :, :, :extent] * w
        return jnp.concatenate([blended, b[:, :, :, extent:]], axis=3)

    def _tiled_decode(self, z):
        _, _, lh, lw, _ = z.shape  # channels-last (B, T, H, W, C)
        th, tw = self.tile_latent_min_height, self.tile_latent_min_width
        overlap_h = int(th * (1 - self.tile_overlap_factor_height))
        overlap_w = int(tw * (1 - self.tile_overlap_factor_width))
        blend_h = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_w = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_h = self.tile_sample_min_height - blend_h
        row_limit_w = self.tile_sample_min_width - blend_w

        rows = []
        for i in range(0, lh, overlap_h):
            row = []
            for j in range(0, lw, overlap_w):
                tile = z[:, :, i:i + th, j:j + tw]
                row.append(self._decode_chunks(tile))
            rows.append(row)

        # diffusers mutates `rows[i][j]` in place during blending, so later
        # tiles blend against already-blended neighbours -- replicate that by
        # writing each blended tile back into the grid before it's used as a
        # neighbour.
        result_rows = []
        for i in range(len(rows)):
            result_row = []
            for j in range(len(rows[i])):
                tile = rows[i][j]
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = self._blend_h(rows[i][j - 1], tile, blend_w)
                rows[i][j] = tile
                result_row.append(tile[:, :, :row_limit_h, :row_limit_w])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        return jnp.concatenate(result_rows, axis=2)

    def encode(self, x):
        num_frames = x.shape[1]
        fbs = NUM_SAMPLE_FRAMES_BATCH_SIZE
        num_batches = max(num_frames // fbs, 1)
        remaining = num_frames % fbs
        cache = None
        outs = []
        for i in range(num_batches):
            start = fbs * i + (0 if i == 0 else remaining)
            end = fbs * (i + 1) + remaining
            chunk = x[:, start:end]
            y, cache = self.encoder(chunk, cache)
            outs.append(y)
        h = jnp.concatenate(outs, axis=1)
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar

    def decode(self, z):
        # z is channels-last (B, T, H, W, C). diffusers `_decode`: tile when
        # the latent is larger than tile_latent_min_*.
        _, _, lh, lw, _ = z.shape
        if self.enable_tiling and (lw > self.tile_latent_min_width or lh > self.tile_latent_min_height):
            return self._tiled_decode(z)
        return self._decode_chunks(z)
