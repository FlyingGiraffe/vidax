"""HunyuanVideo 1.0 causal 3D-conv KL-VAE ("884-16c-hy") -- port of
``hyvideo/vae/{autoencoder_kl_causal_3d.py,vae.py,unet_causal_3d_blocks.py}``'s
``AutoencoderKLCausal3D``/``EncoderCausal3D``/``DecoderCausal3D``/
``ResnetBlockCausal3D``/``UNetMidBlockCausal3D``/``DownsampleCausal3D``/
``UpsampleCausal3D``/``CausalConv3d``.

**Genuinely different from ``hunyuan_video_1_5/vae.py``** (confirmed by
reading both reference repos directly, not assumed from the shared
"causal 3D-conv KL-VAE" family name): this is diffusers' standard
``AutoencoderKLCausal3D`` (GroupNorm, plain-conv down/upsample, a diffusers
``Attention`` mid-block), not 1.5's bespoke ``RMS_norm`` + pixel-(un)shuffle
design -- so this is a fresh port, not a reuse. Real key names/shapes
confirmed directly against the downloaded
``tencent/HunyuanVideo/hunyuan-video-t2v-720p/vae/pytorch_model.pt`` (248
leaves) -- see ``docs/lessons/hunyuan_video_1_debugging.md``.

**Layout**: channel-last (B, T, H, W, C) throughout, matching this repo's
established VAE convention -- the reference is channel-first (B, C, T, H, W);
callers convert at the boundary.

**Real config** (``vae/config.json``, not hardcoded): ``block_out_channels=
[128,256,512,512]``, ``layers_per_block=2``, ``latent_channels=16``,
``norm_num_groups=32``, ``in_channels=out_channels=3``,
``spatial_compression_ratio=8`` (default, not itself a config.json key),
``time_compression_ratio=4``, ``scaling_factor=0.476986``,
``mid_block_add_attention=True``. No ``shift_factor`` (unlike 1.5's VAE) --
the reference never applies one for this VAE.

**Per-level down/upsample schedule** (``_level_strides``, replicated exactly
from ``EncoderCausal3D``/``DecoderCausal3D``'s own bookkeeping, not
transcribed as a fixed per-checkpoint table): for ``num_spatial =
log2(spatial_compression_ratio)`` and ``num_time = log2(time_compression_ratio)``,
level ``i`` (of ``L = len(block_out_channels)`` total) adds a spatial
downsample iff ``i < num_spatial`` and a temporal downsample iff
``i >= L - 1 - num_time`` and ``i`` isn't the final level -- confirmed against
the real config (``[128,256,512,512]``, spatial=8, time=4, L=4) to give
levels 0/1/2 spatial (8x total), levels 1/2 temporal (4x total), level 3 no
downsample at all (a plain 2-ResnetBlock level with no ``downsamplers``/
``upsamplers`` key in the real checkpoint -- confirmed directly).

**Upsample's frame-0 special case** (``UpsampleCausal3D.forward``): the
first (index-0) latent frame is *only* ever spatially upsampled (never
temporally duplicated) -- ported as ``_upsample_causal`` splitting frame 0
from the rest before any temporal ``upsample_factor`` is applied, then
re-concatenating. This preserves the causal convention that latent frame 0
always maps to exactly one output pixel frame, regardless of temporal
upsample factor. Downsample has no such special case (plain strided causal
conv) -- confirmed by reading ``DownsampleCausal3D.forward`` directly (no
frame-splitting logic there at all).

**Mid-block attention**: single-head (``heads=in_channels//attention_head_dim
== 1`` since ``attention_head_dim=block_out_channels[-1]==in_channels`` for
this VAE's mid-block), diffusers ``Attention`` module (``group_norm``,
``to_q``/``to_k``/``to_v``/``to_out.0``, residual connection), with the same
causal (query frame i attends to key frames <= i, full spatial attention
within/across those frames) mask as 1.5's ``AttnBlock`` -- reused verbatim
(structurally identical masking logic, ported from the same
``prepare_causal_attention_mask`` reference function), just GroupNorm
instead of RMSNorm before the QKV projections.

Only spatial tiling is implemented (mirroring 1.5's VAE + this repo's
general precedent) -- staged per-level decode (same OOM-avoidance rationale
as ``hunyuan_video_1_5/vae.py``'s ``stage_level``/``stage_level_block``/
``stage_level_upsample`` docstrings) is provided via
``decode_stage_*``/``num_decoder_levels``/``num_blocks_per_level``, used by
``examples/generate_hunyuan_video_1_0.py``'s own ``spatial_tiled_vae_decode``.
"""
import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.nn as jnn

NORM_GROUPS = 32
NORM_EPS = 1e-6


def causal_conv3d(x: jnp.ndarray, features: int, kernel_size: int, name: str,
                   strides: Tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    """``CausalConv3d`` (``pad_mode='replicate'``): edge-pad H/W symmetrically
    by ``k//2`` and T causally (``k-1`` on the front only), *regardless of
    stride* (the reference's own padding formula never depends on
    ``stride`` -- confirmed by reading ``CausalConv3d.__init__`` directly).
    """
    k = kernel_size
    pad_hw = k // 2
    pad_t = k - 1
    x = jnp.pad(x, ((0, 0), (pad_t, 0), (pad_hw, pad_hw), (pad_hw, pad_hw), (0, 0)), mode="edge")
    return nn.Conv(features, (k, k, k), strides=strides, padding="VALID", name=name)(x)


def _conv1x1(x: jnp.ndarray, features: int, name: str) -> jnp.ndarray:
    return nn.Conv(features, (1, 1, 1), padding="VALID", name=name)(x)


class AttnBlock(nn.Module):
    """Diffusers ``Attention`` (single-head, ``group_norm`` + ``to_q``/
    ``to_k``/``to_v``/``to_out_0``), with the same causal
    (frame_q >= frame_k) full-spatial-attention mask as
    ``hunyuan_video_1_5.vae.AttnBlock`` -- see module docstring.
    """
    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = nn.GroupNorm(num_groups=NORM_GROUPS, epsilon=NORM_EPS, name="group_norm")(x)
        b, f, ht, w, c = h.shape
        h_flat = h.reshape(b, f * ht * w, c)
        q = nn.Dense(self.in_channels, name="to_q")(h_flat)
        k = nn.Dense(self.in_channels, name="to_k")(h_flat)
        v = nn.Dense(self.in_channels, name="to_v")(h_flat)

        n_hw = ht * w
        frame_idx = jnp.repeat(jnp.arange(f), n_hw)
        allowed = frame_idx[None, :] <= frame_idx[:, None]  # (S, S), True = key visible to query

        scale = 1.0 / jnp.sqrt(jnp.array(self.in_channels, dtype=jnp.float32))
        logits = jnp.einsum("bqc,bkc->bqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        logits = jnp.where(allowed[None, :, :], logits, -1e9)
        weights = jnn.softmax(logits, axis=-1).astype(v.dtype)
        attn = jnp.einsum("bqk,bkc->bqc", weights, v)
        out = nn.Dense(self.in_channels, name="to_out_0")(attn)
        out = out.reshape(b, f, ht, w, self.in_channels)
        return residual + out


class ResnetBlock(nn.Module):
    """``ResnetBlockCausal3D`` (``time_embedding_norm="default"``,
    ``temb_channels=None`` -- this VAE never actually conditions on a
    timestep, so the ``time_emb_proj`` branch is dead code, not ported).
    """
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=NORM_GROUPS, epsilon=NORM_EPS, name="norm1")(x)
        h = jnn.silu(h)
        h = causal_conv3d(h, self.out_channels, 3, "conv1")
        h = nn.GroupNorm(num_groups=NORM_GROUPS, epsilon=NORM_EPS, name="norm2")(h)
        h = jnn.silu(h)
        h = causal_conv3d(h, self.out_channels, 3, "conv2")
        if self.in_channels != self.out_channels:
            x = causal_conv3d(x, self.out_channels, 1, "conv_shortcut")
        return x + h


def _level_schedule(num_levels: int, spatial_compression_ratio: int, time_compression_ratio: int):
    """Replicates ``EncoderCausal3D``/``DecoderCausal3D``'s per-level
    ``add_spatial_downsample``/``add_time_downsample`` bookkeeping exactly
    (see module docstring) -- returns one ``(add_spatial, add_time)`` tuple
    per level, in encoder order (level 0 = highest resolution).
    """
    num_spatial = int(math.log2(spatial_compression_ratio))
    num_time = int(math.log2(time_compression_ratio))
    out = []
    for i in range(num_levels):
        is_final = i == num_levels - 1
        add_spatial = i < num_spatial
        add_time = (i >= (num_levels - 1 - num_time)) and not is_final
        out.append((add_spatial, add_time))
    return out


def _upsample_causal(x: jnp.ndarray, features: int, add_time: bool, name: str) -> jnp.ndarray:
    """``UpsampleCausal3D``: frame-0 is only ever spatially nearest-upsampled
    (never temporally duplicated -- see module docstring), the rest of the
    frames use the full (T,H,W) nearest-upsample factor, then a single
    causal conv (kernel 3, stride 1) is applied to the whole result.
    """
    b, f, h, w, c = x.shape
    first = x[:, :1]
    first = jnp.repeat(jnp.repeat(first, 2, axis=2), 2, axis=3)  # spatial-only nearest x2
    if f > 1:
        rest = x[:, 1:]
        rest = jnp.repeat(rest, 2, axis=2)
        rest = jnp.repeat(rest, 2, axis=3)
        if add_time:
            rest = jnp.repeat(rest, 2, axis=1)
        x = jnp.concatenate([first, rest], axis=1)
    else:
        x = first
    return causal_conv3d(x, features, 3, name)


def _downsample_stride(add_spatial: bool, add_time: bool) -> Tuple[int, int, int]:
    return ((2 if add_time else 1), (2 if add_spatial else 1), (2 if add_spatial else 1))


class Encoder(nn.Module):
    in_channels: int
    latent_channels: int
    block_out_channels: Tuple[int, ...]
    layers_per_block: int
    spatial_compression_ratio: int
    time_compression_ratio: int
    mid_block_add_attention: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        block_in = self.block_out_channels[0]
        h = causal_conv3d(x, block_in, 3, "conv_in")

        schedule = _level_schedule(
            len(self.block_out_channels), self.spatial_compression_ratio, self.time_compression_ratio)
        for i_level, (block_out, (add_spatial, add_time)) in enumerate(zip(self.block_out_channels, schedule)):
            for i_block in range(self.layers_per_block):
                h = ResnetBlock(block_in, block_out, name=f"down_blocks_{i_level}_resnets_{i_block}")(h)
                block_in = block_out
            if add_spatial or add_time:
                h = causal_conv3d(
                    h, block_out, 3, f"down_blocks_{i_level}_downsamplers_0",
                    strides=_downsample_stride(add_spatial, add_time))

        h = ResnetBlock(block_in, block_in, name="mid_block_resnets_0")(h)
        if self.mid_block_add_attention:
            h = AttnBlock(block_in, name="mid_block_attentions_0")(h)
        h = ResnetBlock(block_in, block_in, name="mid_block_resnets_1")(h)

        h = nn.GroupNorm(num_groups=NORM_GROUPS, epsilon=NORM_EPS, name="conv_norm_out")(h)
        h = jnn.silu(h)
        h = causal_conv3d(h, 2 * self.latent_channels, 3, "conv_out")
        return h


class Decoder(nn.Module):
    latent_channels: int
    out_channels: int
    block_out_channels: Tuple[int, ...]  # already reversed by the caller (decoder order)
    layers_per_block: int
    spatial_compression_ratio: int
    time_compression_ratio: int
    mid_block_add_attention: bool = True

    def _schedule(self):
        return _level_schedule(
            len(self.block_out_channels), self.spatial_compression_ratio, self.time_compression_ratio)

    @nn.compact
    def stage_in_and_mid(self, z: jnp.ndarray) -> jnp.ndarray:
        block_in = self.block_out_channels[0]
        h = causal_conv3d(z, block_in, 3, "conv_in")
        h = ResnetBlock(block_in, block_in, name="mid_block_resnets_0")(h)
        if self.mid_block_add_attention:
            h = AttnBlock(block_in, name="mid_block_attentions_0")(h)
        h = ResnetBlock(block_in, block_in, name="mid_block_resnets_1")(h)
        return h

    @nn.compact
    def stage_level(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        """One up-level's ``layers_per_block + 1`` ResnetBlocks + optional
        Upsample, as its own separately-jit-able call -- see
        ``hunyuan_video_1_5/vae.py``'s ``Decoder.stage_level`` docstring for
        why real frame counts need this split rather than a single fused
        ``Decoder.__call__``.
        """
        schedule = self._schedule()
        add_spatial, add_time = schedule[i_level]
        block_in = self.block_out_channels[i_level - 1] if i_level > 0 else self.block_out_channels[0]
        block_out = self.block_out_channels[i_level]
        for i_block in range(self.layers_per_block + 1):
            h = ResnetBlock(block_in, block_out, name=f"up_blocks_{i_level}_resnets_{i_block}")(h)
            block_in = block_out
        if add_spatial or add_time:
            h = _upsample_causal(h, block_out, add_time, f"up_blocks_{i_level}_upsamplers_0_conv")
        return h

    @nn.compact
    def stage_level_block(self, h: jnp.ndarray, i_level: int, i_block: int) -> jnp.ndarray:
        block_in_level = self.block_out_channels[i_level - 1] if i_level > 0 else self.block_out_channels[0]
        block_out = self.block_out_channels[i_level]
        block_in = block_in_level if i_block == 0 else block_out
        return ResnetBlock(block_in, block_out, name=f"up_blocks_{i_level}_resnets_{i_block}")(h)

    @nn.compact
    def stage_level_upsample(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        add_spatial, add_time = self._schedule()[i_level]
        block_out = self.block_out_channels[i_level]
        if not (add_spatial or add_time):
            return h
        return _upsample_causal(h, block_out, add_time, f"up_blocks_{i_level}_upsamplers_0_conv")

    @nn.compact
    def stage_out(self, h: jnp.ndarray) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=NORM_GROUPS, epsilon=NORM_EPS, name="conv_norm_out")(h)
        h = jnn.silu(h)
        return causal_conv3d(h, self.out_channels, 3, "conv_out")

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = self.stage_in_and_mid(z)
        for i_level in range(len(self.block_out_channels)):
            h = self.stage_level(h, i_level)
        return self.stage_out(h)


def blend_h(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
    """Linear cross-fade along the width axis (channel-last: -2) -- see
    ``hunyuan_video_1_5.vae.blend_h``'s identical docstring."""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    if blend_extent <= 0:
        return b
    weight = (jnp.arange(blend_extent, dtype=jnp.float32) / blend_extent).reshape(
        (1,) * (a.ndim - 2) + (blend_extent, 1))
    a_edge = a[..., -blend_extent:, :].astype(jnp.float32)
    b_edge = b[..., :blend_extent, :].astype(jnp.float32)
    blended = (a_edge * (1 - weight) + b_edge * weight).astype(b.dtype)
    return jnp.concatenate([blended, b[..., blend_extent:, :]], axis=-2)


def blend_v(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
    """Same as ``blend_h`` but along the height axis (channel-last: -3)."""
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    if blend_extent <= 0:
        return b
    weight = (jnp.arange(blend_extent, dtype=jnp.float32) / blend_extent).reshape(
        (1,) * (a.ndim - 3) + (blend_extent, 1, 1))
    a_edge = a[..., -blend_extent:, :, :].astype(jnp.float32)
    b_edge = b[..., :blend_extent, :, :].astype(jnp.float32)
    blended = (a_edge * (1 - weight) + b_edge * weight).astype(b.dtype)
    return jnp.concatenate([blended, b[..., blend_extent:, :, :]], axis=-3)


class HunyuanVideo10VAE(nn.Module):
    """Top-level channel-last KL-VAE, port of ``AutoencoderKLCausal3D``.
    ``scaling_factor`` (no ``shift_factor`` for this VAE) is applied by the
    caller, matching diffusers convention -- this module operates on raw
    (unnormalized) latents in both ``encode``/``decode``.
    """
    in_channels: int
    out_channels: int
    latent_channels: int
    block_out_channels: Tuple[int, ...]
    layers_per_block: int
    spatial_compression_ratio: int
    time_compression_ratio: int
    mid_block_add_attention: bool = True

    def setup(self):
        self.encoder = Encoder(
            self.in_channels, self.latent_channels, self.block_out_channels, self.layers_per_block,
            self.spatial_compression_ratio, self.time_compression_ratio, self.mid_block_add_attention,
            name="encoder")
        self.decoder = Decoder(
            self.latent_channels, self.out_channels, tuple(reversed(self.block_out_channels)),
            self.layers_per_block, self.spatial_compression_ratio, self.time_compression_ratio,
            self.mid_block_add_attention, name="decoder")
        self.quant_conv = nn.Conv(2 * self.latent_channels, (1, 1, 1), padding="VALID", name="quant_conv")
        self.post_quant_conv = nn.Conv(self.latent_channels, (1, 1, 1), padding="VALID", name="post_quant_conv")

    def encode(self, x: jnp.ndarray):
        """x: (B, T, H, W, in_channels) -> (mean, logvar), each (B, T', H', W', latent_channels)."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = jnp.split(moments, 2, axis=-1)
        return mean, logvar

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """z: (B, T, H, W, latent_channels) -> pixels (B, T*4, H*8, W*8, out_channels)."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @property
    def num_decoder_levels(self) -> int:
        return len(self.block_out_channels)

    @property
    def num_blocks_per_level(self) -> int:
        return self.layers_per_block + 1

    def decode_stage_in_and_mid(self, z: jnp.ndarray) -> jnp.ndarray:
        z = self.post_quant_conv(z)
        return self.decoder.stage_in_and_mid(z)

    def decode_stage_level(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        return self.decoder.stage_level(h, i_level)

    def decode_stage_level_block(self, h: jnp.ndarray, i_level: int, i_block: int) -> jnp.ndarray:
        return self.decoder.stage_level_block(h, i_level, i_block)

    def decode_stage_level_upsample(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        return self.decoder.stage_level_upsample(h, i_level)

    def decode_stage_out(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.decoder.stage_out(h)
