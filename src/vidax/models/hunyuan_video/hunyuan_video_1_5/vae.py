"""HunyuanVideo-1.5 causal 3D-conv KL-VAE -- port of
``hyvideo/models/autoencoders/hunyuanvideo_15_vae.py``'s ``Encoder``,
``Decoder``, ``AutoencoderKLConv3D`` (``AttnBlock``/``ResnetBlock``/
``Downsample``/``Upsample``/``RMS_norm``/``CausalConv3d``).

**Layout**: channel-last (B, T, H, W, C) throughout, matching this repo's
established Wan-VAE convention (``vidax.models.wan.common.vae_layers``) --
the reference is channel-first (B, C, T, H, W); callers convert at the
boundary (`jnp.moveaxis(x, 1, -1)` / back), same as every other VAE port
here.

**Scope for this pass**: single-tile (no spatial/temporal tiling --
``enable_temporal_tiling`` is unconditionally unsupported in the reference
itself; spatial tiling is a memory optimization for large resolutions,
deferred as documented follow-up, same precedent as this repo's DiT
``--offload_dit_weights`` and the ltx2_5 diffusion-decoder plan's own
"single full-volume tile first" approach).

Real ``block_out_channels``/``layers_per_block``/etc. come from the
downloaded checkpoint's ``vae/config.json`` (read via ``configs.py``'s
``load_hunyuan_video_1_5_vae_config`` + a kwargs builder -- not hardcoded
here), transcribed at the time of writing as: ``block_out_channels=
[128,256,512,1024,1024]``, ``layers_per_block=2``, ``ffactor_spatial=16``,
``ffactor_temporal=4``, ``latent_channels=32``, ``in_channels=out_channels=
3``, ``scaling_factor=1.03682``, ``shift_factor=None``.
"""
import math
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.nn as jnn


class RMSNormLayer(nn.Module):
    """Channel-last port of ``RMS_norm(dim, images=False)`` (``bias=False``
    default, unused in this VAE): ``F.normalize(x, dim=channel) * sqrt(dim)
    * gamma`` -- algebraically identical to standard (no-eps, well,
    eps=1e-12) RMSNorm: ``x * rsqrt(mean(x**2) + eps) * gamma``.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dim = x.shape[-1]
        gamma = self.param("gamma", nn.initializers.ones, (dim,))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + 1e-12)
        return (normed * gamma.astype(jnp.float32)).astype(x.dtype)


def causal_conv3d(x: jnp.ndarray, features: int, kernel_size: int, name: str) -> jnp.ndarray:
    """``CausalConv3d``: replicate-pad H/W symmetrically by ``k//2``, pad T
    causally (``k-1`` on the front only) with ``mode='edge'`` (replicate) --
    matches the reference's ``pad_mode='replicate'`` default exactly (NOT
    zero-padding, unlike Wan's causal conv).
    """
    k = kernel_size
    pad_hw = k // 2
    pad_t = k - 1
    x = jnp.pad(x, ((0, 0), (pad_t, 0), (pad_hw, pad_hw), (pad_hw, pad_hw), (0, 0)), mode="edge")
    return nn.Conv(features, (k, k, k), strides=(1, 1, 1), padding="VALID", name=name)(x)


def _conv1x1(x: jnp.ndarray, features: int, name: str) -> jnp.ndarray:
    return nn.Conv(features, (1, 1, 1), padding="VALID", name=name)(x)


class AttnBlock(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = RMSNormLayer(name="norm")(x)
        q = _conv1x1(h, self.in_channels, "q")
        k = _conv1x1(h, self.in_channels, "k")
        v = _conv1x1(h, self.in_channels, "v")
        b, f, ht, w, c = q.shape
        q = q.reshape(b, f * ht * w, c)
        k = k.reshape(b, f * ht * w, c)
        v = v.reshape(b, f * ht * w, c)

        # Causal attention mask: query frame i may attend to all key
        # positions in frames <= i (full spatial attention within/across
        # allowed frames) -- port of `prepare_causal_attention_mask`.
        n_hw = ht * w
        frame_idx = jnp.repeat(jnp.arange(f), n_hw)
        allowed = frame_idx[None, :] <= frame_idx[:, None]  # (S, S), True = key visible to query

        scale = 1.0 / jnp.sqrt(jnp.array(c, dtype=jnp.float32))
        logits = jnp.einsum("bqc,bkc->bqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        logits = jnp.where(allowed[None, :, :], logits, -1e9)
        weights = jnn.softmax(logits, axis=-1).astype(v.dtype)
        attn = jnp.einsum("bqk,bkc->bqc", weights, v)
        attn = attn.reshape(b, f, ht, w, c)

        out = _conv1x1(attn, self.in_channels, "proj_out")
        return x + out


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = RMSNormLayer(name="norm1")(x)
        h = jnn.silu(h)
        h = causal_conv3d(h, self.out_channels, 3, "conv1")
        h = RMSNormLayer(name="norm2")(h)
        h = jnn.silu(h)
        h = causal_conv3d(h, self.out_channels, 3, "conv2")
        if self.in_channels != self.out_channels:
            x = _conv1x1(x, self.out_channels, "nin_shortcut")
        return x + h


def _split_channel_to_hw(x: jnp.ndarray, r2: int, r3: int) -> jnp.ndarray:
    """(b,f,h,w,c) -> (b,f,h//r2,w//r3,(r2 r3 c)); port of einops
    ``"b c f (h r2) (w r3) -> b (r2 r3 c) f h w"`` in channel-last layout.
    """
    b, f, h, w, c = x.shape
    x = x.reshape(b, f, h // r2, r2, w // r3, r3, c)
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # b f h0 w0 r2 r3 c
    return x.reshape(b, f, h // r2, w // r3, r2 * r3 * c)


def _split_channel_to_thw(x: jnp.ndarray, r1: int, r2: int, r3: int) -> jnp.ndarray:
    """(b,f,h,w,c) -> (b,f//r1,h//r2,w//r3,(r1 r2 r3 c)); port of einops
    ``"b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w"``.
    """
    b, f, h, w, c = x.shape
    x = x.reshape(b, f // r1, r1, h // r2, r2, w // r3, r3, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))  # b f0 h0 w0 r1 r2 r3 c
    return x.reshape(b, f // r1, h // r2, w // r3, r1 * r2 * r3 * c)


def _merge_hw_from_channel(x: jnp.ndarray, r2: int, r3: int) -> jnp.ndarray:
    """Inverse of ``_split_channel_to_hw``: (b,f,h,w,(r2 r3 c)) -> (b,f,h*r2,w*r3,c)."""
    b, f, h, w, C = x.shape
    c = C // (r2 * r3)
    x = x.reshape(b, f, h, w, r2, r3, c)
    x = jnp.transpose(x, (0, 1, 2, 4, 3, 5, 6))  # b f h r2 w r3 c
    return x.reshape(b, f, h * r2, w * r3, c)


def _merge_thw_from_channel(x: jnp.ndarray, r1: int, r2: int, r3: int) -> jnp.ndarray:
    """Inverse of ``_split_channel_to_thw``: (b,f,h,w,(r1 r2 r3 c)) -> (b,f*r1,h*r2,w*r3,c)."""
    b, f, h, w, C = x.shape
    c = C // (r1 * r2 * r3)
    x = x.reshape(b, f, h, w, r1, r2, r3, c)
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))  # b f r1 h r2 w r3 c
    return x.reshape(b, f * r1, h * r2, w * r3, c)


class Downsample(nn.Module):
    in_channels: int
    out_channels: int
    add_temporal_downsample: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        factor = 8 if self.add_temporal_downsample else 4
        assert self.out_channels % factor == 0
        conv_out = self.out_channels // factor
        group_size = factor * self.in_channels // self.out_channels

        h = causal_conv3d(x, conv_out, 3, "conv")

        if self.add_temporal_downsample:
            h_first = _split_channel_to_hw(h[:, :1], 2, 2)
            h_first = jnp.concatenate([h_first, h_first], axis=-1)
            h_next = _split_channel_to_thw(h[:, 1:], 2, 2, 2)
            h = jnp.concatenate([h_first, h_next], axis=1)  # concat along frame axis

            x_first = _split_channel_to_hw(x[:, :1], 2, 2)
            b, f, ht, w, C = x_first.shape
            x_first = x_first.reshape(b, f, ht, w, h.shape[-1], group_size // 2).mean(axis=-1)

            x_next = _split_channel_to_thw(x[:, 1:], 2, 2, 2)
            b, f, ht, w, C = x_next.shape
            x_next = x_next.reshape(b, f, ht, w, h.shape[-1], group_size).mean(axis=-1)
            shortcut = jnp.concatenate([x_first, x_next], axis=1)
        else:
            h = _split_channel_to_thw(h, 1, 2, 2)
            shortcut = _split_channel_to_thw(x, 1, 2, 2)
            b, f, ht, w, C = shortcut.shape
            shortcut = shortcut.reshape(b, f, ht, w, h.shape[-1], group_size).mean(axis=-1)

        return h + shortcut


class Upsample(nn.Module):
    in_channels: int
    out_channels: int
    add_temporal_upsample: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        factor = 8 if self.add_temporal_upsample else 4
        conv_out = self.out_channels * factor
        repeats = factor * self.out_channels // self.in_channels

        h = causal_conv3d(x, conv_out, 3, "conv")

        if self.add_temporal_upsample:
            h_first = _merge_hw_from_channel(h[:, :1], 2, 2)
            h_first = h_first[..., : h_first.shape[-1] // 2]
            h_next = _merge_thw_from_channel(h[:, 1:], 2, 2, 2)
            h = jnp.concatenate([h_first, h_next], axis=1)

            x_first = _merge_hw_from_channel(x[:, :1], 2, 2)
            x_first = jnp.repeat(x_first, repeats // 2, axis=-1)
            x_next = _merge_thw_from_channel(x[:, 1:], 2, 2, 2)
            x_next = jnp.repeat(x_next, repeats, axis=-1)
            shortcut = jnp.concatenate([x_first, x_next], axis=1)
        else:
            h = _merge_thw_from_channel(h, 1, 2, 2)
            shortcut = jnp.repeat(x, repeats, axis=-1)
            shortcut = _merge_thw_from_channel(shortcut, 1, 2, 2)

        return h + shortcut


class Encoder(nn.Module):
    in_channels: int
    z_channels: int
    block_out_channels: Tuple[int, ...]
    num_res_blocks: int
    ffactor_spatial: int
    ffactor_temporal: int
    downsample_match_channel: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = causal_conv3d(x, self.block_out_channels[0], 3, "conv_in")
        block_in = self.block_out_channels[0]

        for i_level, ch in enumerate(self.block_out_channels):
            block_out = ch
            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(block_in, block_out, name=f"down_{i_level}_block_{i_block}")(h)
                block_in = block_out
            add_spatial = i_level < math.log2(self.ffactor_spatial)
            add_temporal = add_spatial and i_level >= math.log2(self.ffactor_spatial // self.ffactor_temporal)
            if add_spatial or add_temporal:
                block_out = self.block_out_channels[i_level + 1] if self.downsample_match_channel else block_in
                h = Downsample(block_in, block_out, add_temporal, name=f"down_{i_level}_downsample")(h)
                block_in = block_out

        h = ResnetBlock(block_in, block_in, name="mid_block_1")(h)
        h = AttnBlock(block_in, name="mid_attn_1")(h)
        h = ResnetBlock(block_in, block_in, name="mid_block_2")(h)

        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        b, f, ht, w, C = h.shape
        shortcut = h.reshape(b, f, ht, w, 2 * self.z_channels, group_size).mean(axis=-1)
        h = RMSNormLayer(name="norm_out")(h)
        h = jnn.silu(h)
        h = causal_conv3d(h, 2 * self.z_channels, 3, "conv_out")
        return h + shortcut


def _decoder_channel_schedule(
    block_out_channels: Tuple[int, ...], ffactor_spatial: int, ffactor_temporal: int,
    upsample_match_channel: bool,
) -> Tuple[dict, ...]:
    """Pure-Python (no tensor ops) replay of ``Decoder.__call__``'s channel
    bookkeeping -- one entry per upsample level, giving the ``block_in``/
    ``block_out`` each level's ResnetBlocks use and whether/how much it
    upsamples. Lets ``stage_level`` reconstruct any single level's exact
    submodule shapes/names in isolation (see ``stage_level``'s docstring
    for why this staged-call decomposition exists at all).
    """
    schedule = []
    block_in = block_out_channels[0]  # after conv_in + mid (mid doesn't change channel count)
    for i_level, block_out in enumerate(block_out_channels):
        add_spatial = i_level < math.log2(ffactor_spatial)
        add_temporal = i_level < math.log2(ffactor_temporal)
        does_upsample = add_spatial or add_temporal
        if does_upsample:
            upsample_out = block_out_channels[i_level + 1] if upsample_match_channel else block_out
        else:
            upsample_out = block_out
        schedule.append(dict(
            block_in=block_in, block_out=block_out,
            does_upsample=does_upsample, upsample_out=upsample_out, add_temporal=add_temporal))
        block_in = upsample_out
    return tuple(schedule)


class Decoder(nn.Module):
    z_channels: int
    out_channels: int
    block_out_channels: Tuple[int, ...]  # already reversed by the caller
    num_res_blocks: int
    ffactor_spatial: int
    ffactor_temporal: int
    upsample_match_channel: bool = True

    def _schedule(self):
        return _decoder_channel_schedule(
            self.block_out_channels, self.ffactor_spatial, self.ffactor_temporal, self.upsample_match_channel)

    @nn.compact
    def stage_in_and_mid(self, z: jnp.ndarray) -> jnp.ndarray:
        """``conv_in`` + the z-repeat shortcut + the 3 mid blocks -- the
        first of several separately-callable decode stages, see
        ``stage_level``'s docstring for why this split exists.
        """
        block_in = self.block_out_channels[0]
        repeats = block_in // self.z_channels
        h = causal_conv3d(z, block_in, 3, "conv_in") + jnp.repeat(z, repeats, axis=-1)
        h = ResnetBlock(block_in, block_in, name="mid_block_1")(h)
        h = AttnBlock(block_in, name="mid_attn_1")(h)
        h = ResnetBlock(block_in, block_in, name="mid_block_2")(h)
        return h

    @nn.compact
    def stage_level(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        """One upsample level's ResnetBlocks + Upsample, as its own
        separately-jit-able call.

        **Why staged, not one fused `__call__`:** a full-resolution decode
        (e.g. 480x832x121) OOM'd inside a single `jax.jit`-traced
        `Decoder.__call__` even though the VAE's own weights are only
        ~2.5GB bf16 on an otherwise-dedicated chip -- the same root cause
        already documented for LTX-2.5's DiT in
        `docs/lessons/ltx2_5_debugging.md`: XLA does not free one level's
        intermediate activations before the next level's ops run inside a
        single fused program, so peak memory grows with the number of
        levels rather than being bounded by any one level's own compute.
        Calling each level as its own top-level `jax.jit`'d
        `decoder.apply(params, h, i_level, method=Decoder.stage_level)`
        (see `examples/generate_hunyuan_video_1_5.py`'s `vae_decode`)
        gives every level's temporaries a real chance to be freed between
        levels, at the cost of `len(block_out_channels)` separate
        compiles instead of one.

        `i_level` must be a concrete Python int (not a traced array) --
        it's `static_argnames`'d at the `jax.jit` call site, since it
        picks which fixed submodule names get built this call (matching
        the corresponding names in the *full* decoder's own param tree,
        e.g. `up_2_block_1` -- passing the complete `vae_params` tree to
        every staged call works fine even though each call only touches
        the slice it needs, since Flax looks submodules up from the param
        tree by name, not by which other submodules happen to exist).
        """
        sched = self._schedule()[i_level]
        block_in, block_out = sched["block_in"], sched["block_out"]
        for i_block in range(self.num_res_blocks + 1):
            h = ResnetBlock(block_in, block_out, name=f"up_{i_level}_block_{i_block}")(h)
            block_in = block_out
        if sched["does_upsample"]:
            h = Upsample(block_in, sched["upsample_out"], sched["add_temporal"], name=f"up_{i_level}_upsample")(h)
        return h

    @nn.compact
    def stage_level_block(self, h: jnp.ndarray, i_level: int, i_block: int) -> jnp.ndarray:
        """One *single* ResnetBlock within a level, as its own
        separately-jit-able call -- a finer-grained sibling to
        `stage_level` for when even one whole level's 3 ResnetBlocks +
        Upsample together still don't fit in one program at real frame
        counts (confirmed: 121 frames at 480p OOM'd a whole `stage_level`
        call by only ~1.8GB, i.e. a single level's *own* temporaries, not
        just cross-level ones, weren't all being freed within that one
        call either -- same root cause, finer granularity). See
        `stage_level`'s docstring for the general reasoning; identical
        `i_level`/full-param-tree considerations apply here.
        """
        sched = self._schedule()[i_level]
        block_in = sched["block_in"] if i_block == 0 else sched["block_out"]
        return ResnetBlock(block_in, sched["block_out"], name=f"up_{i_level}_block_{i_block}")(h)

    @nn.compact
    def stage_level_upsample(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        """The `Upsample` half of a level, as its own separately-jit-able
        call -- see `stage_level_block`."""
        sched = self._schedule()[i_level]
        if not sched["does_upsample"]:
            return h
        return Upsample(sched["block_out"], sched["upsample_out"], sched["add_temporal"], name=f"up_{i_level}_upsample")(h)

    @nn.compact
    def stage_out(self, h: jnp.ndarray) -> jnp.ndarray:
        """Final ``norm_out`` + ``conv_out`` -- the last staged-decode call."""
        h = RMSNormLayer(name="norm_out")(h)
        h = jnn.silu(h)
        return causal_conv3d(h, self.out_channels, 3, "conv_out")

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Single-trace decode (used by the bit-exact verification suite
        and any caller not worried about the peak-memory issue
        `stage_level` documents) -- algebraically identical to running
        `stage_in_and_mid`/`stage_level` (for every level)/`stage_out` in
        sequence, since Flax submodule identity is name-addressed, not
        call-order-addressed, so splitting or fusing these calls changes
        nothing about which parameters get used where.
        """
        h = self.stage_in_and_mid(z)
        for i_level in range(len(self.block_out_channels)):
            h = self.stage_level(h, i_level)
        return self.stage_out(h)


def blend_h(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
    """Linear cross-fade of ``b``'s left edge into ``a``'s right edge along
    the width axis (channel-last: axis -2) -- port of
    ``AutoencoderKLConv3D.blend_h`` (channel-first there, blending along its
    last axis). Used by ``examples/generate_hunyuan_video_1_5.py``'s
    spatial-tiled decode to stitch adjacent tiles without a visible seam.
    """
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
    """Same as ``blend_h`` but along the height axis (channel-last: axis -3)
    -- port of ``AutoencoderKLConv3D.blend_v``."""
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    if blend_extent <= 0:
        return b
    weight = (jnp.arange(blend_extent, dtype=jnp.float32) / blend_extent).reshape(
        (1,) * (a.ndim - 3) + (blend_extent, 1, 1))
    a_edge = a[..., -blend_extent:, :, :].astype(jnp.float32)
    b_edge = b[..., :blend_extent, :, :].astype(jnp.float32)
    blended = (a_edge * (1 - weight) + b_edge * weight).astype(b.dtype)
    return jnp.concatenate([blended, b[..., blend_extent:, :, :]], axis=-3)


class HunyuanVideo15VAE(nn.Module):
    """Top-level channel-last KL-VAE, port of ``AutoencoderKLConv3D``
    (single full-volume tile, no spatial/temporal tiling -- see module
    docstring). ``scaling_factor``/``shift_factor`` are applied by the
    caller (matching diffusers convention: ``latents = (encode(x).mean -
    shift) * scale``; ``decode((latents / scale) + shift)``), not inside
    this module, so `encode`/`decode` operate on raw (unnormalized) latents.
    """
    in_channels: int
    out_channels: int
    latent_channels: int
    block_out_channels: Tuple[int, ...]
    layers_per_block: int
    ffactor_spatial: int
    ffactor_temporal: int
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True

    def setup(self):
        self.encoder = Encoder(
            self.in_channels, self.latent_channels, self.block_out_channels,
            self.layers_per_block, self.ffactor_spatial, self.ffactor_temporal,
            self.downsample_match_channel, name="encoder")
        self.decoder = Decoder(
            self.latent_channels, self.out_channels, tuple(reversed(self.block_out_channels)),
            self.layers_per_block, self.ffactor_spatial, self.ffactor_temporal,
            self.upsample_match_channel, name="decoder")

    def encode(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """x: (B, T, H, W, in_channels) -> (mean, logvar), each (B, T', H', W', latent_channels)."""
        h = self.encoder(x)
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """z: (B, T, H, W, latent_channels) -> (B, T*ffactor_temporal (approx), H*ffactor_spatial, W*ffactor_spatial, out_channels)."""
        return self.decoder(z)

    @property
    def num_decoder_levels(self) -> int:
        return len(self.block_out_channels)

    def decode_stage_in_and_mid(self, z: jnp.ndarray) -> jnp.ndarray:
        """Staged-decode entry point 1/3 -- see ``Decoder.stage_level``'s
        docstring for why a real (>2x) frame-count decode needs this split
        instead of plain ``decode``."""
        return self.decoder.stage_in_and_mid(z)

    def decode_stage_level(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        """Staged-decode entry point 2/3, called once per
        ``range(self.num_decoder_levels)``."""
        return self.decoder.stage_level(h, i_level)

    @property
    def num_blocks_per_level(self) -> int:
        return self.layers_per_block + 1

    def decode_stage_level_block(self, h: jnp.ndarray, i_level: int, i_block: int) -> jnp.ndarray:
        """Finer-grained alternative to `decode_stage_level` -- call once
        per ``range(self.num_blocks_per_level)`` per level, then
        `decode_stage_level_upsample` once per level. See
        `Decoder.stage_level_block`'s docstring for when this extra
        granularity is actually needed (very large frame counts)."""
        return self.decoder.stage_level_block(h, i_level, i_block)

    def decode_stage_level_upsample(self, h: jnp.ndarray, i_level: int) -> jnp.ndarray:
        return self.decoder.stage_level_upsample(h, i_level)

    def decode_stage_out(self, h: jnp.ndarray) -> jnp.ndarray:
        """Staged-decode entry point 3/3."""
        return self.decoder.stage_out(h)
