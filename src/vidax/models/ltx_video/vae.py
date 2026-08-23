"""LTX-Video 0.9.8 causal VAE (Flax/JAX), encoder and decoder.

A structural port of the reference PyTorch `CausalVideoAutoencoder`/
`Encoder`/`Decoder`/`ResnetBlock3D`/`UNetMidBlock3D`/`SpaceToDepthDownsample`/
`DepthToSpaceUpsample`/`CausalConv3d`/`PixelNorm`/`PixelShuffleND` from
`refs/LTX-Video-main/ltx_video/models/autoencoders/{causal_video_autoencoder,
causal_conv3d,pixel_norm,pixel_shuffle,vae}.py`, for the specific
architecture every released 0.9.8 checkpoint (2B and 13B) actually uses --
see `vidax.models.ltx_video.configs.VAE_CONFIG` (read from the checkpoints'
own embedded metadata): `norm_layer="pixel_norm"`, `patch_size=4`,
`latent_log_var="uniform"`, `use_quant_conv=False`, `causal_decoder=False`,
`timestep_conditioning=True`.

Simplifications made possible by that specific config (not general to every
config the reference supports):

- Every conv is `kernel_size=3`, `stride=1` -- the reference's `CausalConv3d`
  ignores whatever `stride=`/`padding=` its caller passes for the actual
  striding in this config (`Encoder`/`Decoder`/`ResnetBlock3D`/
  `SpaceToDepthDownsample`/`DepthToSpaceUpsample` never pass anything but
  the default `stride=1` here -- true spatial/temporal downsampling comes
  entirely from the pixel-(un)shuffle rearranges below, not from strided
  convolution). One `causal_conv3d` helper covers every conv site.
- Every `ResnetBlock3D` in this config has `in_channels == out_channels`
  (channel width only changes at `compress_*_res`/`compress_all` sites, and
  this config never uses the channel-changing `res_x_y` block type) -- so
  the reference's `conv_shortcut`/`norm3` are always the identity, and
  aren't ported at all (there are no such keys in the checkpoint to load).
- `dims=3` always, never `(2, 1)` -- `DualConv3d` is unreachable, not ported.
- Only `Decoder` is timestep-conditioned (`causal_decoder=False` just
  controls the *decoder's* own `causal=` flag at call time -- confusingly
  named, unrelated to whether the encoder is causal, which it always is).

The reference's `"(c p1 p2 p3)"`/`"(c p q)"`-style channel-merge rearranges
(`SpaceToDepthDownsample`/`DepthToSpaceUpsample`/`patchify`/`unpatchify`)
are all one pattern -- channel slowest, spatial/temporal sub-indices
fastest, in `(..., p_time, p_height, p_width)` order -- implemented once
below as `_merge_subpixel`/`_split_subpixel` (channels-last, unlike the
reference's channel-first tensors) and verified against a brute-force
nested-loop transcription of the reference's einops patterns before use
(see `docs/lessons/ltx_video_debugging.md`).

Unlike `vidax.models.wan.wan2_1.vae` (which streams one latent frame at a
time to bound peak HBM, because the reference itself is only ever called
that way), this v1 port runs the whole encode/decode in a single forward
pass over the full tensor -- matching the reference's own default,
non-tiled code path (`enable_z_tiling`/`enable_hw_tiling` are opt-in and
off by default; see `AutoencoderKLWrapper`). Chunked/tiled decoding can be
added later the same way Wan's was, if a resolution/frame count needs it.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp


def _pixel_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """`PixelNorm(dim=1)`, adapted to this port's channels-last (B, F, H, W,
    C) layout (channel is the *last* axis here, not axis 1) -- no learnable
    params.
    """
    return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def _causal_pad_time(x: jnp.ndarray, num_frames: int) -> jnp.ndarray:
    """Prepends `num_frames` copies of the first frame along the time axis
    (axis 1 in (B, F, H, W, C)).
    """
    first = jnp.repeat(x[:, :1], num_frames, axis=1)
    return jnp.concatenate([first, x], axis=1)


def causal_conv3d(x: jnp.ndarray, out_channels: int, name: str, causal: bool = True) -> jnp.ndarray:
    """`CausalConv3d(kernel_size=3, stride=1)`. Temporal padding depends on
    `causal` (the reference's own `forward(x, causal=True)` argument, fed
    from each caller's own `causal=` config -- see `Encoder`/`Decoder`):
    `causal=True` front-replicates 2 frames (`time_kernel_size - 1`);
    `causal=False` (every released checkpoint's *decoder*, per
    `causal_decoder=False`) symmetrically replicates 1 frame on each side
    instead. Spatial padding is always symmetric, 1 each side (`kernel_size
    // 2`) -- covers every conv site in this VAE (see module docstring for
    why `stride` is always 1 here).
    """
    if causal:
        x = _causal_pad_time(x, num_frames=2)
    else:
        first = x[:, :1]
        last = x[:, -1:]
        x = jnp.concatenate([first, x, last], axis=1)
    return nn.Conv(
        out_channels, (3, 3, 3), strides=(1, 1, 1),
        padding=[(0, 0), (1, 1), (1, 1)], name=name)(x)


def _merge_subpixel(x: jnp.ndarray, pt: int, ph: int, pw: int) -> jnp.ndarray:
    """Channels-last transcription of the reference's `"b c (d p1) (h p2)
    (w p3) -> b (c p1 p2 p3) d h w"` (`SpaceToDepthDownsample`'s two
    rearranges, and `patchify` with `pt=1`): `(B, D, H, W, C) -> (B, D/pt,
    H/ph, W/pw, C*pt*ph*pw)`, channel axis ordered `(c, pt, ph, pw)` (c
    slowest, pw fastest). Verified against a brute-force nested-loop
    reimplementation of the einops pattern -- see module docstring.
    """
    b, d, h, w, c = x.shape
    x = x.reshape(b, d // pt, pt, h // ph, ph, w // pw, pw, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))  # (b, d', h', w', c, pt, ph, pw)
    return x.reshape(b, d // pt, h // ph, w // pw, c * pt * ph * pw)


def _split_subpixel(x: jnp.ndarray, pt: int, ph: int, pw: int) -> jnp.ndarray:
    """Inverse of `_merge_subpixel`: the reference's `"b (c p1 p2 p3) d h w
    -> b c (d p1) (h p2) (w p3)"` (`PixelShuffleND`/`unpatchify`).
    """
    b, d, h, w, cppp = x.shape
    c = cppp // (pt * ph * pw)
    x = x.reshape(b, d, h, w, c, pt, ph, pw)
    x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))  # (b, d, pt, h, ph, w, pw, c)
    return x.reshape(b, d * pt, h * ph, w * pw, c)


class ResnetBlock3D(nn.Module):
    """`in_channels == out_channels` always in this config -- see module
    docstring (no `conv_shortcut`/`norm3`).
    """
    channels: int
    timestep_conditioning: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timestep_embed: Optional[jnp.ndarray] = None, causal: bool = True,
    ) -> jnp.ndarray:
        residual = x
        h = _pixel_norm(x)
        if self.timestep_conditioning:
            scale_shift_table = self.param(
                "scale_shift_table", nn.initializers.normal(stddev=self.channels ** -0.5), (4, self.channels))
            ada = scale_shift_table.astype(jnp.float32) + timestep_embed.astype(jnp.float32)
            shift1, scale1, shift2, scale2 = ada[..., 0, :], ada[..., 1, :], ada[..., 2, :], ada[..., 3, :]
            h = (h.astype(jnp.float32) * (1 + scale1) + shift1).astype(x.dtype)
        h = nn.silu(h)
        h = causal_conv3d(h, self.channels, name="conv1", causal=causal)

        h = _pixel_norm(h)
        if self.timestep_conditioning:
            h = (h.astype(jnp.float32) * (1 + scale2) + shift2).astype(x.dtype)
        h = nn.silu(h)
        h = causal_conv3d(h, self.channels, name="conv2", causal=causal)
        return residual + h


def _get_timestep_sinusoidal_embedding(timesteps: jnp.ndarray, num_channels: int = 256) -> jnp.ndarray:
    """Same formula as `vidax.models.ltx_video.dit`'s copy -- kept
    duplicated rather than shared to avoid a VAE<->DiT import dependency
    for one ~6-line function; see that module for the formula's provenance.
    """
    half_dim = num_channels // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)  # flip_sin_to_cos


class UNetMidBlock3D(nn.Module):
    """A stack of `num_layers` `ResnetBlock3D`s (the `"res_x"` block type),
    sharing one `time_embedder` (if `timestep_conditioning`) computed once
    and reused by every resnet in the stack -- matches the checkpoint's
    `up_blocks.{i}.time_embedder.*` (singular) vs.
    `up_blocks.{i}.res_blocks.{j}.scale_shift_table` (per-resnet).
    """
    channels: int
    num_layers: int
    timestep_conditioning: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timestep: Optional[jnp.ndarray] = None, causal: bool = True,
    ) -> jnp.ndarray:
        timestep_embed = None
        if self.timestep_conditioning:
            embed_dim = self.channels * 4
            t_sin = _get_timestep_sinusoidal_embedding(timestep.reshape(-1), 256).astype(x.dtype)
            t_emb = nn.Dense(embed_dim, name="time_embedder_timestep_embedder_linear_1")(t_sin)
            t_emb = nn.silu(t_emb)
            t_emb = nn.Dense(embed_dim, name="time_embedder_timestep_embedder_linear_2")(t_emb)
            # (B, embed_dim) -> (B, 1, 1, 1, 4, channels): broadcasts over
            # (F, H, W); the leading "4" splits shift1/scale1/shift2/scale2.
            timestep_embed = t_emb.reshape(t_emb.shape[0], 1, 1, 1, 4, self.channels)

        for i in range(self.num_layers):
            x = ResnetBlock3D(
                self.channels, timestep_conditioning=self.timestep_conditioning,
                name=f"res_blocks_{i}")(x, timestep_embed, causal=causal)
        return x


class SpaceToDepthDownsample(nn.Module):
    """Encoder-only downsampling block (`"compress_{space,time,all}_res"`):
    a residual pixel-unshuffle path plus a causal-conv-refined path, each
    reduced/expanded to `out_channels` and summed. Transcribed 1:1 from the
    reference's `forward` using the verified `_merge_subpixel` helper.
    """
    in_channels: int
    out_channels: int
    stride: Tuple[int, int, int]

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal: bool = True) -> jnp.ndarray:
        pt, ph, pw = self.stride
        prod_stride = pt * ph * pw
        if pt == 2:
            x = _causal_pad_time(x, num_frames=1)

        group_size = self.in_channels * prod_stride // self.out_channels
        x_in = _merge_subpixel(x, pt, ph, pw)
        x_in = x_in.reshape(*x_in.shape[:-1], self.out_channels, group_size)
        x_in = jnp.mean(x_in, axis=-1)

        conv_out_channels = self.out_channels // prod_stride
        y = causal_conv3d(x, conv_out_channels, name="conv", causal=causal)
        y = _merge_subpixel(y, pt, ph, pw)

        return y + x_in


class DepthToSpaceUpsample(nn.Module):
    """Decoder-only upsampling block (`"compress_all"`, `residual=True` in
    this config). Transcribed 1:1 from the reference's `forward` using the
    verified `_split_subpixel` helper.
    """
    in_channels: int
    stride: Tuple[int, int, int]
    out_channels_reduction_factor: int = 1
    residual: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal: bool = True) -> jnp.ndarray:
        pt, ph, pw = self.stride
        prod_stride = pt * ph * pw
        out_channels = prod_stride * self.in_channels // self.out_channels_reduction_factor

        y = causal_conv3d(x, out_channels, name="conv", causal=causal)
        y = _split_subpixel(y, pt, ph, pw)
        if pt == 2:
            y = y[:, 1:]

        if not self.residual:
            return y

        num_repeat = prod_stride // self.out_channels_reduction_factor
        x_in = _split_subpixel(x, pt, ph, pw)
        x_in = jnp.tile(x_in, (1, 1, 1, 1, num_repeat))
        if pt == 2:
            x_in = x_in[:, 1:]

        return y + x_in


def _patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Encoder input pixel-unshuffle (`patch_size_t=1` always): `(B, F, H,
    W, C) -> (B, F, H/p, W/p, C*p*p)`.

    The reference's top-level `patchify`/`unpatchify` (distinct from
    `PixelShuffleND`, used by `SpaceToDepthDownsample`/
    `DepthToSpaceUpsample` above) merge the channel axis in `(c, p_time,
    p_width, p_height)` order -- width *before* height -- unlike
    `PixelShuffleND`'s natural `(c, p_time, p_height, p_width)` order that
    `_merge_subpixel`/`_split_subpixel` implement (`rearrange(x, "b c (f p)
    (h q) (w r) -> b (c p r q) f h w")`: `r` (width) precedes `q` (height)
    in the merge group despite `q`/`h` appearing first in the un-merged
    pattern). Verified against a brute-force nested-loop transcription of
    that exact einops pattern -- see `docs/lessons/ltx_video_debugging.md`.
    Reusing `_merge_subpixel` here directly (as an earlier version of this
    file did) silently swaps height and width, which still produces
    correctly-*shaped* output -- only visibly wrong once compared
    numerically against the reference, not from a shape-mismatch crash.
    """
    if patch_size == 1:
        return x
    b, f, h, w, c = x.shape
    ph = pw = patch_size
    x = x.reshape(b, f, 1, h // ph, ph, w // pw, pw, c)  # leading "1" is the (always-1) time-patch axis.
    x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 6, 4))  # (b, f, h', w', c, p_time, p_width, p_height)
    return x.reshape(b, f, h // ph, w // pw, c * ph * pw)


def _unpatchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Inverse of `_patchify` (decoder output). Transposes are the exact
    inverse of `_patchify`'s, re-derived (not assumed) and roundtrip-tested
    -- see `docs/lessons/ltx_video_debugging.md`.
    """
    if patch_size == 1:
        return x
    b, f, h, w, cpp = x.shape
    ph = pw = patch_size
    c = cpp // (ph * pw)
    x = x.reshape(b, f, h, w, c, 1, pw, ph)  # trailing "1" is the (always-1) time-patch axis.
    x = jnp.transpose(x, (0, 1, 5, 2, 7, 3, 6, 4))  # (b, f, p_time, h, p_height, w, p_width, c)
    return x.reshape(b, f * 1, h * ph, w * pw, c)


class Encoder(nn.Module):
    """Never timestep-conditioned -- see module docstring."""
    in_channels: int
    base_channels: int
    encoder_blocks: Tuple
    patch_size: int
    latent_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _patchify(x, self.patch_size)
        channels = self.base_channels
        x = causal_conv3d(x, channels, name="conv_in")

        for i, (block_name, block_params) in enumerate(self.encoder_blocks):
            if block_name == "res_x":
                x = UNetMidBlock3D(channels, block_params["num_layers"], name=f"down_blocks_{i}")(x)
            elif block_name in ("compress_space_res", "compress_time_res", "compress_all_res"):
                multiplier = block_params.get("multiplier", 2)
                stride = {
                    "compress_space_res": (1, 2, 2),
                    "compress_time_res": (2, 1, 1),
                    "compress_all_res": (2, 2, 2),
                }[block_name]
                out_channels = channels * multiplier
                x = SpaceToDepthDownsample(
                    channels, out_channels, stride, name=f"down_blocks_{i}")(x)
                channels = out_channels
            else:
                raise ValueError(f"unsupported encoder block: {block_name}")

        x = _pixel_norm(x)
        x = nn.silu(x)
        # `latent_log_var="uniform"`: conv_out produces latent_channels + 1
        # (one shared log-var channel), then that single channel is
        # repeated to fill out a full second `latent_channels`-wide half --
        # see `Encoder.forward`'s `latent_log_var == "uniform"` branch.
        x = causal_conv3d(x, self.latent_channels + 1, name="conv_out")
        last_channel = x[..., -1:]
        repeated = jnp.repeat(last_channel, self.latent_channels - 1, axis=-1)
        return jnp.concatenate([x, repeated], axis=-1)


class Decoder(nn.Module):
    latent_channels: int
    out_channels: int
    base_channels: int
    decoder_blocks: Tuple
    patch_size: int
    timestep_conditioning: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, timestep: jnp.ndarray) -> jnp.ndarray:
        # Output channel count of the *last* block (closest to conv_in) --
        # the reference computes this by walking `blocks` in reverse before
        # building any layers (`Decoder.__init__`'s own pre-pass).
        channels = self.base_channels
        for block_name, block_params in reversed(self.decoder_blocks):
            if block_name.startswith("compress"):
                channels *= block_params.get("multiplier", 1)

        # `causal_decoder=False` for every released checkpoint -- the
        # decoder's own convs (unlike the encoder's, which are always
        # causal) symmetrically replicate 1 frame on each side instead of
        # front-only replicating 2 -- see `causal_conv3d`'s docstring. A
        # real bug during this port: getting this wrong produced
        # plausible-shaped but numerically wrong output throughout the
        # whole decoder (see `docs/lessons/ltx_video_debugging.md`).
        causal = False
        x = causal_conv3d(x, channels, name="conv_in", causal=causal)

        # `timestep_scale_multiplier` is a *learned* scalar (unlike the
        # DiT's fixed constant 1000) -- initialize to 1000.0 to match the
        # reference's own init, but the checkpoint's trained value always
        # overrides this via the translator.
        timestep_scale_multiplier = self.param(
            "timestep_scale_multiplier", nn.initializers.constant(1000.0), ())
        timestep = timestep.astype(jnp.float32) * timestep_scale_multiplier

        # The reference builds `up_blocks` by iterating `reversed(blocks)`
        # (see `Decoder.__init__`) -- for every released checkpoint's
        # palindromic `decoder_blocks`, reversed happens to equal the
        # original order, but iterate `reversed(...)` explicitly anyway to
        # match the reference's actual algorithm rather than rely on that
        # coincidence.
        for i, (block_name, block_params) in enumerate(reversed(self.decoder_blocks)):
            if block_name == "res_x":
                x = UNetMidBlock3D(
                    channels, block_params["num_layers"],
                    timestep_conditioning=self.timestep_conditioning,
                    name=f"up_blocks_{i}")(x, timestep, causal=causal)
            elif block_name == "compress_all":
                multiplier = block_params.get("multiplier", 1)
                out_channels_next = channels // multiplier
                x = DepthToSpaceUpsample(
                    channels, (2, 2, 2), out_channels_reduction_factor=multiplier,
                    residual=block_params.get("residual", False), name=f"up_blocks_{i}")(x, causal=causal)
                channels = out_channels_next
            else:
                raise ValueError(f"unsupported decoder block: {block_name}")

        x = _pixel_norm(x)

        # Final timestep-conditioned AdaLN (the decoder's own, separate
        # from any `UNetMidBlock3D`'s -- `last_time_embedder`/
        # `last_scale_shift_table` in the checkpoint).
        embed_dim = channels * 2
        t_sin = _get_timestep_sinusoidal_embedding(timestep.reshape(-1), 256).astype(x.dtype)
        t_emb = nn.Dense(embed_dim, name="last_time_embedder_timestep_embedder_linear_1")(t_sin)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(embed_dim, name="last_time_embedder_timestep_embedder_linear_2")(t_emb)
        last_scale_shift_table = self.param(
            "last_scale_shift_table", nn.initializers.normal(stddev=channels ** -0.5), (2, channels))
        t_emb = t_emb.reshape(t_emb.shape[0], 1, 1, 1, 2, channels)
        ada = last_scale_shift_table.astype(jnp.float32) + t_emb.astype(jnp.float32)
        shift, scale = ada[..., 0, :], ada[..., 1, :]
        x = (x.astype(jnp.float32) * (1 + scale) + shift).astype(x.dtype)

        x = nn.silu(x)
        x = causal_conv3d(x, self.out_channels * self.patch_size ** 2, name="conv_out", causal=causal)
        return _unpatchify(x, self.patch_size)


class LTXVAE(nn.Module):
    """Top-level `CausalVideoAutoencoder`: `encode` (deterministic-sample
    Gaussian posterior, matching `vae_encode`'s `.sample()` call) and
    `decode` (requires `timestep` -- always `timestep_conditioning=True`
    for every released checkpoint).

    Config-driven -- pass `vidax.models.ltx_video.configs.VAE_CONFIG` (same
    for both the 2B and 13B checkpoints) as constructor kwargs.
    """
    latent_channels: int = 128
    encoder_blocks: Tuple = ()
    decoder_blocks: Tuple = ()
    patch_size: int = 4
    base_channels: int = 128

    def setup(self):
        self.encoder = Encoder(
            in_channels=3, base_channels=self.base_channels, encoder_blocks=self.encoder_blocks,
            patch_size=self.patch_size, latent_channels=self.latent_channels, name="encoder")
        self.decoder = Decoder(
            latent_channels=self.latent_channels, out_channels=3, base_channels=self.base_channels,
            decoder_blocks=self.decoder_blocks, patch_size=self.patch_size,
            timestep_conditioning=True, name="decoder")

    def encode(self, x: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, F, H, W, 3) RGB video/image in [-1, 1].
            noise: same shape as the returned mean -- standard normal noise
                for the posterior sample (`DiagonalGaussianDistribution
                .sample()`; caller supplies it so this stays a pure
                function under `jax.jit`).

        Returns:
            (B, F', H', W', latent_channels) sampled latent (unnormalized
            -- see `examples/generate_ltx_video.py` for per-channel
            normalization using the checkpoint's own statistics).
        """
        moments = self.encoder(x)
        mean, log_var = jnp.split(moments, 2, axis=-1)
        log_var = jnp.clip(log_var, -30.0, 20.0)
        std = jnp.exp(0.5 * log_var)
        return mean + std * noise

    def decode(self, z: jnp.ndarray, timestep: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: (B, F', H', W', latent_channels) unnormalized latent.
            timestep: (B,) decoder noise-conditioning timestep (the
                reference's `decode_timestep`, typically a small constant
                like 0.05 -- see `examples/generate_ltx_video.py`).

        Returns:
            (B, F, H, W, 3) RGB video in [-1, 1].
        """
        return self.decoder(z, timestep)
