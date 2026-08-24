"""LTX-2.5 causal VAE (Flax/JAX), conv decoder variant, encoder and decoder.

A structural port of the reference PyTorch `VideoEncoder`/`ConvVideoDecoder`/
`ResnetBlock3D`/`UNetMidBlock3D`/`SpaceToDepthDownsample`/
`DepthToSpaceUpsample`/`CausalConv3d`/`PixelNorm`/`PixelShuffleND` from
`refs/LTX-2-main/packages/ltx-core/src/ltx_core/model/video_vae/
{video_vae,conv_video_decoder,resnet,convolution,ops}.py`, for the specific
config the released `ltx-2.5-video-vae-conv-bf16.safetensors` checkpoint
uses -- see `vidax.models.ltx2_5.configs.VAE_CONFIG` (read from the
checkpoint's own embedded metadata during this port):
`norm_layer="pixel_norm"`, `patch_size=4`, `latent_log_var="uniform"`,
`causal_decoder=False`, **`timestep_conditioning=False`**,
`spatial_padding_mode="zeros"`.

This is architecturally the same family as `vidax.models.ltx_video.vae`
(same causal-conv3d/pixel-norm/pixel-shuffle machinery, same block-type
vocabulary), not shared code with it because of one real, config-driven
difference this module threads as an actual conditional rather than
hardcoding one way:

- **Self-normalizing, deterministic encode.** Unlike
  `vidax.models.ltx_video.vae` (external caller samples from `(mean,
  log_var)` and normalizes separately), LTX-2.5's real `VideoEncoder.
  forward` discards its own log-var estimate and returns only
  `per_channel_statistics.normalize(mean)` -- no exposed sampling.
  `Decoder` correspondingly un-normalizes internally before `conv_in`. See
  `Encoder`'s own docstring below -- caught only by running the real
  reference on random input, not from reading the class alone (a naive
  `chunk(2, dim=1)` on the raw encoder output, by analogy with LTX-Video,
  silently produces a wrong-but-plausible-shaped half-sized-channel
  result instead of erroring).
- **`timestep_conditioning=False`.** LTX-Video's VAE is *always*
  timestep-conditioned (noise-augmented decoder input, a final AdaLN block
  driven by a learned `last_time_embedder`/`last_scale_shift_table`), so
  that port hardcodes the conditioned path unconditionally. LTX-2.5's conv
  decoder checkpoint has **no** such parameters at all -- decoding is a
  plain forward pass, no `timestep` argument, no noise injection, no final
  AdaLN. Passing `timestep_conditioning=True` here anyway would look up
  parameters the checkpoint doesn't have.

The pixel-(un)shuffle/patchify machinery, per-block config-driven Encoder/
Decoder construction, and the `causal` calling convention (encoder always
causal; decoder's own convs symmetric, matching `causal_decoder=False`) are
otherwise identical in spirit to `vidax.models.ltx_video.vae` -- see that
module's docstring for the underlying einops-order details already verified
there (`_merge_subpixel`/`_split_subpixel` vs. the top-level `_patchify`/
`_unpatchify`'s different width/height merge order).
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


def _pixel_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """`PixelNorm(dim=1)`, channels-last (B, F, H, W, C) here -- no params."""
    return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def _causal_pad_time(x: jnp.ndarray, num_frames: int) -> jnp.ndarray:
    first = jnp.repeat(x[:, :1], num_frames, axis=1)
    return jnp.concatenate([first, x], axis=1)


def causal_conv3d(
    x: jnp.ndarray, out_channels: int, name: str, causal: bool = True,
    precision: "jax.lax.PrecisionLike" = None,
) -> jnp.ndarray:
    """`CausalConv3d(kernel_size=3, stride=1)`. See
    `vidax.models.ltx_video.vae.causal_conv3d`'s docstring for the
    `causal=True`/`False` temporal-padding distinction -- identical here.

    `precision`: pass `jax.lax.Precision.HIGHEST` for verification against
    the real reference; `None` (ambient default) for production bf16 TPU
    speed. Spatial padding below is applied via an explicit `jnp.pad` +
    `padding="VALID"` conv, **not** `nn.Conv`'s own `padding=[(0, 0), (1,
    1), (1, 1)]` list-spec -- confirmed as a real, reproducible XLA
    quirk, not a verification-only artifact: with the list-spec, `lax.
    conv_general_dilated` (whether called directly or through `nn.Conv`)
    silently gives a *different* result than explicit-pad-then-`"VALID"`
    for this VAE's largest decoder upsample conv (256->512 channels) --
    diverging from the real reference by up to ~6.0 absolute even with
    `precision=jax.lax.Precision.HIGHEST` explicitly set on the list-spec
    path, while the explicit-pad path matches the reference to `~1e-12`
    (verified independently via a brute-force manual numpy convolution at
    several output positions, not just a two-framework coincidence). Since
    this reproduces at `precision=HIGHEST`, it isn't only a low-precision
    rounding difference -- there is no principled reason to trust the
    list-spec path is safe at bf16 production precision either, so the
    explicit-pad form is used unconditionally, not gated behind
    verification.
    """
    if causal:
        x = _causal_pad_time(x, num_frames=2)
    else:
        first = x[:, :1]
        last = x[:, -1:]
        x = jnp.concatenate([first, x, last], axis=1)
    x = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
    return nn.Conv(
        out_channels, (3, 3, 3), strides=(1, 1, 1),
        padding="VALID", precision=precision, name=name)(x)


def _merge_subpixel(x: jnp.ndarray, pt: int, ph: int, pw: int) -> jnp.ndarray:
    """See `vidax.models.ltx_video.vae._merge_subpixel` -- identical."""
    b, d, h, w, c = x.shape
    x = x.reshape(b, d // pt, pt, h // ph, ph, w // pw, pw, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
    return x.reshape(b, d // pt, h // ph, w // pw, c * pt * ph * pw)


def _split_subpixel(x: jnp.ndarray, pt: int, ph: int, pw: int) -> jnp.ndarray:
    """See `vidax.models.ltx_video.vae._split_subpixel` -- identical."""
    b, d, h, w, cppp = x.shape
    c = cppp // (pt * ph * pw)
    x = x.reshape(b, d, h, w, c, pt, ph, pw)
    x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    return x.reshape(b, d * pt, h * ph, w * pw, c)


class ResnetBlock3D(nn.Module):
    channels: int
    timestep_conditioning: bool = False
    precision: "jax.lax.PrecisionLike" = None

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
        h = causal_conv3d(h, self.channels, name="conv1", causal=causal, precision=self.precision)

        h = _pixel_norm(h)
        if self.timestep_conditioning:
            h = (h.astype(jnp.float32) * (1 + scale2) + shift2).astype(x.dtype)
        h = nn.silu(h)
        h = causal_conv3d(h, self.channels, name="conv2", causal=causal, precision=self.precision)
        return residual + h


def _get_timestep_sinusoidal_embedding(timesteps: jnp.ndarray, num_channels: int = 256) -> jnp.ndarray:
    half_dim = num_channels // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)


class UNetMidBlock3D(nn.Module):
    channels: int
    num_layers: int
    timestep_conditioning: bool = False
    precision: "jax.lax.PrecisionLike" = None

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
            timestep_embed = t_emb.reshape(t_emb.shape[0], 1, 1, 1, 4, self.channels)

        for i in range(self.num_layers):
            x = ResnetBlock3D(
                self.channels, timestep_conditioning=self.timestep_conditioning, precision=self.precision,
                name=f"res_blocks_{i}")(x, timestep_embed, causal=causal)
        return x


class SpaceToDepthDownsample(nn.Module):
    in_channels: int
    out_channels: int
    stride: Tuple[int, int, int]
    precision: "jax.lax.PrecisionLike" = None

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
        y = causal_conv3d(x, conv_out_channels, name="conv", causal=causal, precision=self.precision)
        y = _merge_subpixel(y, pt, ph, pw)

        return y + x_in


class DepthToSpaceUpsample(nn.Module):
    in_channels: int
    stride: Tuple[int, int, int]
    out_channels_reduction_factor: int = 1
    residual: bool = False
    precision: "jax.lax.PrecisionLike" = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal: bool = True) -> jnp.ndarray:
        pt, ph, pw = self.stride
        prod_stride = pt * ph * pw
        out_channels = prod_stride * self.in_channels // self.out_channels_reduction_factor

        y = causal_conv3d(x, out_channels, name="conv", causal=causal, precision=self.precision)
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
    """See `vidax.models.ltx_video.vae._patchify` -- identical width-before-
    height merge order, verified there against a brute-force transcription.
    """
    if patch_size == 1:
        return x
    b, f, h, w, c = x.shape
    ph = pw = patch_size
    x = x.reshape(b, f, 1, h // ph, ph, w // pw, pw, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 6, 4))
    return x.reshape(b, f, h // ph, w // pw, c * ph * pw)


def _unpatchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Inverse of `_patchify` -- see `vidax.models.ltx_video.vae._unpatchify`."""
    if patch_size == 1:
        return x
    b, f, h, w, cpp = x.shape
    ph = pw = patch_size
    c = cpp // (ph * pw)
    x = x.reshape(b, f, h, w, c, 1, pw, ph)
    x = jnp.transpose(x, (0, 1, 5, 2, 7, 3, 6, 4))
    return x.reshape(b, f * 1, h * ph, w * pw, c)


_BLOCK_STRIDES = {
    "compress_space_res": (1, 2, 2), "compress_time_res": (2, 1, 1), "compress_all_res": (2, 2, 2),
    "compress_space": (1, 2, 2), "compress_time": (2, 1, 1), "compress_all": (2, 2, 2),
}


class Encoder(nn.Module):
    """Never timestep-conditioned (matches the reference: only the decoder
    is ever conditioned).

    Unlike `vidax.models.ltx_video.vae.Encoder` (which returns raw,
    unnormalized `(mean, log_var)` moments for the *caller* to sample and
    normalize), LTX-2.5's real `VideoEncoder.forward` is deterministic and
    self-normalizing: it builds the `latent_log_var="uniform"` expanded
    `(means, repeated_logvar)` pair internally purely to reuse a
    `chunk(2)`-shaped code path, then **discards the log-var half entirely**
    and returns only `per_channel_statistics.normalize(means)` -- no
    exposed sampling at inference. Verified directly against the real
    reference (`ltx_core.model.video_vae.video_vae.VideoEncoder.forward`):
    calling it on random input returns a `(B, latent_channels, F', H', W')`
    tensor, not the `2*latent_channels`-wide moments an earlier version of
    this port assumed by analogy with LTX-Video's differently-shaped
    `Encoder`. `encode()` below has no `noise` argument as a result.
    """
    in_channels: int
    base_channels: int
    encoder_blocks: Tuple
    patch_size: int
    latent_channels: int
    precision: "jax.lax.PrecisionLike" = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = _patchify(x, self.patch_size)
        channels = self.base_channels
        x = causal_conv3d(x, channels, name="conv_in", precision=self.precision)

        for i, (block_name, block_params) in enumerate(self.encoder_blocks):
            if block_name == "res_x":
                x = UNetMidBlock3D(
                    channels, block_params["num_layers"], precision=self.precision, name=f"down_blocks_{i}")(x)
            elif block_name in _BLOCK_STRIDES:
                multiplier = block_params.get("multiplier", 2)
                out_channels = channels * multiplier
                x = SpaceToDepthDownsample(
                    channels, out_channels, _BLOCK_STRIDES[block_name], precision=self.precision,
                    name=f"down_blocks_{i}")(x)
                channels = out_channels
            else:
                raise ValueError(f"unsupported encoder block: {block_name}")

        x = _pixel_norm(x)
        x = nn.silu(x)
        # `latent_log_var="uniform"`: conv_out emits latent_channels + 1
        # channels; the last is discarded below (see class docstring), only
        # the mean half is ever used.
        x = causal_conv3d(x, self.latent_channels + 1, name="conv_out", precision=self.precision)
        mean = x[..., :self.latent_channels]

        stat_mean = self.param(
            "per_channel_statistics_mean", nn.initializers.zeros, (self.latent_channels,))
        stat_std = self.param(
            "per_channel_statistics_std", nn.initializers.ones, (self.latent_channels,))
        return (mean - stat_mean.astype(mean.dtype)) / stat_std.astype(mean.dtype)


class Decoder(nn.Module):
    """`timestep_conditioning=False` for the released conv-decoder
    checkpoint (see module docstring) -- noise injection and the final
    AdaLN block below are real conditionals, not always-on like
    `vidax.models.ltx_video.vae.Decoder`.
    """
    latent_channels: int
    out_channels: int
    base_channels: int
    decoder_blocks: Tuple
    patch_size: int
    causal_decoder: bool = False
    timestep_conditioning: bool = False
    precision: "jax.lax.PrecisionLike" = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, timestep: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        channels = self.base_channels
        for block_name, block_params in reversed(self.decoder_blocks):
            if block_name.startswith("compress"):
                channels *= block_params.get("multiplier", 1)

        # Un-normalize first (`ConvVideoDecoder.forward`'s own
        # `per_channel_statistics.un_normalize` call, before `conv_in`) --
        # `vidax.models.ltx2_5.vae.Encoder` normalizes on the way out, so
        # `decode` accepts the same normalized-latent convention `encode`
        # produces, with no external normalization step needed by callers
        # (unlike `vidax.models.ltx_video.vae`, where normalization is the
        # caller's responsibility -- see this module's docstring).
        stat_mean = self.param(
            "per_channel_statistics_mean", nn.initializers.zeros, (self.latent_channels,))
        stat_std = self.param(
            "per_channel_statistics_std", nn.initializers.ones, (self.latent_channels,))
        x = x * stat_std.astype(x.dtype) + stat_mean.astype(x.dtype)

        causal = self.causal_decoder
        x = causal_conv3d(x, channels, name="conv_in", causal=causal, precision=self.precision)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' must be provided when timestep_conditioning=True")
            timestep_scale_multiplier = self.param(
                "timestep_scale_multiplier", nn.initializers.constant(1000.0), ())
            scaled_timestep = timestep.astype(jnp.float32) * timestep_scale_multiplier

        for i, (block_name, block_params) in enumerate(reversed(self.decoder_blocks)):
            if block_name == "res_x":
                x = UNetMidBlock3D(
                    channels, block_params["num_layers"],
                    timestep_conditioning=self.timestep_conditioning, precision=self.precision,
                    name=f"up_blocks_{i}")(x, scaled_timestep, causal=causal)
            elif block_name in _BLOCK_STRIDES:
                multiplier = block_params.get("multiplier", 1)
                out_channels_next = channels // multiplier
                x = DepthToSpaceUpsample(
                    channels, _BLOCK_STRIDES[block_name], out_channels_reduction_factor=multiplier,
                    residual=block_params.get("residual", False), precision=self.precision,
                    name=f"up_blocks_{i}")(x, causal=causal)
                channels = out_channels_next
            else:
                raise ValueError(f"unsupported decoder block: {block_name}")

        x = _pixel_norm(x)

        if self.timestep_conditioning:
            embed_dim = channels * 2
            t_sin = _get_timestep_sinusoidal_embedding(scaled_timestep.reshape(-1), 256).astype(x.dtype)
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
        x = causal_conv3d(
            x, self.out_channels * self.patch_size ** 2, name="conv_out", causal=causal, precision=self.precision)
        return _unpatchify(x, self.patch_size)


class LTXVAE(nn.Module):
    """Top-level conv-decoder `CausalVideoAutoencoder`. Config-driven --
    pass `vidax.models.ltx2_5.configs.VAE_CONFIG` (read from the checkpoint's
    own embedded metadata) as constructor kwargs.

    `precision`: **verification only** -- see `causal_conv3d`'s docstring.
    Left `None` (fastest bf16 TPU path) for production; a bit-exact check
    against the real reference must pass `precision=jax.lax.Precision.
    HIGHEST` explicitly (the global `jax_default_matmul_precision` config
    is not sufficient for this VAE's convolutions).
    """
    latent_channels: int = 128
    encoder_blocks: Tuple = ()
    decoder_blocks: Tuple = ()
    patch_size: int = 4
    base_channels: int = 128
    causal_decoder: bool = False
    timestep_conditioning: bool = False
    precision: "jax.lax.PrecisionLike" = None

    def setup(self):
        self.encoder = Encoder(
            in_channels=3, base_channels=self.base_channels, encoder_blocks=self.encoder_blocks,
            patch_size=self.patch_size, latent_channels=self.latent_channels, precision=self.precision,
            name="encoder")
        self.decoder = Decoder(
            latent_channels=self.latent_channels, out_channels=3, base_channels=self.base_channels,
            decoder_blocks=self.decoder_blocks, patch_size=self.patch_size,
            causal_decoder=self.causal_decoder, timestep_conditioning=self.timestep_conditioning,
            precision=self.precision, name="decoder")

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, F, H, W, 3) RGB video/image in [-1, 1].

        Returns:
            (B, F', H', W', latent_channels) **normalized** latent mean --
            deterministic, no sampling (see `Encoder`'s docstring: the real
            `VideoEncoder.forward` discards its own log-var estimate and
            returns only `per_channel_statistics.normalize(mean)`).
        """
        return self.encoder(x)

    def decode(self, z: jnp.ndarray, timestep: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            z: (B, F', H', W', latent_channels) **normalized** latent (the
                same convention `encode` returns -- un-normalization happens
                internally, see `Decoder`).
            timestep: ignored -- `timestep_conditioning=False` for the
                released conv-decoder checkpoint (kept as an argument only
                for interface parity with `vidax.models.ltx_video.vae`).

        Returns:
            (B, F, H, W, 3) RGB video in [-1, 1].
        """
        return self.decoder(z, timestep)
