"""LTX-Video 0.9.8 Diffusion Transformer backbone (Flax/JAX), t2v and i2v.

A structural port of the reference PyTorch `Transformer3DModel`/
`BasicTransformerBlock`/`Attention` from `refs/LTX-Video-main/ltx_video/
models/transformers/{transformer3d,attention}.py`, plus diffusers'
`AdaLayerNormSingle`/`PixArtAlphaCombinedTimestepSizeEmbeddings`/
`PixArtAlphaTextProjection` (transcribed from the pinned `diffusers==0.26.3`
source in `refs/diffusers-ltx/` -- the reference itself imports these from
`diffusers`, not vendored code) and RMSNorm (`refs/diffusers-ltx/
normalization.py`).

Submodule names are chosen to match the reference's own PyTorch module
names one-for-one wherever possible (`patchify_proj`, `attn1`/`attn2` each
containing `to_q`/`to_k`/`to_v`/`to_out`, `caption_projection`, etc.) --
this keeps the translator (`vidax.translator.mappings.ltx_video`) a
near-mechanical prefix-strip, and means `to_q`/`to_k`/`to_v`/`to_out`
already fall under `vidax.core.sharding`'s existing Cosmos3-derived
`COLUMN_PARALLEL_NAMES`/`ROW_PARALLEL_NAMES` entries with no changes needed
there for attention (only the FFN's `ff_proj`/`ff_out` are LTX-specific
additions -- see that module).

Architectural notes not obvious from the reference's own code:

- RoPE (`vidax.models.ltx_video.rope`) is computed once over the model's
  *full* `inner_dim` and applied to query/key *before* the per-head
  reshape -- not the usual per-head convention. See that module's
  docstring.
- `norm1`/`norm2`/`norm_out` are RMSNorm with `elementwise_affine=False`
  for every released checkpoint (`norm_elementwise_affine: false` in every
  checkpoint's embedded config) -- i.e. plain normalization with no
  learnable scale, distinct from `q_norm`/`k_norm` (RMSNorm *with* a
  learnable scale, applied over the full un-split `inner_dim`, mirroring
  Wan's own QK-RMSNorm convention).
- Feed-forward uses `activation_fn="gelu-approximate"` (tanh-approximate
  GELU, single gate) for every released checkpoint, not GEGLU -- simpler
  than it looks from the reference's generic `FeedForward` class, which
  supports several activations.
- `timestep` may be `(B,)` (T2V: one timestep per sample) or `(B, N)`
  (I2V: one timestep per *token*, letting conditioning tokens sit at a
  different effective noise level than the rest -- see
  `examples/generate_ltx_video.py`). The AdaLN path below handles both
  via `.reshape(batch, -1, ...)`, exactly mirroring the reference.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from vidax.models.ltx_video.rope import apply_rope, create_ltx_rope_freqs


def _rms_norm_no_affine(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    """`RMSNorm(dim, eps, elementwise_affine=False)`: normalize only, no
    learnable scale -- used for every block's `norm1`/`norm2` and the
    model's final `norm_out` (LayerNorm in the reference, but with
    `elementwise_affine=False` it's numerically identical to RMSNorm-
    no-affine plus a mean-subtraction the reference's own LayerNorm still
    does; matched exactly below via `nn.LayerNorm` with `use_scale=False,
    use_bias=False` for `norm_out`, and plain RMS division for `norm1`/
    `norm2`, per each's own `standardization_norm`/`RMSNorm` class).
    """
    orig_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + eps)).astype(orig_dtype)


def _rms_norm_affine(x: jnp.ndarray, scale: jnp.ndarray, eps: float) -> jnp.ndarray:
    """`RMSNorm(dim, eps, elementwise_affine=True)`: LTX's `q_norm`/
    `k_norm`, applied over the full concatenated `inner_dim` (before the
    per-head reshape) -- eps is always 1e-5 for these regardless of the
    block-level `norm1`/`norm2` eps, matching `Attention.__init__`'s
    hardcoded `RMSNorm(dim_head * heads, eps=1e-5)`.
    """
    orig_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = x32 * jax.lax.rsqrt(var + eps)
    # Reference: casts to the weight's dtype only if that dtype is fp16/bf16,
    # otherwise keeps input_dtype -- weight here is always float, so this
    # simplifies to casting back to the input's own dtype either way.
    return (normed.astype(orig_dtype)) * scale.astype(orig_dtype)


def _get_timestep_sinusoidal_embedding(timesteps: jnp.ndarray, num_channels: int = 256) -> jnp.ndarray:
    """`diffusers.models.embeddings.get_timestep_embedding` with
    `flip_sin_to_cos=True, downscale_freq_shift=0, max_period=10000`
    (`Timesteps(num_channels=256, ...)`'s fixed call in
    `PixArtAlphaCombinedTimestepSizeEmbeddings`). `timesteps` is 1D.
    """
    half_dim = num_channels // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    # flip_sin_to_cos: [sin, cos] -> [cos, sin]
    emb = jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


class LTXAttention(nn.Module):
    """One `Attention` module (`attn1` = self-attn with RoPE, `attn2` =
    cross-attn, no RoPE) -- both share this class, matching the reference
    (`use_rope` is simply `False` for cross-attn calls since `freqs_cis`
    is only applied when `encoder_hidden_states is None`, i.e. self-attn).
    """
    inner_dim: int
    num_heads: int
    head_dim: int
    is_cross_attn: bool = False
    eps: float = 1e-5  # q_norm/k_norm eps -- always 1e-5, see `_rms_norm_affine`.
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        freqs: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        kv_input = encoder_hidden_states if self.is_cross_attn else hidden_states
        b, seq_q = hidden_states.shape[0], hidden_states.shape[1]

        q = nn.Dense(self.inner_dim, name="to_q")(hidden_states)
        q_scale = self.param("q_norm_scale", nn.initializers.ones, (self.inner_dim,))
        q = _rms_norm_affine(q, q_scale, self.eps)

        k = nn.Dense(self.inner_dim, name="to_k")(kv_input)
        k_scale = self.param("k_norm_scale", nn.initializers.ones, (self.inner_dim,))
        k = _rms_norm_affine(k, k_scale, self.eps)

        if not self.is_cross_attn:
            cos, sin = freqs
            k = apply_rope(k, cos, sin)
            q = apply_rope(q, cos, sin)

        v = nn.Dense(self.inner_dim, name="to_v")(kv_input)

        seq_k = k.shape[1]
        q = q.reshape(b, seq_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, seq_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, seq_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
        if encoder_attention_bias is not None:
            logits = logits + encoder_attention_bias
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, seq_q, self.inner_dim)

        out = nn.Dense(self.inner_dim, name="to_out_0")(out)
        return out


class LTXFeedForward(nn.Module):
    """`FeedForward` with `activation_fn="gelu-approximate"`: `net.0.proj`
    (Linear + tanh-approx GELU fused into one `GELU` submodule in the
    reference) -> `net.2` (Linear back down). No GEGLU gate -- see this
    module's file docstring.
    """
    dim: int
    inner_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.inner_dim, name="ff_proj")(x)
        h = nn.gelu(h, approximate=True)
        return nn.Dense(self.dim, name="ff_out")(h)


class LTXDiTBlock(nn.Module):
    """`BasicTransformerBlock`: self-attn (RoPE) -> cross-attn -> FFN, each
    AdaLN-modulated via this block's own `scale_shift_table` (6, dim)
    combined with the shared per-sample-or-per-token `timestep` embedding
    computed once at the model level (see `LTXDiT`).
    """
    dim: int
    num_heads: int
    head_dim: int
    ff_inner_dim: int
    cross_attention_dim: int
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_bias: Optional[jnp.ndarray],
        timestep: jnp.ndarray,
    ) -> jnp.ndarray:
        b = x.shape[0]
        scale_shift_table = self.param(
            "scale_shift_table", nn.initializers.normal(stddev=self.dim ** -0.5), (6, self.dim))
        ada = scale_shift_table[None, None].astype(jnp.float32) + timestep.astype(jnp.float32).reshape(
            b, timestep.shape[1], 6, self.dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            ada[:, :, i] for i in range(6)]

        norm_x = _rms_norm_no_affine(x, self.eps)
        norm_x = (norm_x.astype(jnp.float32) * (1 + scale_msa) + shift_msa).astype(self.compute_dtype)
        attn_out = LTXAttention(
            self.dim, self.num_heads, self.head_dim, is_cross_attn=False,
            compute_dtype=self.compute_dtype, name="attn1")(norm_x, freqs=freqs)
        x = x + (gate_msa * attn_out.astype(jnp.float32)).astype(x.dtype)

        attn2_out = LTXAttention(
            self.dim, self.num_heads, self.head_dim, is_cross_attn=True,
            compute_dtype=self.compute_dtype, name="attn2")(
                x, encoder_hidden_states=encoder_hidden_states,
                encoder_attention_bias=encoder_attention_bias)
        x = x + attn2_out

        norm_h = _rms_norm_no_affine(x, self.eps)
        norm_h = (norm_h.astype(jnp.float32) * (1 + scale_mlp) + shift_mlp).astype(self.compute_dtype)
        ff_out = LTXFeedForward(self.dim, self.ff_inner_dim, name="ff")(norm_h)
        x = x + (gate_mlp * ff_out.astype(jnp.float32)).astype(x.dtype)
        return x


class LTXDiT(nn.Module):
    """LTX-Video's `Transformer3DModel`. Config-driven -- pass
    `vidax.models.ltx_video.configs.DIT_2B_CONFIG`/`DIT_13B_CONFIG` (or the
    dict read directly from a checkpoint's own embedded metadata via
    `load_ltx_checkpoint_metadata`) as constructor kwargs.

    Built with `setup()` (not `@nn.compact`), mirroring `vidax.models.wan.
    wan2_1.dit.WanDiT`'s `pre_process`/block-loop/`post_process` split, for
    the same reason: letting `--offload_dit_weights` stream one block's
    parameters into HBM at a time instead of requiring the whole DiT
    resident. `__call__` chains all three in one pass, unchanged behavior.

    Unlike Wan, there is no separate patchify step here: the caller
    (`examples/generate_ltx_video.py`, via `vidax.models.ltx_video.
    patchifier`) is expected to already have reshaped the video latent into
    a `(B, N, in_channels)` token sequence before calling this model --
    `patchify_proj` is a plain linear projection of already-flattened
    tokens, not a learned spatial patchifier (LTX's actual spatial/temporal
    downsampling happens entirely in the VAE).
    """
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 28
    cross_attention_dim: int = 2048
    caption_channels: int = 4096
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    timestep_scale_multiplier: int = 1000
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.patchify_proj = nn.Dense(self.inner_dim, name="patchify_proj")
        self.adaln_timestep_embedder_linear_1 = nn.Dense(self.inner_dim, name="adaln_timestep_embedder_linear_1")
        self.adaln_timestep_embedder_linear_2 = nn.Dense(self.inner_dim, name="adaln_timestep_embedder_linear_2")
        self.adaln_linear = nn.Dense(6 * self.inner_dim, name="adaln_linear")
        self.caption_projection_linear_1 = nn.Dense(self.inner_dim, name="caption_projection_linear_1")
        self.caption_projection_linear_2 = nn.Dense(self.inner_dim, name="caption_projection_linear_2")
        self.proj_out = nn.Dense(self.out_channels, name="proj_out")
        self.norm_out = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=self.eps, name="norm_out")
        self.scale_shift_table = self.param(
            "scale_shift_table", nn.initializers.normal(stddev=self.inner_dim ** -0.5), (2, self.inner_dim))
        ff_inner_dim = self.inner_dim * 4
        self.blocks = [
            LTXDiTBlock(
                dim=self.inner_dim, num_heads=self.num_attention_heads,
                head_dim=self.attention_head_dim, ff_inner_dim=ff_inner_dim,
                cross_attention_dim=self.cross_attention_dim, eps=self.eps,
                compute_dtype=self.compute_dtype, name=f"blocks_{i}")
            for i in range(self.num_layers)
        ]

    def pre_process(
        self,
        latents: jnp.ndarray,
        latent_coords: jnp.ndarray,
        timestep: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
    ):
        """Everything before the block loop: `patchify_proj`, AdaLN
        timestep embedding, RoPE table, and caption projection. Returns
        `(x, freqs, encoder_hidden_states, encoder_attention_bias,
        timestep_mod, embedded_timestep, input_dtype)` -- the last two are
        only needed by `post_process`.

        `latents` is already the flattened `(B, N, in_channels)` token
        sequence (see class docstring); `latent_coords` is `(B, 3, N)`
        pixel-space coordinates (see `vidax.models.ltx_video.patchifier`)
        for RoPE. `timestep` is `(B,)` or `(B, N)` (see file docstring).
        """
        input_dtype = latents.dtype
        b = latents.shape[0]
        x = self.patchify_proj(latents)

        scaled_timestep = self.timestep_scale_multiplier * timestep
        freqs = create_ltx_rope_freqs(
            latent_coords, self.inner_dim, self.positional_embedding_theta,
            self.positional_embedding_max_pos, dtype=self.compute_dtype)

        t_flat = scaled_timestep.reshape(-1)
        t_sin = _get_timestep_sinusoidal_embedding(t_flat, 256).astype(self.compute_dtype)
        t_emb = self.adaln_timestep_embedder_linear_1(t_sin)
        t_emb = nn.silu(t_emb)
        embedded_timestep_flat = self.adaln_timestep_embedder_linear_2(t_emb)
        timestep_mod_flat = self.adaln_linear(nn.silu(embedded_timestep_flat))
        timestep_mod = timestep_mod_flat.reshape(b, -1, 6 * self.inner_dim)
        embedded_timestep = embedded_timestep_flat.reshape(b, -1, self.inner_dim)

        context = self.caption_projection_linear_1(encoder_hidden_states)
        context = nn.gelu(context, approximate=True)
        context = self.caption_projection_linear_2(context)
        context = context.reshape(b, -1, self.inner_dim)

        encoder_attention_bias = None
        if encoder_attention_mask is not None:
            encoder_attention_bias = (
                (1.0 - encoder_attention_mask.astype(jnp.float32)) * -10000.0)[:, None, None, :]

        return x, freqs, context, encoder_attention_bias, timestep_mod, embedded_timestep, input_dtype

    def post_process(self, x: jnp.ndarray, embedded_timestep: jnp.ndarray, input_dtype: jnp.dtype) -> jnp.ndarray:
        """Final AdaLN modulation + output projection (the reference's
        `scale_shift_table` (2, inner_dim), not a per-block one).
        """
        scale_shift_values = self.scale_shift_table[None, None].astype(jnp.float32) + embedded_timestep.astype(
            jnp.float32)[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = self.norm_out(x)
        x = (x.astype(jnp.float32) * (1 + scale) + shift).astype(self.compute_dtype)
        x = self.proj_out(x)
        return x.astype(input_dtype)

    def __call__(
        self,
        latents: jnp.ndarray,
        latent_coords: jnp.ndarray,
        timestep: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            latents: (B, N, in_channels) patchified video-latent tokens.
            latent_coords: (B, 3, N) pixel-space (f, h, w) coordinates per
                token, for RoPE (see `vidax.models.ltx_video.patchifier`).
            timestep: (B,) or (B, N) diffusion timesteps in [0, 1000]
                (already on the training scale, *before*
                `timestep_scale_multiplier` -- matching the reference's
                own convention despite the name; see `pre_process`).
            encoder_hidden_states: (B, L, caption_channels) T5 text
                embeddings.
            encoder_attention_mask: (B, L) 1 = keep, 0 = discard.

        Returns:
            (B, N, out_channels) velocity prediction.
        """
        x, freqs, context, encoder_attention_bias, timestep_mod, embedded_timestep, input_dtype = self.pre_process(
            latents, latent_coords, timestep, encoder_hidden_states, encoder_attention_mask)
        for block in self.blocks:
            x = block(x, freqs, context, encoder_attention_bias, timestep_mod)
        return self.post_process(x, embedded_timestep, input_dtype)
