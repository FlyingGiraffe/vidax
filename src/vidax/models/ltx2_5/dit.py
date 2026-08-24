"""LTX-2.5 Diffusion Transformer backbone (Flax/JAX), video-only, t2v and i2v.

A structural port of `ltx_core.model.transformer.{model,transformer,
transformer_args,attention,ops,adaln,timestep_embedding}` from
`refs/LTX-2-main/packages/ltx-core/src/ltx_core/model/transformer/`, scoped
to `LTXModelType.VideoOnly` (`LTXVideoOnlyModelConfigurator`) -- no audio
branches, no audio/video cross-attention. See `docs/models/ltx2_5.md` and
`vidax.models.ltx_video.dit` for the closely-related LTX-Video port this
mirrors structurally.

Differences from `vidax.models.ltx_video.dit.LTXDiT`, confirmed by reading
the reference *and* the real `ltx-2.5-22b-{dev,distilled}` checkpoints' own
embedded metadata (not assumed from the "same family" naming -- an earlier
draft of this file assumed several of these flags off, which the real
metadata contradicted):

- **RoPE.** `vidax.models.ltx2_5.rope`'s "split" (rotate-half, per-head)
  convention evaluated at each patch's midpoint, in float64 precision
  (`frequencies_precision: "float64"` in the real checkpoint config) --
  see that module's docstring.
- **`q_norm`/`k_norm` eps.** The checkpoint's own `norm_eps` (`1e-6`), not
  LTX-Video's hardcoded `1e-5`.
- **`cross_attention_adaln: true`** (real checkpoint value). Cross-attention
  gets its own AdaLN modulation: the block's `scale_shift_table` grows to
  `(9, dim)` (rows 6:9 are cross-attn query shift/scale/gate, on top of the
  usual self-attn 0:3 / FFN 3:6), and a *second*, model-level
  `prompt_adaln_single` embeds `sigma` (the scalar current denoising step,
  distinct from the possibly-per-token `timestep`) into a per-block
  `prompt_scale_shift_table (2, dim)`-modulated key/value transform. See
  `ltx_core.model.transformer.transformer.apply_cross_attention_adaln`.
- **`apply_gated_attention: true`** (real checkpoint value). Every
  `Attention` (self- and cross-) has an extra `to_gate_logits` Dense
  producing a per-head `2*sigmoid(...)` gate applied to the attention
  output before `to_out` -- `ltx_core.model.transformer.ops.
  PytorchGatedAttention`.
- **Cross-attention query input** is a weightless RMSNorm of the residual
  stream taken right after the self-attention residual add (`post_sa_
  function`'s second return value), not the raw residual.
- **No `caption_projection` in the DiT.** `encoder_hidden_states` arrives
  already projected to `cross_attention_dim` by the embeddings connector
  (`vidax.models.ltx2_5.connector`, part of the text-encoder pipeline, not
  this module) -- `caption_proj_before_connector: true`.
- **`use_keyframes_abs_pos_embedding: true`.** A trained `(1, inner_dim)`
  parameter added to tokens marked in an optional `keyframes_mask` --
  all-zero (no-op) for plain T2V/I2V with no generated-keyframe
  conditioning, but the parameter is present in the checkpoint and must be
  loaded regardless.

Everything else -- the PixArt-style `AdaLayerNormSingle` timestep embedding,
the `gelu-approximate` single-gate FeedForward, the `setup()`-based
pre_process/block-loop/post_process split for `--offload_dit_weights`
compatibility -- matches LTX-Video's port and is not re-derived here.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from vidax.models.ltx2_5.rope import apply_rope, create_ltx2_5_rope_freqs


def _rms_norm_no_affine(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Weightless RMSNorm -- `rms_norm(x, norm_weights=None, eps)`."""
    orig_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + eps)).astype(orig_dtype)


def _rms_norm_affine(x: jnp.ndarray, scale: jnp.ndarray, eps: float) -> jnp.ndarray:
    """`RMSNorm(inner_dim, eps=norm_eps)` -- LTX-2.5's `q_norm`/`k_norm`."""
    orig_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = x32 * jax.lax.rsqrt(var + eps)
    return normed.astype(orig_dtype) * scale.astype(orig_dtype)


def _get_timestep_sinusoidal_embedding(timesteps: jnp.ndarray, num_channels: int = 256) -> jnp.ndarray:
    """`Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)`."""
    half_dim = num_channels // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    emb = jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


def _pixart_timestep_embed(
    t_flat: jnp.ndarray, linear_1: nn.Dense, linear_2: nn.Dense, compute_dtype: jnp.dtype,
) -> jnp.ndarray:
    """`PixArtAlphaCombinedTimestepSizeEmbeddings`: sinusoidal(256) -> Linear
    -> SiLU -> Linear. `linear_1`/`linear_2` are pre-built `nn.Dense`
    submodules (declared in `LTXDiT.setup()`, not created here) -- `LTXDiT`
    uses `setup()` rather than `@nn.compact` (see its docstring), so
    submodule construction can't happen inside a plain helper called from
    `pre_process`.
    """
    t_sin = _get_timestep_sinusoidal_embedding(t_flat, 256).astype(compute_dtype)
    h = linear_1(t_sin)
    h = nn.silu(h)
    return linear_2(h)


class LTXAttention(nn.Module):
    """`attn1` (self-attn, RoPE) / `attn2` (cross-attn, no RoPE)."""
    inner_dim: int
    num_heads: int
    head_dim: int
    is_cross_attn: bool = False
    eps: float = 1e-6  # q_norm/k_norm eps == the checkpoint's norm_eps.
    apply_gated_attention: bool = False
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

        v = nn.Dense(self.inner_dim, name="to_v")(kv_input)

        seq_k = k.shape[1]
        q = q.reshape(b, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(b, seq_k, self.num_heads, self.head_dim)
        v_heads = v.reshape(b, seq_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if not self.is_cross_attn:
            cos, sin = freqs
            q = apply_rope(q.transpose(0, 2, 1, 3), cos, sin).transpose(0, 2, 1, 3)
            k = apply_rope(k.transpose(0, 2, 1, 3), cos, sin).transpose(0, 2, 1, 3)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        # Promote (never downcast) for the softmax -- a hardcoded
        # `.astype(jnp.float32)` here silently truncated a verification
        # run's float64 activations, and with `vidax.models.ltx2_5.
        # connector`'s pre-final-norm residual stream growing into the
        # thousands over 8 layers, that truncation overflowed to `inf`/
        # `NaN` -- a real bug, not just lost precision, caught only by a
        # float64 bit-exact check.
        softmax_dtype = jnp.promote_types(q.dtype, jnp.float32)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(softmax_dtype) * scale
        if encoder_attention_bias is not None:
            logits = logits + encoder_attention_bias
        weights = jax.nn.softmax(logits, axis=-1).astype(v_heads.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v_heads)
        out = out.transpose(0, 2, 1, 3).reshape(b, seq_q, self.inner_dim)

        if self.apply_gated_attention:
            gate_logits = nn.Dense(self.num_heads, name="to_gate_logits")(hidden_states)
            gates = 2.0 * jax.nn.sigmoid(gate_logits.astype(jnp.float32)).astype(out.dtype)
            out = out.reshape(b, seq_q, self.num_heads, self.head_dim) * gates[..., None]
            out = out.reshape(b, seq_q, self.inner_dim)

        out = nn.Dense(self.inner_dim, name="to_out_0")(out)
        return out


class LTXFeedForward(nn.Module):
    """`activation_fn="gelu-approximate"`: single-gate GELU FFN."""
    dim: int
    inner_dim: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.inner_dim, use_bias=self.use_bias, name="ff_proj")(x)
        h = nn.gelu(h, approximate=True)
        return nn.Dense(self.dim, use_bias=self.use_bias, name="ff_out")(h)


class LTXDiTBlock(nn.Module):
    """`BasicAVTransformerBlock` (video-only): self-attn (RoPE) -> weightless
    RMSNorm -> cross-attn -> FFN, each AdaLN-modulated via this block's own
    `scale_shift_table` (`(9, dim)` when `cross_attention_adaln`, else
    `(6, dim)`). See module docstring for the cross-attention-AdaLN and
    gated-attention deltas vs. LTX-Video.
    """
    dim: int
    num_heads: int
    head_dim: int
    ff_inner_dim: int
    cross_attention_dim: int
    eps: float = 1e-6
    ff_bias: bool = True
    cross_attention_adaln: bool = False
    apply_gated_attention: bool = False
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_bias: Optional[jnp.ndarray],
        timestep: jnp.ndarray,
        prompt_timestep: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        b = x.shape[0]
        num_ada_params = 9 if self.cross_attention_adaln else 6
        scale_shift_table = self.param(
            "scale_shift_table", nn.initializers.normal(stddev=self.dim ** -0.5), (num_ada_params, self.dim))
        ada = scale_shift_table[None, None].astype(jnp.float32) + timestep.astype(jnp.float32).reshape(
            b, timestep.shape[1], num_ada_params, self.dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            ada[:, :, i] for i in range(6)]

        norm_x = _rms_norm_no_affine(x, self.eps)
        norm_x = (norm_x.astype(jnp.float32) * (1 + scale_msa) + shift_msa).astype(self.compute_dtype)
        attn_out = LTXAttention(
            self.dim, self.num_heads, self.head_dim, is_cross_attn=False,
            eps=self.eps, apply_gated_attention=self.apply_gated_attention,
            compute_dtype=self.compute_dtype, name="attn1")(norm_x, freqs=freqs)
        x = x + (gate_msa * attn_out.astype(jnp.float32)).astype(x.dtype)

        x_normed = _rms_norm_no_affine(x, self.eps)
        if self.cross_attention_adaln:
            shift_q, scale_q, gate_ca = ada[:, :, 6], ada[:, :, 7], ada[:, :, 8]
            prompt_scale_shift_table = self.param(
                "prompt_scale_shift_table", nn.initializers.normal(stddev=self.dim ** -0.5), (2, self.dim))
            kv_mod = prompt_scale_shift_table[None, None].astype(jnp.float32)
            if prompt_timestep is not None:
                kv_mod = kv_mod + prompt_timestep.astype(jnp.float32).reshape(b, prompt_timestep.shape[1], 2, self.dim)
            shift_kv, scale_kv = kv_mod[:, :, 0], kv_mod[:, :, 1]
            attn_input = (x_normed.astype(jnp.float32) * (1 + scale_q) + shift_q).astype(self.compute_dtype)
            ctx_mod = (encoder_hidden_states.astype(jnp.float32) * (1 + scale_kv) + shift_kv).astype(self.compute_dtype)
            attn2_out = LTXAttention(
                self.dim, self.num_heads, self.head_dim, is_cross_attn=True,
                eps=self.eps, apply_gated_attention=self.apply_gated_attention,
                compute_dtype=self.compute_dtype, name="attn2")(
                    attn_input, encoder_hidden_states=ctx_mod,
                    encoder_attention_bias=encoder_attention_bias)
            x = x + (gate_ca * attn2_out.astype(jnp.float32)).astype(x.dtype)
        else:
            attn2_out = LTXAttention(
                self.dim, self.num_heads, self.head_dim, is_cross_attn=True,
                eps=self.eps, apply_gated_attention=self.apply_gated_attention,
                compute_dtype=self.compute_dtype, name="attn2")(
                    x_normed, encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_bias=encoder_attention_bias)
            x = x + attn2_out

        norm_h = _rms_norm_no_affine(x, self.eps)
        norm_h = (norm_h.astype(jnp.float32) * (1 + scale_mlp) + shift_mlp).astype(self.compute_dtype)
        ff_out = LTXFeedForward(self.dim, self.ff_inner_dim, use_bias=self.ff_bias, name="ff")(norm_h)
        x = x + (gate_mlp * ff_out.astype(jnp.float32)).astype(x.dtype)
        return x


class LTXDiT(nn.Module):
    """`LTXModel(model_type=LTXModelType.VideoOnly)`. Config-driven -- pass
    `vidax.models.ltx2_5.configs.DIT_22B_CONFIG` (or the dict read directly
    from a checkpoint's own embedded metadata via `load_ltx2_5_metadata`) as
    constructor kwargs.

    `setup()`-based, mirroring `vidax.models.ltx_video.dit.LTXDiT`.
    """
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    timestep_scale_multiplier: int = 1000
    ff_bias: bool = True
    cross_attention_adaln: bool = False
    apply_gated_attention: bool = False
    use_keyframes_abs_pos_embedding: bool = False
    double_precision_rope: bool = False
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.patchify_proj = nn.Dense(self.inner_dim, name="patchify_proj")
        self.adaln_single_emb_timestep_embedder_linear_1 = nn.Dense(
            self.inner_dim, name="adaln_single_emb_timestep_embedder_linear_1")
        self.adaln_single_emb_timestep_embedder_linear_2 = nn.Dense(
            self.inner_dim, name="adaln_single_emb_timestep_embedder_linear_2")
        adaln_coeff = 9 if self.cross_attention_adaln else 6
        self.adaln_linear = nn.Dense(adaln_coeff * self.inner_dim, name="adaln_linear")
        if self.cross_attention_adaln:
            self.prompt_adaln_single_emb_timestep_embedder_linear_1 = nn.Dense(
                self.inner_dim, name="prompt_adaln_single_emb_timestep_embedder_linear_1")
            self.prompt_adaln_single_emb_timestep_embedder_linear_2 = nn.Dense(
                self.inner_dim, name="prompt_adaln_single_emb_timestep_embedder_linear_2")
            self.prompt_adaln_single_linear = nn.Dense(2 * self.inner_dim, name="prompt_adaln_single_linear")
        self.proj_out = nn.Dense(self.out_channels, name="proj_out")
        self.norm_out = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=self.eps, name="norm_out")
        self.scale_shift_table = self.param(
            "scale_shift_table", nn.initializers.normal(stddev=self.inner_dim ** -0.5), (2, self.inner_dim))
        if self.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = self.param(
                "keyframes_abs_pos_embedding", nn.initializers.zeros, (1, self.inner_dim))
        ff_inner_dim = self.inner_dim * 4
        self.blocks = [
            LTXDiTBlock(
                dim=self.inner_dim, num_heads=self.num_attention_heads,
                head_dim=self.attention_head_dim, ff_inner_dim=ff_inner_dim,
                cross_attention_dim=self.cross_attention_dim, eps=self.eps,
                ff_bias=self.ff_bias, cross_attention_adaln=self.cross_attention_adaln,
                apply_gated_attention=self.apply_gated_attention,
                compute_dtype=self.compute_dtype, name=f"blocks_{i}")
            for i in range(self.num_layers)
        ]

    def pre_process(
        self,
        latents: jnp.ndarray,
        latent_coords: jnp.ndarray,
        timestep: jnp.ndarray,
        sigma: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        keyframes_mask: Optional[jnp.ndarray] = None,
    ):
        """`latents` is `(B, N, in_channels)`; `latent_coords` is
        `(B, 3, N, 2)` fractional pixel-space `[start, end)` patch bounds
        per token -- RoPE is evaluated at the midpoint of each. `timestep`
        is `(B,)` or `(B, N)`; `sigma` is `(B,)`, the scalar current
        denoising step used by cross-attention AdaLN's key/value modulation
        (independent of any per-token `timestep` masking).
        """
        input_dtype = latents.dtype
        b = latents.shape[0]
        x = self.patchify_proj(latents)

        if self.use_keyframes_abs_pos_embedding and keyframes_mask is not None:
            x = x + keyframes_mask.astype(x.dtype) * self.keyframes_abs_pos_embedding.astype(x.dtype)

        scaled_timestep = self.timestep_scale_multiplier * timestep
        freqs = create_ltx2_5_rope_freqs(
            latent_coords, self.inner_dim, self.positional_embedding_theta,
            self.positional_embedding_max_pos, self.num_attention_heads,
            dtype=self.compute_dtype, double_precision=self.double_precision_rope)

        t_flat = scaled_timestep.reshape(-1)
        embedded_timestep_flat = _pixart_timestep_embed(
            t_flat, self.adaln_single_emb_timestep_embedder_linear_1,
            self.adaln_single_emb_timestep_embedder_linear_2, self.compute_dtype)
        timestep_mod_flat = self.adaln_linear(nn.silu(embedded_timestep_flat))
        adaln_coeff = 9 if self.cross_attention_adaln else 6
        timestep_mod = timestep_mod_flat.reshape(b, -1, adaln_coeff * self.inner_dim)
        embedded_timestep = embedded_timestep_flat.reshape(b, -1, self.inner_dim)

        prompt_timestep = None
        if self.cross_attention_adaln:
            scaled_sigma = self.timestep_scale_multiplier * sigma
            prompt_embedded = _pixart_timestep_embed(
                scaled_sigma.reshape(-1), self.prompt_adaln_single_emb_timestep_embedder_linear_1,
                self.prompt_adaln_single_emb_timestep_embedder_linear_2, self.compute_dtype)
            prompt_linear = self.prompt_adaln_single_linear(nn.silu(prompt_embedded))
            prompt_timestep = prompt_linear.reshape(b, -1, 2 * self.inner_dim)

        context = encoder_hidden_states.reshape(b, -1, self.cross_attention_dim)

        encoder_attention_bias = None
        if encoder_attention_mask is not None:
            encoder_attention_bias = (
                (1.0 - encoder_attention_mask.astype(jnp.float32)) * jnp.finfo(jnp.float32).min)[:, None, None, :]

        return (x, freqs, context, encoder_attention_bias, timestep_mod, prompt_timestep,
                embedded_timestep, input_dtype)

    def post_process(self, x: jnp.ndarray, embedded_timestep: jnp.ndarray, input_dtype: jnp.dtype) -> jnp.ndarray:
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
        sigma: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        keyframes_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            latents: (B, N, in_channels) patchified video-latent tokens.
            latent_coords: (B, 3, N, 2) fractional pixel-space [start, end)
                patch bounds per token, for RoPE.
            timestep: (B,) or (B, N) diffusion timesteps in [0, 1], scaled
                internally by `timestep_scale_multiplier`.
            sigma: (B,) scalar current denoising step (for cross-attention
                AdaLN's key/value modulation, see `pre_process`).
            encoder_hidden_states: (B, L, cross_attention_dim) connector-
                projected Gemma-4 text embeddings.
            encoder_attention_mask: (B, L) 1 = keep, 0 = discard.
            keyframes_mask: (B, N, 1) non-zero at generated-keyframe token
                positions; all-`None`/zero for plain T2V/I2V.

        Returns:
            (B, N, out_channels) velocity prediction.
        """
        (x, freqs, context, encoder_attention_bias, timestep_mod, prompt_timestep,
         embedded_timestep, input_dtype) = self.pre_process(
            latents, latent_coords, timestep, sigma, encoder_hidden_states,
            encoder_attention_mask, keyframes_mask)
        for block in self.blocks:
            x = block(x, freqs, context, encoder_attention_bias, timestep_mod, prompt_timestep)
        return self.post_process(x, embedded_timestep, input_dtype)
