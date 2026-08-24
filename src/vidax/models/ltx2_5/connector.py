"""LTX-2.5's "embeddings connector": an 8-layer 1D self-attention transformer
that projects Gemma-4's per-token text features into the DiT's
`cross_attention_dim` space, with padded positions substituted by learnable
"register" tokens.

A structural port of `Embeddings1DConnector`/`_BasicTransformerBlock1D` from
`refs/LTX-2-main/packages/ltx-core/src/ltx_core/text_encoders/gemma/
embeddings_connector.py`. Its weights live inside the *DiT* checkpoint
(`model.diffusion_model.video_embeddings_connector.*`, alongside
`audio_embeddings_connector.*` which this video-only port ignores), not the
text-encoder checkpoint -- confirmed from the real
`ltx-2.5-22b-distilled-transformer-bf16.safetensors`' own keys, not assumed.
It is a separate model from `LTXDiT` here (not folded into it) because it
runs once per prompt on Gemma's raw output, upstream of and unchanged across
every denoising step -- the same reason T5 encoding is a separate call in
`vidax.models.ltx_video`'s pipeline.

Reuses `vidax.models.ltx2_5.dit.LTXAttention`/`LTXFeedForward` directly:
the connector's own `_BasicTransformerBlock1D` is architecturally the same
gated, RoPE'd, `q_norm`/`k_norm`'d self-attention + `gelu-approximate` FFN
as a `LTXDiTBlock`'s self-attention half, just with **no AdaLN** (plain
weightless-RMSNorm pre-norm instead of timestep-modulated scale/shift/gate)
and self-attention only (no cross-attention, no FFN gate).

RoPE here is 1D (`n_pos_dims=1`, plain sequence index `0..S-1`, not
`[start, end)` patch bounds) -- `vidax.models.ltx2_5.rope.
create_ltx2_5_rope_freqs` is reused by passing `start == end == index`
(midpoint of a degenerate `[i, i]` range is exactly `i`), matching the
reference's `use_middle_indices_grid=False` (single-point) call here
exactly without a second RoPE implementation.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from vidax.models.ltx2_5.dit import LTXAttention, LTXFeedForward, _rms_norm_no_affine
from vidax.models.ltx2_5.rope import create_ltx2_5_rope_freqs


class _ConnectorBlock(nn.Module):
    dim: int
    num_heads: int
    head_dim: int
    apply_gated_attention: bool
    ff_bias: bool
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, freqs: Tuple[jnp.ndarray, jnp.ndarray],
        attention_bias: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        norm_x = _rms_norm_no_affine(x, self.eps)
        attn_out = LTXAttention(
            self.dim, self.num_heads, self.head_dim, is_cross_attn=False,
            eps=self.eps, apply_gated_attention=self.apply_gated_attention,
            compute_dtype=self.compute_dtype, name="attn1")(
                norm_x, freqs=freqs, encoder_attention_bias=attention_bias)
        x = x + attn_out

        norm_x = _rms_norm_no_affine(x, self.eps)
        ff_out = LTXFeedForward(self.dim, self.dim * 4, use_bias=self.ff_bias, name="ff")(norm_x)
        x = x + ff_out
        return x


class Embeddings1DConnector(nn.Module):
    """Config-driven -- pass `connector_num_attention_heads`/
    `connector_attention_head_dim`/`connector_num_layers`/
    `connector_positional_embedding_max_pos`/`connector_num_learnable_
    registers`/`connector_apply_gated_attention`/`connector_ff_bias` read
    from the DiT checkpoint's own embedded metadata (see
    `vidax.models.ltx2_5.configs`).
    """
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 8
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: Tuple[int] = (4096,)
    num_learnable_registers: int = 128
    apply_gated_attention: bool = False
    ff_bias: bool = True
    double_precision_rope: bool = False
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, hidden_states: jnp.ndarray, additive_attention_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            hidden_states: (B, S, inner_dim) already-projected-to-inner_dim
                per-token text features (the Gemma feature extractor's
                `video_aggregate_embed` output -- see
                `vidax.models.ltx2_5.gemma4`).
            additive_attention_mask: (B, 1, 1, S), `0.0` valid / large-
                negative padding -- `None` if every position is valid.

        Returns:
            (encoded, additive_attention_mask): `additive_attention_mask`
                is all-zero (no masking) whenever registers replaced every
                padded position, matching the reference.
        """
        inner_dim = self.num_attention_heads * self.attention_head_dim
        b, s, _ = hidden_states.shape

        if self.num_learnable_registers:
            # Reference init is `torch.rand(...)*2-1` (uniform on [-1, 1)) --
            # doesn't matter in practice since real checkpoints always
            # supply trained values, but the `-1` belongs only to a fresh
            # initializer, never to the loaded parameter itself (an earlier
            # version of this file applied it unconditionally in the
            # forward pass, silently shifting every *real* register value
            # down by 1 -- caught only by a bit-exact check with padding
            # exercised, since the no-padding path never touches this
            # parameter at all).
            registers = self.param(
                "learnable_registers", lambda key, shape: nn.initializers.uniform(scale=2.0)(key, shape) - 1.0,
                (self.num_learnable_registers, inner_dim))
            registers = registers.astype(hidden_states.dtype)
            reps = s // self.num_learnable_registers
            registers = jnp.tile(registers, (reps, 1))
            registers = jnp.broadcast_to(registers[None], (b, s, inner_dim))
            if additive_attention_mask is not None:
                binary_mask = (additive_attention_mask[:, 0, 0, :] >= 0).astype(hidden_states.dtype)[..., None]
                hidden_states = binary_mask * hidden_states + (1 - binary_mask) * registers
            additive_attention_mask = jnp.zeros_like(additive_attention_mask) if additive_attention_mask is not None else None

        index = jnp.arange(s, dtype=jnp.float32)
        index = jnp.broadcast_to(index[None, None, :], (b, 1, s))
        positions = jnp.stack([index, index], axis=-1)  # degenerate [i, i] bounds -> midpoint == i.
        freqs = create_ltx2_5_rope_freqs(
            positions, inner_dim, self.positional_embedding_theta, self.positional_embedding_max_pos,
            self.num_attention_heads, dtype=self.compute_dtype, double_precision=self.double_precision_rope)

        x = hidden_states
        for i in range(self.num_layers):
            x = _ConnectorBlock(
                inner_dim, self.num_attention_heads, self.attention_head_dim,
                apply_gated_attention=self.apply_gated_attention, ff_bias=self.ff_bias,
                eps=self.eps, compute_dtype=self.compute_dtype, name=f"transformer_1d_blocks_{i}")(
                    x, freqs, additive_attention_mask)

        x = _rms_norm_no_affine(x, self.eps)
        return x, additive_attention_mask
