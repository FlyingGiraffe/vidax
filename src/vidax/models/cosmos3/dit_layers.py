"""Dual-pathway ("und"/"gen") transformer block for Cosmos3's DiT
(`Cosmos3PackedMoTAttention`, `Cosmos3VLTextMLP`, `Cosmos3VLTextMoTDecoderLayer`
in the reference, refs/diffusers-cosmos3/transformer_cosmos3.py). Shared by
both Cosmos3-Nano and Cosmos3-Edge, which use this exact same weight layout
at different sizes and with different per-checkpoint toggles (see
`vidax.models.cosmos3.configs`): Edge sets `qk_norm_for_text=False`,
`use_und_k_norm_for_gen=True`, and `hidden_act="relu2"`; Nano uses this
module's defaults.

Ported against a fixed-shape `(B, seq_len, hidden)` packing instead of the
reference's ragged/flat-buffer-with-global-indices design: `und_seq` (text,
causal) and `gen_seq` (vision, full-attention) are two separate
`(B, len, hidden)` tensors, not slices of one flat buffer. `und` (text) is
padded to a fixed max length for JAX's static shapes, unlike the reference's
exact-length token lists -- so `gen`'s cross-attention over `und`'s
keys/values needs an explicit padding mask (`und_valid_mask`) that the
reference never needs. `und`'s own causal self-attention does *not* need it:
causal masking already keeps every real token from seeing any (necessarily
later) padding position.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import RMSNorm, dot_product_attention
from vidax.models.cosmos3.mrope import apply_mrope


def _repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """(B, S, num_kv_heads, head_dim) -> (B, S, num_kv_heads * n_rep, head_dim)."""
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return jnp.broadcast_to(x[:, :, :, None, :], (b, s, h, n_rep, d)).reshape(b, s, h * n_rep, d)


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway attention: `und` self-attends causally; `gen` attends
    (non-causal, full) over both `und` and `gen` keys/values -- one-directional
    information flow, `und -> gen` only, `und` never reads `gen`.

    `qk_norm_for_text`: whether `und`'s own q/k get an RMSNorm (True for
    Nano, False for Edge -- Edge's checkpoint has no `norm_q`/`norm_k`
    weights at all).

    `use_und_k_norm_for_gen`: only takes effect when `qk_norm_for_text` is
    False (matches the reference's `use_und_k_norm_for_gen and not
    qk_norm_for_text` gate). When True (Edge), `gen`'s cross-attention reads
    `und`'s keys through a second, separately-normed-and-RoPE'd projection
    (`k_norm_und_for_gen`) instead of the plain `k_und` that `und`'s own
    causal self-attention uses. When False (Nano, where `qk_norm_for_text` is
    already True), both pathways share the same `k_und`.
    """
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    eps: float
    qk_norm_for_text: bool = True
    use_und_k_norm_for_gen: bool = False
    mesh: Optional[Mesh] = None

    @nn.compact
    def __call__(
        self,
        und_seq: jnp.ndarray,
        gen_seq: jnp.ndarray,
        rotary_emb: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        causal_mask: jnp.ndarray,
        und_valid_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b = und_seq.shape[0]
        inner_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        n_rep = self.num_attention_heads // self.num_key_value_heads

        q_und = nn.Dense(inner_dim, use_bias=False, name="to_q")(und_seq).reshape(
            b, -1, self.num_attention_heads, self.head_dim)
        k_und = nn.Dense(kv_dim, use_bias=False, name="to_k")(und_seq).reshape(
            b, -1, self.num_key_value_heads, self.head_dim)
        v_und = nn.Dense(kv_dim, use_bias=False, name="to_v")(und_seq).reshape(
            b, -1, self.num_key_value_heads, self.head_dim)
        q_gen = nn.Dense(inner_dim, use_bias=False, name="add_q_proj")(gen_seq).reshape(
            b, -1, self.num_attention_heads, self.head_dim)
        k_gen = nn.Dense(kv_dim, use_bias=False, name="add_k_proj")(gen_seq).reshape(
            b, -1, self.num_key_value_heads, self.head_dim)
        v_gen = nn.Dense(kv_dim, use_bias=False, name="add_v_proj")(gen_seq).reshape(
            b, -1, self.num_key_value_heads, self.head_dim)

        if self.qk_norm_for_text:
            q_und = RMSNorm(self.head_dim, eps=self.eps, name="norm_q")(q_und)
            k_und = RMSNorm(self.head_dim, eps=self.eps, name="norm_k")(k_und)
        use_k_norm_und_for_gen = self.use_und_k_norm_for_gen and not self.qk_norm_for_text
        k_und_for_gen = (
            RMSNorm(self.head_dim, eps=self.eps, name="k_norm_und_for_gen")(k_und)
            if use_k_norm_und_for_gen else k_und)
        q_gen = RMSNorm(self.head_dim, eps=self.eps, name="norm_added_q")(q_gen)
        k_gen = RMSNorm(self.head_dim, eps=self.eps, name="norm_added_k")(k_gen)

        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        q_und = apply_mrope(q_und, cos_und, sin_und)
        k_und = apply_mrope(k_und, cos_und, sin_und)
        k_und_for_gen = (
            apply_mrope(k_und_for_gen, cos_und, sin_und) if use_k_norm_und_for_gen else k_und)
        q_gen = apply_mrope(q_gen, cos_gen, sin_gen)
        k_gen = apply_mrope(k_gen, cos_gen, sin_gen)

        # --- causal pathway ("und" self-attention) ---
        # `und` is small (a padded text prompt, at most a few hundred tokens),
        # so the boolean-mask-forced XLA fallback path costs nothing that matters.
        causal_out = dot_product_attention(
            q_und, _repeat_kv(k_und, n_rep), _repeat_kv(v_und, n_rep), mask=causal_mask,
            mesh=self.mesh)
        causal_out = causal_out.reshape(b, -1, inner_dim)

        # --- full pathway ("gen" cross-attends to und + gen) ---
        # `gen` is large (tens of thousands of video patch tokens at real
        # resolutions) -- an additive `bias` (not a boolean `mask`) is used here
        # specifically so the flash-attention path stays available (see
        # `dot_product_attention`'s docstring).
        all_k = jnp.concatenate([k_und_for_gen, k_gen], axis=1)
        all_v = jnp.concatenate([v_und, v_gen], axis=1)
        full_bias = None
        if und_valid_mask is not None:
            gen_len = gen_seq.shape[1]
            all_valid = jnp.concatenate(
                [und_valid_mask, jnp.ones((b, gen_len), dtype=jnp.bool_)], axis=1)
            full_bias = jnp.where(all_valid, 0.0, -jnp.inf)[:, None, None, :]
        full_out = dot_product_attention(
            q_gen, _repeat_kv(all_k, n_rep), _repeat_kv(all_v, n_rep), bias=full_bias,
            mesh=self.mesh)
        full_out = full_out.reshape(b, -1, inner_dim)

        und_out = nn.Dense(self.hidden_size, use_bias=False, name="to_out")(causal_out)
        gen_out = nn.Dense(self.hidden_size, use_bias=False, name="to_add_out")(full_out)
        return und_out, gen_out


class Cosmos3VLTextMLP(nn.Module):
    """No bias. `hidden_act="silu"` (Nano): SwiGLU (`gate_proj`+`up_proj`).
    `hidden_act="relu2"` (Edge): squared ReLU, no `gate_proj` -- matches
    Edge's checkpoint, which has no `mlp.gate_proj` weight.
    """
    hidden_size: int
    intermediate_size: int
    hidden_act: str = "silu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        up = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)
        if self.hidden_act == "relu2":
            h = jnp.square(jax.nn.relu(up))
        else:
            gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
            h = nn.silu(gate) * up
        return nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(h)


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """Pre-norm dual-pathway decoder layer: separate `und`/`gen` norms,
    weights, and residual streams throughout -- the only interaction between
    the two pathways is `gen`'s cross-attention reading `und`'s keys/values.
    """
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    eps: float
    hidden_act: str = "silu"
    qk_norm_for_text: bool = True
    use_und_k_norm_for_gen: bool = False
    mesh: Optional[Mesh] = None

    @nn.compact
    def __call__(
        self,
        und_seq: jnp.ndarray,
        gen_seq: jnp.ndarray,
        rotary_emb: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        causal_mask: jnp.ndarray,
        und_valid_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        und_norm = RMSNorm(self.hidden_size, eps=self.eps, name="input_layernorm")(und_seq)
        gen_norm = RMSNorm(self.hidden_size, eps=self.eps, name="input_layernorm_moe_gen")(gen_seq)

        und_attn_out, gen_attn_out = Cosmos3PackedMoTAttention(
            hidden_size=self.hidden_size, head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads, eps=self.eps,
            qk_norm_for_text=self.qk_norm_for_text,
            use_und_k_norm_for_gen=self.use_und_k_norm_for_gen, mesh=self.mesh,
            name="self_attn",
        )(und_norm, gen_norm, rotary_emb, causal_mask, und_valid_mask)
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = Cosmos3VLTextMLP(
            self.hidden_size, self.intermediate_size, self.hidden_act, name="mlp",
        )(RMSNorm(self.hidden_size, eps=self.eps, name="post_attention_layernorm")(residual_und))
        mlp_out_gen = Cosmos3VLTextMLP(
            self.hidden_size, self.intermediate_size, self.hidden_act, name="mlp_moe_gen",
        )(RMSNorm(self.hidden_size, eps=self.eps, name="post_attention_layernorm_moe_gen")(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen
