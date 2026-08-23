"""LTX-Video's text encoder (Flax/JAX): standard (non-UMT5) T5-XXL encoder,
`PixArt-alpha/PixArt-XL-2-1024-MS`'s `text_encoder` (a plain HuggingFace
`T5EncoderModel`, `feed_forward_proj="gated-gelu"`, `is_gated_act=True`).

Structurally very close to `vidax.models.wan.common.t5.T5Encoder` (same
bidirectional relative-position bucketing formula, same gated-GELU FFN, no
biases anywhere, no QK-norm, no RoPE, no `1/sqrt(head_dim)` attention
scaling -- all genuinely T5-family conventions, not Wan-specific), but with
one real architectural difference: **this checkpoint shares one
relative-position-bias table across every layer** (`shared_pos=True`, T5's
default), unlike UMT5's per-layer tables -- confirmed directly against the
downloaded checkpoint's key list: only `encoder.block.0.layer.0.
SelfAttention.relative_attention_bias.weight` exists; no other block has
one. A self-contained new module (not a subclass/parameterization of
`vidax.models.wan.common.t5.T5Encoder`) to keep this port's files fully
independent of Wan's, per this port's non-regression design -- see
`vidax.models.ltx_video`'s package docstring.
"""
from typing import List, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from vidax.core.attention import RMSNorm, dot_product_attention


def _relative_position_bucket(
    relative_position: jnp.ndarray, num_buckets: int, max_distance: int = 128
) -> jnp.ndarray:
    """Bidirectional T5 relative-position bucketing (encoder self-attention)
    -- identical formula to (and independently duplicated from)
    `vidax.models.wan.common.t5`'s copy; genuinely shared T5-family math,
    not something UMT5 changed.
    """
    num_buckets = num_buckets // 2
    relative_buckets = jnp.where(relative_position > 0, num_buckets, 0)
    relative_position = jnp.abs(relative_position)

    max_exact = num_buckets // 2
    relative_position_large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_large = jnp.minimum(relative_position_large, num_buckets - 1)

    return relative_buckets + jnp.where(
        relative_position < max_exact, relative_position, relative_position_large)


def _t5_attention(
    x_q: jnp.ndarray, x_kv: jnp.ndarray, dim: int, num_heads: int, head_dim: int,
    prefix: str, bias: jnp.ndarray,
) -> jnp.ndarray:
    """`T5Attention` (`SelfAttention`): plain (no QK-norm, no RoPE, no
    softmax scaling) bias-additive attention, no biases on any projection.
    """
    dim_attn = num_heads * head_dim
    b = x_q.shape[0]

    q = nn.Dense(dim_attn, use_bias=False, name=f"{prefix}_q")(x_q)
    k = nn.Dense(dim_attn, use_bias=False, name=f"{prefix}_k")(x_kv)
    v = nn.Dense(dim_attn, use_bias=False, name=f"{prefix}_v")(x_kv)

    q = q.reshape(b, -1, num_heads, head_dim)
    k = k.reshape(b, -1, num_heads, head_dim)
    v = v.reshape(b, -1, num_heads, head_dim)

    # T5 does not scale QK^T by 1/sqrt(head_dim).
    out = dot_product_attention(q, k, v, bias=bias, scale=1.0)
    out = out.reshape(b, -1, num_heads * head_dim)
    return nn.Dense(dim, use_bias=False, name=f"{prefix}_o")(out)


class T5Block(nn.Module):
    """One T5 encoder self-attention block. `pos_bias` (the *shared*
    relative-position bias, computed once by `T5Encoder`) is passed in
    rather than owned per-block -- the one real difference from UMT5's
    `T5Block`.
    """
    dim: int
    num_heads: int
    head_dim: int
    dim_ffn: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, pos_bias: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        bias = pos_bias if attn_mask is None else pos_bias + attn_mask

        h = RMSNorm(self.dim, eps=self.eps, name="norm1")(x)
        x = x + _t5_attention(h, h, self.dim, self.num_heads, self.head_dim, "attn", bias)

        h = RMSNorm(self.dim, eps=self.eps, name="norm2")(x)
        gate = nn.gelu(nn.Dense(self.dim_ffn, use_bias=False, name="ffn_gate_0")(h), approximate=True)
        h = nn.Dense(self.dim_ffn, use_bias=False, name="ffn_fc1")(h) * gate
        h = nn.Dense(self.dim, use_bias=False, name="ffn_fc2")(h)
        return x + h


class T5Encoder(nn.Module):
    """Defaults match `PixArt-alpha/PixArt-XL-2-1024-MS`'s `text_encoder`
    config (`d_model=4096`, `num_heads=64`, `d_kv=64`, `d_ff=10240`,
    `num_layers=24`, `vocab_size=32128`, `relative_attention_num_buckets=32`,
    `relative_attention_max_distance=128`).
    """
    vocab_size: int = 32128
    dim: int = 4096
    num_heads: int = 64
    head_dim: int = 64
    dim_ffn: int = 10240
    num_layers: int = 24
    num_buckets: int = 32
    max_distance: int = 128
    eps: float = 1e-6

    @nn.compact
    def __call__(self, ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            ids: (B, L) int32 token ids.
            mask: Optional (B, L) attention mask (1 for real tokens, 0 for padding).

        Returns:
            (B, L, dim) contextual token embeddings.
        """
        x = nn.Embed(self.vocab_size, self.dim, name="token_embedding")(ids)
        seq_len = x.shape[1]

        # Shared across every block -- computed once here (owned by
        # `blocks_0`'s `relative_attention_bias`, matching the checkpoint's
        # own key: only block 0 has this weight).
        context_position = jnp.arange(seq_len)[:, None]
        memory_position = jnp.arange(seq_len)[None, :]
        bucket = _relative_position_bucket(memory_position - context_position, self.num_buckets, self.max_distance)
        values = nn.Embed(self.num_buckets, self.num_heads, name="relative_attention_bias")(bucket)
        pos_bias = jnp.transpose(values, (2, 0, 1))[None]  # (1, num_heads, L, L)

        attn_mask = None
        if mask is not None:
            attn_mask = jnp.where(mask[:, None, None, :] > 0, 0.0, jnp.finfo(jnp.float32).min)

        for i in range(self.num_layers):
            x = T5Block(
                self.dim, self.num_heads, self.head_dim, self.dim_ffn, self.eps,
                name=f"blocks_{i}")(x, pos_bias, attn_mask)

        x = RMSNorm(self.dim, eps=self.eps, name="norm")(x)
        return x


class PixArtT5Tokenizer:
    """Thin wrapper around the HuggingFace T5 tokenizer shipped alongside
    `PixArt-alpha/PixArt-XL-2-1024-MS`'s `text_encoder` (`tokenizer/`
    subfolder), producing fixed-length, zero-padded (ids, attention_mask)
    arrays ready for `T5Encoder`. Mirrors `vidax.models.wan.common.t5.
    Umt5Tokenizer`'s shape/API (a separate class, not a shared base, per
    this port's file-independence design).
    """

    def __init__(self, tokenizer_path: str, seq_len: int = 256):
        from transformers import AutoTokenizer  # Optional dependency; install the `text` extra.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.seq_len = seq_len

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True,
            max_length=self.seq_len)
        return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)
