"""UMT5-XXL text encoder (Flax/JAX), encoder-only.

A structural port of the reference PyTorch ``T5Encoder``/``umt5_xxl`` from
Wan2.1-main/wan/modules/t5.py — this is the text encoder Wan2.1 conditions
its DiT on. Only the encoder is implemented (Wan2.1 never runs the T5
decoder). Unlike vanilla T5, UMT5 does *not* share one relative-position
bias across layers (``shared_pos=False``): every block owns its own
relative-position embedding table.
"""
import html
import re
from typing import List, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from vidax.core.attention import RMSNorm, dot_product_attention


def _relative_position_bucket(
    relative_position: jnp.ndarray, num_buckets: int, max_distance: int = 128
) -> jnp.ndarray:
    """Bidirectional T5 relative-position bucketing (encoder self-attention)."""
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


def _relative_position_bias(seq_len: int, num_buckets: int, num_heads: int, name: str):
    """Per-layer learned relative-position attention bias, shape (1, num_heads, L, L)."""
    context_position = jnp.arange(seq_len)[:, None]
    memory_position = jnp.arange(seq_len)[None, :]
    relative_position = memory_position - context_position
    bucket = _relative_position_bucket(relative_position, num_buckets)
    values = nn.Embed(num_buckets, num_heads, name=name)(bucket)  # (L, L, num_heads)
    return jnp.transpose(values, (2, 0, 1))[None]  # (1, num_heads, L, L)


def _t5_attention(
    x_q: jnp.ndarray, x_kv: jnp.ndarray, dim: int, dim_attn: int, num_heads: int,
    prefix: str, bias: jnp.ndarray,
) -> jnp.ndarray:
    """T5Attention: plain (no QK-norm, no RoPE, no softmax scaling) bias-additive attention."""
    head_dim = dim_attn // num_heads
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
    """One UMT5 encoder self-attention block, with its own relative-position bias."""
    dim: int
    dim_attn: int
    dim_ffn: int
    num_heads: int
    num_buckets: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        seq_len = x.shape[1]
        pos_bias = _relative_position_bias(
            seq_len, self.num_buckets, self.num_heads, name="pos_embedding")
        bias = pos_bias if attn_mask is None else pos_bias + attn_mask

        h = RMSNorm(self.dim, eps=self.eps, name="norm1")(x)
        x = x + _t5_attention(h, h, self.dim, self.dim_attn, self.num_heads, "attn", bias)

        h = RMSNorm(self.dim, eps=self.eps, name="norm2")(x)
        gate = nn.gelu(nn.Dense(self.dim_ffn, use_bias=False, name="ffn_gate_0")(h),
                        approximate=True)
        h = nn.Dense(self.dim_ffn, use_bias=False, name="ffn_fc1")(h) * gate
        h = nn.Dense(self.dim, use_bias=False, name="ffn_fc2")(h)
        x = x + h
        return x


class T5Encoder(nn.Module):
    """UMT5-XXL text encoder. Defaults match Wan2.1's ``umt5_xxl`` config."""
    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
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

        attn_mask = None
        if mask is not None:
            attn_mask = jnp.where(
                mask[:, None, None, :] > 0, 0.0, jnp.finfo(jnp.float32).min)

        for i in range(self.num_layers):
            x = T5Block(
                self.dim, self.dim_attn, self.dim_ffn, self.num_heads,
                self.num_buckets, self.eps, name=f"blocks_{i}")(x, attn_mask)

        x = RMSNorm(self.dim, eps=self.eps, name="norm")(x)
        return x


def _whitespace_clean(text: str) -> str:
    """Matches Wan2.1's `clean='whitespace'` tokenizer preprocessing, minus
    ftfy's mojibake repair (not worth an extra dependency for well-formed
    UTF-8 prompts).
    """
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


class Umt5Tokenizer:
    """Thin wrapper around the UMT5 HuggingFace tokenizer (`google/umt5-xxl`),
    producing fixed-length, zero-padded (ids, attention_mask) arrays ready
    for `T5Encoder`.
    """

    def __init__(self, tokenizer_path: str, seq_len: int = 512):
        from transformers import AutoTokenizer  # Optional dependency; install the `text` extra.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.seq_len = seq_len

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [_whitespace_clean(t) for t in texts]
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True,
            max_length=self.seq_len)
        return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)
