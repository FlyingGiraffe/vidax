"""Reason1 text encoder (Flax/JAX), text-only decoder-tower.

Cosmos-Predict2.5-2B conditions its DiT on "Reason1": a Reason1-SFT-finetuned
Qwen2.5-VL-7B-Instruct, wrapped by `cosmos_predict2/_src/predict2/text_encoders/
reason1.py`'s `QwenVLBaseModel` and orchestrated by that directory's
`text_encoder.py`'s `TextEncoder`. For text-to-video (no `pixel_values`/
`pixel_values_videos`), the vision encoder/patch-merger is never invoked
(`QwenVLBaseModel._forward` only calls `self.visual(...)` when pixel inputs
are not None) -- so only the text decoder-language-model tower is ported
here: embed_tokens, 28 standard Qwen2 decoder blocks, final RMSNorm. No
`lm_head` (only hidden_states are needed, never logits/generation).

M-RoPE-for-text-only risk (resolved): `Qwen2_5_VLForConditionalGeneration`
normally computes 3D (temporal/height/width) M-RoPE position ids via
`get_rope_index` (qwen2_5_vl.py:2041). But `QwenVLBaseModel.forward` never
passes `attention_mask` (`text_encoder.py`'s `compute_text_embeddings_online`
calls `self.model(input_ids_batch, {})`, an empty `data_batch`), and with no
image/video grid *and* no attention_mask, `get_rope_index`
(qwen2_5_vl.py:2218-2229) falls straight to the plain branch:
`position_ids = arange(seq_len).view(1,1,-1).expand(3, B, -1)` -- i.e. the
t/h/w position ids are identical to each other and to ordinary sequential
positions. `apply_multimodal_rotary_pos_emb` (qwen2_5_vl.py:662-698) then
splits cos/sin by `mrope_section` ([16, 24, 24], summing to head_dim//2=64)
and picks `cos[i % 3]`/`sin[i % 3]` per section -- but since all three
position-id rows are identical, `cos[0] == cos[1] == cos[2]` for every
token, so the section splitting is a no-op and the result is bit-identical
to plain single-axis RoPE. This module therefore implements standard 1D
rotary embeddings (HF Llama/Qwen2 "rotate_half" convention, NOT the
adjacent-pair convention `vidax.core.rope3d` uses for Wan's video RoPE) and
that is exact for this text-only call path, not an approximation.

Embedding-extraction pipeline (`TextEncoder.compute_text_embeddings_online`,
text_encoder.py:131-220, called with `embedding_concat_strategy="full_concat"`
per `get_reason1_embeddings`, text_encoder.py:223-238):
  1. Wrap the caption in a fixed two-turn chat template (system: "You are a
     helpful assistant who will provide prompts to an image generator.",
     user: the caption), tokenize, pad/truncate to exactly
     `NUM_EMBEDDING_PADDING_TOKENS=512` tokens with the tokenizer's pad id.
  2. Forward pass, text-only, `output_hidden_states=True` -> a tuple of 29
     hidden-state tensors: index 0 is the embedding-layer output, indices
     1..27 are each layer's raw (pre-final-norm) output, index 28 is
     `norm(layer_27_output)` (standard HF `output_hidden_states` semantics --
     the final entry is post-final-norm, not a 29th decoder layer).
  3. Skip index 0. Per-token mean/std-normalize each of the remaining 28
     layers: `(h - mean(h, -1)) / (std(h, -1) + 1e-8)`.
  4. Concatenate all 28 normalized (B, 512, 3584) tensors along the feature
     axis -> (B, 512, 28*3584=100352). This is what feeds the DiT's
     `crossattn_proj`.
"""
from typing import List, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from vidax.core.attention import RMSNorm, dot_product_attention

NUM_EMBEDDING_PADDING_TOKENS = 512


def _qwen_rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Standard HF rotary tables (Llama/Qwen2 `Qwen2_5_VLRotaryEmbedding` with
    `rope_type="default"`, qwen2_5_vl.py:574-641): `inv_freq` over even
    channel indices, duplicated across the two halves of `head_dim` (the
    "rotate_half" convention, not the interleaved-pair convention
    `vidax.core.rope3d` uses).

    Returns:
        (cos, sin), each shape (seq_len, head_dim).
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)  # (S, head_dim // 2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (S, head_dim)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """x: (B, S, H, head_dim); cos/sin: (S, head_dim), float32.

    Casts back to `x`'s original dtype before returning (matching
    `vidax.core.rope3d.apply_rope3d`'s convention for Wan's RoPE): `cos`/
    `sin` are float32, so left uncast this silently upcasts q/k to float32
    while v stays at the model's compute dtype (bf16 in practice) -- fine
    numerically, but `jax.nn.dot_product_attention` hard-requires q/k/v to
    share one dtype and raises rather than promoting for you.
    """
    orig_dtype = x.dtype
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = x.astype(jnp.float32) * cos + _rotate_half(x.astype(jnp.float32)) * sin
    return out.astype(orig_dtype)


class Qwen2Attention(nn.Module):
    """GQA self-attention: `num_attention_heads` query heads share
    `num_key_value_heads` key/value heads (repeated `num_heads // num_kv_heads`
    times, HF's `repeat_kv`). q/k/v_proj carry bias, o_proj does not
    (qwen2_5_vl.py:748-751, Qwen2-family convention)."""
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    @nn.compact
    def __call__(self, x, cos, sin, causal_mask):
        b, s, _ = x.shape
        head_dim = self.hidden_size // self.num_heads

        q = nn.Dense(self.num_heads * head_dim, use_bias=True, name="q_proj")(x)
        k = nn.Dense(self.num_kv_heads * head_dim, use_bias=True, name="k_proj")(x)
        v = nn.Dense(self.num_kv_heads * head_dim, use_bias=True, name="v_proj")(x)

        q = q.reshape(b, s, self.num_heads, head_dim)
        k = k.reshape(b, s, self.num_kv_heads, head_dim)
        v = v.reshape(b, s, self.num_kv_heads, head_dim)

        q = _apply_rotary_pos_emb(q, cos, sin)
        k = _apply_rotary_pos_emb(k, cos, sin)

        rep = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)

        out = dot_product_attention(q, k, v, mask=causal_mask)
        out = out.reshape(b, s, self.num_heads * head_dim)
        return nn.Dense(self.hidden_size, use_bias=False, name="o_proj")(out)


class Qwen2MLP(nn.Module):
    """SwiGLU, no bias (qwen2_5_vl.py:649-658)."""
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)
        h = nn.silu(gate) * up
        return nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(h)


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm decoder block (qwen2_5_vl.py:1058-1139): RMSNorm -> self-attn
    -> residual, RMSNorm -> SwiGLU MLP -> residual."""
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, cos, sin, causal_mask):
        h = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="input_layernorm")(x)
        x = x + Qwen2Attention(
            self.hidden_size, self.num_heads, self.num_kv_heads, name="self_attn",
        )(h, cos, sin, causal_mask)

        h = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="post_attention_layernorm")(x)
        x = x + Qwen2MLP(self.hidden_size, self.intermediate_size, name="mlp")(h)
        return x


class Qwen2TextModel(nn.Module):
    """Text-only Qwen2.5 decoder tower. Defaults match the Reason1-finetuned
    Qwen2.5-VL-7B-Instruct text config used by Cosmos-Predict2.5-2B
    (text_encoder.py:48-62): hidden_size=3584, 28 layers, 28 query / 4 KV
    heads (head_dim=128), intermediate_size=18944, vocab_size=152064,
    rope_theta=1e6 (model_config_qwen.py:164-168). No `lm_head` -- only
    hidden_states are needed.
    """
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """
        Args:
            input_ids: (B, S) int32 token ids.

        Returns:
            Tuple of `num_hidden_layers + 1` hidden-state tensors, each
            (B, S, hidden_size), matching HF `output_hidden_states=True`:
            index 0 is the embedding layer, indices 1..N-1 are each decoder
            layer's raw output, and the last index is `norm(final layer
            output)` -- not a separate (N+1)-th layer.
        """
        b, s = input_ids.shape
        head_dim = self.hidden_size // self.num_attention_heads

        x = nn.Embed(self.vocab_size, self.hidden_size, name="embed_tokens")(input_ids)
        cos, sin = _qwen_rope_cos_sin(s, head_dim, self.rope_theta)
        # Standard autoregressive causal mask (decoder-only LM); True = attend.
        causal_mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, None]

        hidden_states: List[jnp.ndarray] = [x]
        for i in range(self.num_hidden_layers):
            x = Qwen2DecoderLayer(
                self.hidden_size, self.intermediate_size,
                self.num_attention_heads, self.num_key_value_heads,
                self.rms_norm_eps, name=f"layers_{i}",
            )(x, cos, sin, causal_mask)
            hidden_states.append(x)

        # HF appends the final hidden state *after* `self.norm`, not the raw
        # last-layer output (see module docstring).
        hidden_states[-1] = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="norm")(hidden_states[-1])

        return tuple(hidden_states)


def mean_normalize(x: jnp.ndarray) -> jnp.ndarray:
    """Per-token mean/std normalization (`TextEncoder.mean_normalize`,
    text_encoder.py:118-129)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def compute_reason1_embeddings(
    params, input_ids: jnp.ndarray, model: Optional[Qwen2TextModel] = None,
) -> jnp.ndarray:
    """Runs the Reason1 text tower and reproduces
    `TextEncoder.compute_text_embeddings_online`'s `FULL_CONCAT` pipeline
    (text_encoder.py:194-220, `embedding_concat_strategy="full_concat"`).

    Args:
        params: Flax param tree for `Qwen2TextModel` (e.g. from
            `map_reason1_text_encoder_keys`).
        input_ids: (B, 512) int32 token ids, already padded/truncated to
            `NUM_EMBEDDING_PADDING_TOKENS` by the caller (see
            `Reason1Tokenizer` below).
        model: Optional pre-constructed `Qwen2TextModel` (defaults to the
            2B checkpoint's config).

    Returns:
        (B, 512, num_hidden_layers * hidden_size) = (B, 512, 100352) text
        embedding, ready for the DiT's `crossattn_proj`.
    """
    if model is None:
        model = Qwen2TextModel()
    hidden_states = model.apply({"params": params["params"]}, input_ids)

    # Skip index 0 (the embedding layer); normalize + concat the 28 layers.
    normalized = [mean_normalize(h) for h in hidden_states[1:]]
    return jnp.concatenate(normalized, axis=-1)


# --------------------------------------------------------------------------
# Prompt formatting + tokenization glue (analogous to `Umt5Tokenizer` in
# vidax.models.wan.common.t5 -- an optional `transformers` dependency, since
# there is no existing Qwen tokenizer port in this repo).
# --------------------------------------------------------------------------

# Fixed system prompt Cosmos-Predict2.5 wraps every caption with
# (text_encoder.py:144-150).
REASON1_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."


class Reason1Tokenizer:
    """Wraps a caption in Reason1's fixed chat template and tokenizes it to
    exactly `NUM_EMBEDDING_PADDING_TOKENS` ids (pad/truncate), matching
    `TextEncoder.compute_text_embeddings_online` (text_encoder.py:139-183).
    """

    def __init__(self, tokenizer_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 seq_len: int = NUM_EMBEDDING_PADDING_TOKENS):
        from transformers import AutoTokenizer  # Optional dependency; install the `text` extra.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.seq_len = seq_len
        # HF `AutoTokenizer` may leave `pad_token_id` unset for some Qwen
        # configs; the reference falls back to its own `tokenizer.pad_id`,
        # which for Qwen2.5-VL is the eos id (151643, `<|endoftext|>`).
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

    def __call__(self, texts) -> "jnp.ndarray":
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        ids_batch = []
        for text in texts:
            conversation = [
                {"role": "system", "content": REASON1_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            ids = self.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=False)
            # Newer `transformers` returns a `BatchEncoding` (dict-like,
            # `{"input_ids": [...], "attention_mask": [...]}`) here rather
            # than a bare list of ids -- unwrap it if so (only `input_ids`
            # is needed; the attention mask is regenerated as an ordinary
            # pad mask below via `self.pad_id`, not taken from here).
            if hasattr(ids, "get") or isinstance(ids, dict):
                ids = ids["input_ids"]
            if self.seq_len > len(ids):
                ids = ids + [self.pad_id] * (self.seq_len - len(ids))
            else:
                ids = ids[: self.seq_len]
            ids_batch.append(ids)

        return np.asarray(ids_batch, dtype=np.int32)
