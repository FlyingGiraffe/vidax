"""HunyuanVideo 1.0's ``"llm"`` text encoder -- a plain Llama3-8B decoder
tower (text-only, no vision), extracted from ``xtuner/llava-llama-3-8b-v1_1-
transformers``'s ``.language_model`` (see
``refs/HunyuanVideo-main/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py``
and ``ckpts/README.md``'s "Download Text Encoder" section -- the reference
explicitly recommends this checkpoint since HunyuanMLLM itself was never
released).

Confirmed directly against the extracted checkpoint's own real ``config.
json`` (not guessed): ``LlamaModel``, ``hidden_size=4096``,
``num_hidden_layers=32``, ``num_attention_heads=32``,
``num_key_value_heads=8`` (head_dim=128), ``intermediate_size=14336``,
``rope_theta=500000.0``, ``rms_norm_eps=1e-5``, ``vocab_size=128320``
(larger than stock Llama-3-8B's 128256 -- xtuner added special image
tokens), ``attention_bias=False``, ``mlp_bias=False`` -- every Dense layer
(``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/``up_proj``/
``down_proj``) is bias-free, unlike Qwen2's ``q``/``k``/``v_proj`` (which
carry bias).

**Why not reuse ``vidax.models.cosmos2_5.reason1.Qwen2TextModel``** (the way
HunyuanVideo-1.5's ``qwen_text.py`` does for its own MLLM): architecturally
very close (RMSNorm, SwiGLU MLP, GQA, rotate-half RoPE -- same family), but
``Qwen2Attention`` hardcodes ``use_bias=True`` on its q/k/v projections
(Qwen2's own convention), which would silently create three extra bias
parameters this Llama checkpoint's state_dict doesn't have. Editing
``reason1.py`` to parameterize that bias is out of scope (a different model
family's file, no genuine bug to justify touching it) -- so this is a fresh,
small port instead, structurally mirroring ``reason1.py``'s pattern.

Only the text-decoder tower is ported (``embed_tokens``, N decoder layers,
final ``norm``) -- no ``lm_head`` (only hidden states are needed).

**Embedding-extraction pipeline** (``TextEncoder.encode``, ``constants.py``'s
``PROMPT_TEMPLATE``/``PROMPT_TEMPLATE_ENCODE_VIDEO``, ``config.py``'s
defaults): wrap the caption in a fixed chat-style template (video default:
``PROMPT_TEMPLATE_ENCODE_VIDEO``, ``crop_start=95``; image:
``PROMPT_TEMPLATE_ENCODE``, ``crop_start=36``), tokenize (right-padded,
``max_length=256`` by default, ``config.py``'s ``--text-len``), forward with
``output_hidden_states=True``, take ``hidden_states[-(hidden_state_skip_layer
+ 1)] == hidden_states[-3]`` (``hidden_state_skip_layer=2`` default -- 2
layers before the final, **not** post-``norm``, since ``apply_final_norm``
defaults to ``False``, confirmed by reading ``config.py``'s
``--apply-final-norm`` ``store_true`` default directly), then crop
``[:, crop_start:]`` to drop the template's own instruction tokens.

Right-padding + causal self-attention means the model's own causal mask
alone (no separate padding-key mask) already gives every *valid* prefix
token -- crop_start onward, up to the real prompt's own length -- the exact
same output an additionally padding-masked forward would (no valid token
ever attends forward to trailing padding), matching
``vidax.models.cosmos2_5.reason1.Qwen2TextModel``'s identical simplification
for the same right-padding-plus-causal-mask setup.
"""
from typing import List, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from vidax.core.attention import RMSNorm, dot_product_attention


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_cos_sin(seq_len: int, head_dim: int, theta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb)[None, None], jnp.sin(emb)[None, None]  # (1, 1, S, head_dim)


def _apply_rotary_pos_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    # x: (B, S, H, D) -> cos/sin: (1, 1, S, D), transpose x to (B, H, S, D) for the multiply.
    orig_dtype = x.dtype
    x_t = jnp.swapaxes(x, 1, 2).astype(jnp.float32)
    out = x_t * cos + _rotate_half(x_t) * sin
    return jnp.swapaxes(out, 1, 2).astype(orig_dtype)


class LlamaAttention(nn.Module):
    """GQA self-attention, all Dense layers bias-free (``attention_bias=
    False``)."""
    hidden_size: int
    num_heads: int
    num_kv_heads: int

    @nn.compact
    def __call__(self, x, cos, sin, causal_mask):
        b, s, _ = x.shape
        head_dim = self.hidden_size // self.num_heads

        q = nn.Dense(self.num_heads * head_dim, use_bias=False, name="q_proj")(x)
        k = nn.Dense(self.num_kv_heads * head_dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(self.num_kv_heads * head_dim, use_bias=False, name="v_proj")(x)

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


class LlamaMLP(nn.Module):
    """SwiGLU, bias-free (``mlp_bias=False``)."""
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)
        h = nn.silu(gate) * up
        return nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(h)


class LlamaDecoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, cos, sin, causal_mask):
        h = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="input_layernorm")(x)
        x = x + LlamaAttention(
            self.hidden_size, self.num_heads, self.num_kv_heads, name="self_attn",
        )(h, cos, sin, causal_mask)

        h = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="post_attention_layernorm")(x)
        x = x + LlamaMLP(self.hidden_size, self.intermediate_size, name="mlp")(h)
        return x


class LlamaTextModel(nn.Module):
    """Text-only Llama3-8B decoder tower. Defaults match the real extracted
    ``xtuner/llava-llama-3-8b-v1_1-transformers`` ``.language_model``
    checkpoint (see module docstring). No ``lm_head``.
    """
    vocab_size: int = 128320
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """
        Args:
            input_ids: (B, S) int32 token ids.

        Returns:
            Tuple of ``num_hidden_layers + 1`` hidden-state tensors, each
            (B, S, hidden_size), matching HF ``output_hidden_states=True``:
            index 0 is the embedding layer, indices 1..N-1 are each decoder
            layer's raw output, and the last index is ``norm(final layer
            output)`` -- not a separate (N+1)-th layer.
        """
        b, s = input_ids.shape
        head_dim = self.hidden_size // self.num_attention_heads

        x = nn.Embed(self.vocab_size, self.hidden_size, name="embed_tokens")(input_ids)
        cos, sin = _rope_cos_sin(s, head_dim, self.rope_theta)
        causal_mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, None]

        hidden_states: List[jnp.ndarray] = [x]
        for i in range(self.num_hidden_layers):
            x = LlamaDecoderLayer(
                self.hidden_size, self.intermediate_size,
                self.num_attention_heads, self.num_key_value_heads,
                self.rms_norm_eps, name=f"layers_{i}",
            )(x, cos, sin, causal_mask)
            hidden_states.append(x)

        hidden_states[-1] = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="norm")(hidden_states[-1])
        return tuple(hidden_states)


def extract_hunyuan_llm_embeddings(
    params, input_ids: jnp.ndarray, attention_mask: jnp.ndarray,
    hidden_state_skip_layer: int = 2, model: Optional[LlamaTextModel] = None,
) -> jnp.ndarray:
    """Runs the Llama tower and returns ``hidden_states[-(skip_layer+1)]``
    (``config.py``'s ``--hidden-state-skip-layer`` default: 2), i.e.
    ``hidden_states[-3]`` -- **not** the final ``norm``-ed layer
    (``apply_final_norm`` defaults to ``False``, see module docstring).
    Cropping (``crop_start``) and the pooled attention-mask slice are done
    by the caller (see ``examples/generate_hunyuan_video_1_0.py``), matching
    ``TextEncoder.encode``'s own separation of concerns.

    Args:
        params: Flax param tree for `LlamaTextModel`.
        input_ids: (B, S) int32 token ids, chat-template-formatted +
            right-padded by the caller's tokenizer.
        attention_mask: (B, S) -- unused by the model itself (see module
            docstring's right-padding argument), returned unchanged so the
            caller can crop it in lockstep with the hidden states.
    """
    if model is None:
        model = LlamaTextModel()
    hidden_states = model.apply({"params": params["params"]}, input_ids)
    return hidden_states[-(hidden_state_skip_layer + 1)]
