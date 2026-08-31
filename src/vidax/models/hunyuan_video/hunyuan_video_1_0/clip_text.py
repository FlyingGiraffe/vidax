"""HunyuanVideo 1.0's ``"clipL"`` text encoder -- a standard OpenAI CLIP
ViT-L/14 **text** tower (`openai/clip-vit-large-patch14`, HF
``CLIPTextModel``), used only for its *pooled* output (``text_states_2``,
``text_states_dim_2=768``), which feeds the DiT's ``vector_in`` alongside
the timestep embedding (see ``dit.py``'s module docstring). Standard,
small, pre-LN transformer -- no existing CLIP *text* tower port in vidax
(``wan/wan2_1/clip_vision.py`` is CLIP *vision*, a different tower
entirely), so this is a fresh port.

Real config (``openai/clip-vit-large-patch14/config.json``'s
``text_config``): ``hidden_size=768``, ``num_hidden_layers=12``,
``num_attention_heads=12``, ``intermediate_size=3072``,
``hidden_act="quick_gelu"``, ``max_position_embeddings=77``,
``layer_norm_eps=1e-5``, ``vocab_size=49408``. Causal self-attention
throughout (standard CLIP text convention -- every token can only attend to
itself and earlier tokens, even though CLIP-L is an encoder, not a
decoder).

**Pooling**: ``pooled_output = final_layer_norm(last_hidden_state)[arange(B),
argmax(input_ids, axis=-1)]`` -- the original CLIP tokenizer's EOS token id
(49407) is the *largest* token id in its vocabulary, so ``argmax`` over
``input_ids`` locates the EOS position for any *unpadded-at-EOS* sequence
(CLIP's own padding token id is 0, the smallest possible id, so it never
wins the argmax) -- HF's ``CLIPTextModel``'s exact pooling rule, confirmed
by reading ``transformers.models.clip.modeling_clip.CLIPTextTransformer.
forward`` directly.
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


def _quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(1.702 * x)


class ClipAttention(nn.Module):
    """Standard MHA (no GQA), causal mask, all Dense layers biased --
    matches ``CLIPAttention`` (``q_proj``/``k_proj``/``v_proj``/``out_proj``,
    every one ``bias=True``)."""
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal_mask: jnp.ndarray) -> jnp.ndarray:
        b, s, _ = x.shape
        head_dim = self.hidden_size // self.num_heads
        scale = head_dim ** -0.5

        q = nn.Dense(self.hidden_size, name="q_proj")(x) * scale
        k = nn.Dense(self.hidden_size, name="k_proj")(x)
        v = nn.Dense(self.hidden_size, name="v_proj")(x)

        q = q.reshape(b, s, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        logits = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
        logits = jnp.where(causal_mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.hidden_size)
        return nn.Dense(self.hidden_size, name="out_proj")(out)


class ClipMLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.intermediate_size, name="fc1")(x)
        h = _quick_gelu(h)
        return nn.Dense(self.hidden_size, name="fc2")(h)


class ClipEncoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, causal_mask: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm(epsilon=self.eps, name="layer_norm1")(x)
        x = x + ClipAttention(self.hidden_size, self.num_heads, name="self_attn")(h, causal_mask)
        h = nn.LayerNorm(epsilon=self.eps, name="layer_norm2")(x)
        x = x + ClipMLP(self.hidden_size, self.intermediate_size, name="mlp")(h)
        return x


class ClipTextModel(nn.Module):
    """``CLIPTextModel``, pooled-output only (see ``extract_clip_pooled``).
    Defaults match ``openai/clip-vit-large-patch14``'s real ``text_config``.
    """
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    eps: float = 1e-5

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """input_ids: (B, S) int32 -> last_hidden_state (B, S, hidden_size),
        post-``final_layer_norm`` (pooling is done by the caller, see
        ``extract_clip_pooled``)."""
        b, s = input_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.hidden_size, name="embeddings_token_embedding")(input_ids)
        pos_ids = jnp.arange(s)
        pos_emb = nn.Embed(
            self.max_position_embeddings, self.hidden_size, name="embeddings_position_embedding")(pos_ids)
        x = tok_emb + pos_emb[None]

        causal_mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, None]
        for i in range(self.num_hidden_layers):
            x = ClipEncoderLayer(
                self.hidden_size, self.intermediate_size, self.num_attention_heads, self.eps,
                name=f"layers_{i}")(x, causal_mask)

        return nn.LayerNorm(epsilon=self.eps, name="final_layer_norm")(x)


def extract_clip_pooled(params, input_ids: jnp.ndarray, model: "ClipTextModel" = None) -> jnp.ndarray:
    """Runs the CLIP text tower and returns the pooled (EOS-position) output
    -- see module docstring's "Pooling" section."""
    if model is None:
        model = ClipTextModel()
    last_hidden_state = model.apply({"params": params["params"]}, input_ids)
    eos_pos = jnp.argmax(input_ids, axis=-1)
    return last_hidden_state[jnp.arange(input_ids.shape[0]), eos_pos]
