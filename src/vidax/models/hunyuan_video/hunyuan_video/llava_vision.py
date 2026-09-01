"""LLaVA's vision side (CLIP ViT-L/14-336 vision tower + the 2-layer MLP
multimodal projector) -- HunyuanVideo-I2V's ``token_replace`` checkpoint
uses the *full* multimodal ``xtuner/llava-llama-3-8b-v1_1-transformers``
(not just its ``.language_model``, unlike T2V's ``llama_text.py``): the
reference image is fed through this vision tower + projector, and the
resulting patch embeddings are spliced directly into the Llama decoder's
input embedding sequence at the ``<image>`` placeholder's fixed positions
(see ``llama_text.py``'s ``forward_from_embeds``/splicing support and
``examples/generate_hunyuan_video.py``'s I2V path) -- a real interleaved
vision-language forward pass, not the channel-concat/token-concat schemes
HunyuanVideo-1.5's I2V uses.

Real config, confirmed against the checkpoint's own ``config.json``
(``xtuner/llava-llama-3-8b-v1_1-transformers``, not guessed):
``vision_config``: standard HF ``CLIPVisionModel`` shape --
``hidden_size=1024``, ``image_size=336``, ``patch_size=14`` (576 patch
tokens), ``intermediate_size=4096``, ``num_hidden_layers=24``,
``num_attention_heads=16``, ``hidden_act="quick_gelu"`` (CLIP's own
default -- confirmed via `transformers.CLIPVisionConfig`'s default, not
present as an explicit key in the checkpoint's config.json),
``layer_norm_eps=1e-5``. Top-level: ``image_token_index=128257``,
``vision_feature_layer=-2`` (second-to-last CLIP encoder layer, **not**
``post_layernorm``-ed -- see ``CLIPVisionModel.forward``'s
``output_hidden_states`` layers, which are the raw per-layer residual
stream, never the final `last_hidden_state`/pooled path),
``vision_feature_select_strategy="default"`` (drops the CLS token
position), ``projector_hidden_act="gelu"`` (exact/erf GELU, not
`quick_gelu` -- a different activation than the vision tower's own).
"""
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from vidax.core.attention import dot_product_attention


def _quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
    """``x * sigmoid(1.702 * x)`` -- CLIP's own default activation, distinct
    from the projector's plain (erf) GELU."""
    return x * nn.sigmoid(1.702 * x)


class ClipEncoderLayer(nn.Module):
    """Pre-LN transformer block -- port of ``CLIPEncoderLayer``
    (``layer_norm1 -> self_attn -> residual -> layer_norm2 -> mlp ->
    residual``, standard pre-LN, not the reference's own confusingly-named
    post-residual-add order -- read directly, matches)."""
    hidden_size: int
    num_heads: int
    intermediate_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, s, _ = x.shape
        head_dim = self.hidden_size // self.num_heads

        h = nn.LayerNorm(epsilon=self.eps, name="layer_norm1")(x)
        q = nn.Dense(self.hidden_size, name="self_attn_q_proj")(h).reshape(b, s, self.num_heads, head_dim)
        k = nn.Dense(self.hidden_size, name="self_attn_k_proj")(h).reshape(b, s, self.num_heads, head_dim)
        v = nn.Dense(self.hidden_size, name="self_attn_v_proj")(h).reshape(b, s, self.num_heads, head_dim)
        attn = dot_product_attention(q, k, v).reshape(b, s, self.hidden_size)
        x = x + nn.Dense(self.hidden_size, name="self_attn_out_proj")(attn)

        h = nn.LayerNorm(epsilon=self.eps, name="layer_norm2")(x)
        h = nn.Dense(self.intermediate_size, name="mlp_fc1")(h)
        h = _quick_gelu(h)
        h = nn.Dense(self.hidden_size, name="mlp_fc2")(h)
        return x + h


class ClipVisionModel(nn.Module):
    """HF ``CLIPVisionModel`` -- port of ``CLIPVisionEmbeddings`` +
    ``pre_layrnorm`` + ``CLIPEncoder`` (``post_layernorm``/pooling not
    ported -- LLaVA's ``get_image_features`` never reaches them, see this
    module's docstring on ``vision_feature_layer``)."""
    hidden_size: int = 1024
    image_size: int = 336
    patch_size: int = 14
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    eps: float = 1e-5

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """
        Args:
            pixel_values: (B, image_size, image_size, 3), CLIP-normalized
                (see ``preprocess_image_for_llava``).

        Returns:
            Tuple of ``num_hidden_layers + 1`` hidden-state tensors, each
            (B, num_patches + 1, hidden_size) -- index 0 is the embedding
            layer (post ``pre_layrnorm``), indices 1..N are each encoder
            layer's raw output. Matches HF's own ``output_hidden_states``
            indexing exactly (``hidden_states[0]`` is pre-encoder, not the
            final ``post_layernorm``-ed state at any index).
        """
        b = pixel_values.shape[0]
        x = nn.Conv(
            self.hidden_size, (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size), padding="VALID",
            use_bias=False, name="patch_embedding")(pixel_values)
        x = x.reshape(b, -1, self.hidden_size)

        cls = self.param("class_embedding", nn.initializers.normal(stddev=0.02), (self.hidden_size,))
        x = jnp.concatenate([jnp.broadcast_to(cls, (b, 1, self.hidden_size)), x], axis=1)

        num_positions = x.shape[1]
        pos = nn.Embed(num_positions, self.hidden_size, name="position_embedding")(jnp.arange(num_positions))
        x = x + pos[None]

        x = nn.LayerNorm(epsilon=self.eps, name="pre_layrnorm")(x)

        hidden_states = [x]
        for i in range(self.num_hidden_layers):
            x = ClipEncoderLayer(
                self.hidden_size, self.num_attention_heads, self.intermediate_size, self.eps,
                name=f"encoder_layers_{i}")(x)
            hidden_states.append(x)
        return tuple(hidden_states)


class LlavaMultiModalProjector(nn.Module):
    """``linear_1 -> GELU -> linear_2`` -- port of ``LlavaMultiModalProjector``.
    Plain (erf) GELU (``projector_hidden_act="gelu"``), distinct from the
    vision tower's own ``quick_gelu``."""
    text_hidden_size: int = 4096

    @nn.compact
    def __call__(self, image_features: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.text_hidden_size, name="linear_1")(image_features)
        x = nn.gelu(x, approximate=False)
        return nn.Dense(self.text_hidden_size, name="linear_2")(x)


def get_llava_image_features(
    clip_params, clip_model: ClipVisionModel, projector_params, projector: LlavaMultiModalProjector,
    pixel_values: jnp.ndarray, vision_feature_layer: int = -2,
) -> jnp.ndarray:
    """``LlavaModel.get_image_features``: vision tower -> pick
    ``vision_feature_layer`` -> drop the CLS token (``"default"`` select
    strategy, the only one this checkpoint uses) -> project.

    Returns (B, 576, text_hidden_size) -- 576 = (336 // 14) ** 2, matching
    ``constants.py``'s ``image_emb_len``.
    """
    hidden_states = clip_model.apply(clip_params, pixel_values)
    selected = hidden_states[vision_feature_layer][:, 1:]  # drop CLS
    return projector.apply(projector_params, selected)
