"""SigLIP vision encoder for HunyuanVideo-1.5's I2V conditioning.

Structural port of HuggingFace `transformers.models.siglip.modeling_siglip`
(read directly from the installed package,
`transformers/models/siglip/modeling_siglip.py` -- the real checkpoint
itself, `black-forest-labs/FLUX.1-Redux-dev`'s `image_encoder` subfolder,
is gated and not yet downloaded, so this is verified against the public
`SiglipVisionModel`/`SiglipVisionConfig` architecture, not yet against real
weights -- do that once the checkpoint is available, same discipline as
every other component in this port).

Reference (`hyvideo/models/vision_encoder/__init__.py`): only
`SiglipVisionModel`'s `last_hidden_state` is used (patch tokens after the
final `post_layernorm`, no CLS token -- SigLIP has none) -- `pooler_output`
(`SiglipMultiheadAttentionPoolingHead`) is never consumed by HunyuanVideo-1.5,
so it's not ported here.

**Config values are placeholders pending the real checkpoint's
`config.json`** (`vision_states_dim=1152` in the DiT's own config strongly
suggests the `google/siglip-so400m-patch14-384`-family checkpoint --
`hidden_size=1152`, `patch_size=14` -- but `image_size`/`num_hidden_layers`/
`num_attention_heads`/`intermediate_size` must be read from the real
config once downloaded, not assumed from the DiT's `vision_states_dim`
alone). `siglip_kwargs_from_config()` below is the intended real-config
entry point once that file exists; `SiglipVisionEncoder`'s dataclass
defaults are HF's own `SiglipVisionConfig()` defaults (`hidden_size=768,
patch_size=16, image_size=224, num_hidden_layers=12,
num_attention_heads=12, intermediate_size=3072`), i.e. *not* HunyuanVideo-1.5's
real values -- do not deploy with the defaults, they are `siglip-base` era.
"""
from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp


def gelu_pytorch_tanh(x: jnp.ndarray) -> jnp.ndarray:
    """`ACT2FN["gelu_pytorch_tanh"]` == `nn.GELU(approximate="tanh")` == Flax's `nn.gelu(approximate=True)`."""
    return nn.gelu(x, approximate=True)


class SiglipMLP(nn.Module):
    intermediate_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.intermediate_size, name="fc1")(x)
        x = gelu_pytorch_tanh(x)
        return nn.Dense(self.hidden_size, name="fc2")(x)


class SiglipAttention(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.hidden_size // self.num_heads
        b, s, _ = x.shape
        q = nn.Dense(self.hidden_size, name="q_proj")(x).reshape(b, s, self.num_heads, head_dim)
        k = nn.Dense(self.hidden_size, name="k_proj")(x).reshape(b, s, self.num_heads, head_dim)
        v = nn.Dense(self.hidden_size, name="v_proj")(x).reshape(b, s, self.num_heads, head_dim)

        scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(b, s, self.hidden_size)
        return nn.Dense(self.hidden_size, name="out_proj")(out)


class SiglipEncoderLayer(nn.Module):
    hidden_size: int
    num_heads: int
    intermediate_size: int
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        h = nn.LayerNorm(epsilon=self.layer_norm_eps, name="layer_norm1")(x)
        h = SiglipAttention(self.hidden_size, self.num_heads, name="self_attn")(h)
        x = residual + h

        residual = x
        h = nn.LayerNorm(epsilon=self.layer_norm_eps, name="layer_norm2")(x)
        h = SiglipMLP(self.intermediate_size, self.hidden_size, name="mlp")(h)
        return residual + h


class SiglipVisionEmbeddings(nn.Module):
    hidden_size: int
    patch_size: int
    image_size: int

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        """pixel_values: (B, H, W, num_channels), channel-last."""
        x = nn.Conv(self.hidden_size, (self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size), padding="VALID",
                    name="patch_embedding")(pixel_values)
        b, gh, gw, c = x.shape
        x = x.reshape(b, gh * gw, c)
        num_positions = (self.image_size // self.patch_size) ** 2
        pos_emb = nn.Embed(num_positions, self.hidden_size, name="position_embedding")(
            jnp.arange(num_positions))
        return x + pos_emb[None]


class SiglipVisionEncoder(nn.Module):
    """`SiglipVisionModel` minus the (unused) pooling head. Defaults are HF's
    own `SiglipVisionConfig()` placeholders -- see module docstring; pass
    the real checkpoint's dims once downloaded.
    """
    hidden_size: int = 768
    patch_size: int = 16
    image_size: int = 224
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        """pixel_values: (B, H, W, 3) -> (B, num_patches, hidden_size)."""
        x = SiglipVisionEmbeddings(self.hidden_size, self.patch_size, self.image_size, name="embeddings")(pixel_values)
        for i in range(self.num_hidden_layers):
            x = SiglipEncoderLayer(
                self.hidden_size, self.num_attention_heads, self.intermediate_size,
                self.layer_norm_eps, name=f"layers_{i}")(x)
        return nn.LayerNorm(epsilon=self.layer_norm_eps, name="post_layernorm")(x)


def siglip_kwargs_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Builds `SiglipVisionEncoder` kwargs from a real `image_encoder/config.json`
    once the (currently gated) checkpoint is downloaded."""
    return dict(
        hidden_size=config["hidden_size"],
        patch_size=config["patch_size"],
        image_size=config["image_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        layer_norm_eps=config.get("layer_norm_eps", 1e-6),
    )
