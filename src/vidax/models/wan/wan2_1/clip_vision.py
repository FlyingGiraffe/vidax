"""CLIP (XLM-RoBERTa-CLIP ViT-H/14) vision tower, for Wan2.1 I2V image conditioning.

A structural port of the *vision* half of the reference
``XLMRobertaCLIP``/``CLIPModel`` (Wan2.1-main/wan/modules/clip.py). The text
tower (``XLMRobertaWithHead``) is not ported: `CLIPModel.visual()` — the
only entry point the I2V pipeline actually calls — never touches it.

Two further truncations match the reference's own `visual()` call
(`use_31_block=True`):
  - Only 31 of the checkpoint's 32 vision transformer layers run; the last
    layer's output is never used, so it isn't instantiated at all.
  - `post_norm` and the pooling `head` (`AttentionPool`/`Linear`/param
    matching `pool_type`) are never reached either, since `use_31_block`
    returns right after the 31st transformer block. Their checkpoint
    weights are simply left unmapped by the converter.
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from vidax.core.attention import dot_product_attention

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipVisionAttentionBlock(nn.Module):
    dim: int = 1280
    num_heads: int = 16
    mlp_ratio: float = 4.0
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, s, _ = x.shape
        head_dim = self.dim // self.num_heads

        h = nn.LayerNorm(epsilon=self.eps, name="norm1")(x.astype(jnp.float32)).astype(x.dtype)
        qkv = nn.Dense(self.dim * 3, name="attn_to_qkv")(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, s, self.num_heads, head_dim)
        k = k.reshape(b, s, self.num_heads, head_dim)
        v = v.reshape(b, s, self.num_heads, head_dim)
        attn_out = dot_product_attention(q, k, v).reshape(b, s, self.dim)
        x = x + nn.Dense(self.dim, name="attn_proj")(attn_out)

        h = nn.LayerNorm(epsilon=self.eps, name="norm2")(x.astype(jnp.float32)).astype(x.dtype)
        h = nn.Dense(int(self.dim * self.mlp_ratio), name="mlp_0")(h)
        h = nn.gelu(h, approximate=False)  # reference uses exact (erf) GELU here, not tanh-approx.
        h = nn.Dense(self.dim, name="mlp_2")(h)
        return x + h


class ClipVisionTransformer(nn.Module):
    """ViT-H/14 vision tower. Defaults match ``clip_xlm_roberta_vit_h_14``."""
    image_size: int = 224
    patch_size: int = 14
    dim: int = 1280
    mlp_ratio: float = 4.0
    num_heads: int = 16
    num_layers: int = 31  # 31, not 32: see module docstring.
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, image_size, image_size, 3), CLIP-normalized (see
                `preprocess_image_for_clip`).

        Returns:
            (B, num_patches + 1, dim) token sequence (cls token + patches),
            after 31 transformer blocks -- *not* pooled or projected.
        """
        b = x.shape[0]
        x = nn.Conv(
            self.dim, (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size), padding="VALID",
            use_bias=False, name="patch_embedding")(x)
        x = x.reshape(b, -1, self.dim)

        cls = self.param("cls_embedding", nn.initializers.normal(stddev=self.dim ** -0.5),
                          (1, 1, self.dim))
        x = jnp.concatenate([jnp.broadcast_to(cls, (b, 1, self.dim)), x], axis=1)

        pos = self.param("pos_embedding", nn.initializers.normal(stddev=self.dim ** -0.5),
                          (1, x.shape[1], self.dim))
        x = x + pos

        x = nn.LayerNorm(epsilon=self.eps, name="pre_norm")(x.astype(jnp.float32)).astype(x.dtype)

        for i in range(self.num_layers):
            x = ClipVisionAttentionBlock(
                self.dim, self.num_heads, self.mlp_ratio, self.eps,
                name=f"transformer_{i}")(x)
        return x


def preprocess_image_for_clip(image: np.ndarray, image_size: int = 224) -> jnp.ndarray:
    """Matches `CLIPModel.visual`'s preprocessing.

    Args:
        image: (H, W, 3) uint8 RGB array (a single frame).
        image_size: CLIP's fixed input resolution (224 for ViT-H/14).

    Returns:
        (1, image_size, image_size, 3) float32 array, CLIP-normalized.
    """
    x = jnp.asarray(image, dtype=jnp.float32) / 255.0  # [0, 1]
    x = jax.image.resize(x, (image_size, image_size, 3), method="bicubic")
    mean = jnp.asarray(CLIP_MEAN, dtype=jnp.float32)
    std = jnp.asarray(CLIP_STD, dtype=jnp.float32)
    x = (x - mean) / std
    return x[None]
