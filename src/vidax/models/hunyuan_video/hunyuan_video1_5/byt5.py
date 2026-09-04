"""Glyph-SDXL-v2 byT5 glyph-conditioning encoder for HunyuanVideo-1.5.

Structural port of ``hyvideo/models/text_encoders/byT5/__init__.py``. Two
pieces:
  - ``ByT5Mapper``: the small MLP that projects byT5's raw per-byte
    embeddings up to the DiT's ``hidden_size`` (this is what
    ``hunyuan_video1_5/dit.py``'s ``byt5_in`` uses directly).
  - The underlying T5-encoder-only tower (Glyph-SDXL-v2 fine-tuned on top
    of ``google/byt5-small``, ``T5ForConditionalGeneration.from_pretrained(
    ...).get_encoder()`` in the reference) that produces those raw
    embeddings from tokenized glyph/color prompt text -- **architecturally
    identical** to ``vidax.models.ltx_video.t5.T5Encoder`` (bidirectional
    relative-position bucketing, gated-GELU FFN, no projection biases, one
    shared relative-position-bias table across layers): confirmed against
    the downloaded checkpoint's own
    ``text_encoder/byt5-small/config.json``: ``d_model=1472``, ``num_heads=6``,
    ``d_kv=64``, ``d_ff=3584``, ``num_layers=12``,
    ``feed_forward_proj="gated-gelu"``, ``relative_attention_num_buckets=
    32`` -- reused directly via ``byt5_encoder()`` below rather than
    re-porting, matching this repo's precedent of not duplicating a
    T5-family architecture that's already been ported once (see
    ``ltx_video/t5.py``'s own docstring on its relationship to
    ``wan/common/t5.py``'s UMT5).

  Real ``vocab_size`` is **not** the base checkpoint's 384 -- Glyph-SDXL-v2
  calls ``tokenizer.add_tokens(...)`` + ``resize_token_embeddings(...)`` to
  add per-language color/font special tokens (counts read from
  ``Glyph-SDXL-v2/assets/{color_idx,multilingual_10-lang_idx}.json`` at
  checkpoint-build time) before saving ``byt5_model.pt`` -- so
  ``vocab_size`` must be read off the real checkpoint's embedding weight
  shape (``shared.weight`` / ``embed_tokens.weight``, see the translator),
  never hardcoded to 384.

The translator for ``byt5_model.pt`` (a raw ``torch.load``'d state_dict
with a ``module.text_tower.encoder.`` prefix stripped per the reference's
``create_byt5``, not a diffusers-style safetensors+config.json layout like
the DiT/VAE) is ``vidax.translator.mappings.hunyuan_video1_5.
map_hunyuan_video1_5_byt5_keys``. The ``Glyph-SDXL-v2``-aware tokenizer
wrapper (``ByT5PromptTokenizer``, mirroring
``vidax.models.ltx_video.t5.PixArtT5Tokenizer``'s pattern plus the added
special tokens) lives in ``examples/generate_hunyuan_video1_5.py``.
"""
import flax.linen as nn
import jax.numpy as jnp

from vidax.models.ltx_video.t5 import T5Encoder


def byt5_encoder(vocab_size: int) -> T5Encoder:
    """Builds a `T5Encoder` parameterized for Glyph-SDXL-v2's byT5-small
    tower. `vocab_size` must come from the real checkpoint's embedding
    weight shape (see module docstring) -- there is no fixed default.
    """
    return T5Encoder(
        vocab_size=vocab_size, dim=1472, num_heads=6, head_dim=64,
        dim_ffn=3584, num_layers=12, num_buckets=32, max_distance=128,
    )


class ByT5Mapper(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear (no residual
    for HunyuanVideo-1.5's usage, ``use_residual=False``) -- port of
    ``ByT5Mapper``.
    """
    in_dim: int = 1472
    hidden_dim: int = 2048
    out_dim1: int = 2048
    out_dim: int = 2048

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=1e-5, name="layernorm")(x)
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_dim, name="fc2")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.out_dim1, name="fc3")(x)
        return x
