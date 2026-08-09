"""Cosmos-Predict2.5 (2B and 14B -- see `configs.py`).

Both released sizes share one architecture (`dit.py`'s `CosmosDiT`),
config-driven the same way `vidax.models.wan.wan2_1`/`wan2_2` are: only
`dim`/`ffn_dim`/`num_heads`/`num_layers` differ between them (cross-checked
against each checkpoint's own tensor shapes) -- see `configs.py`.

The VAE is the Wan2.1 causal VAE, reused verbatim (Cosmos-Predict2.5's own
tokenizer config wraps Wan's `WanVAE` unchanged -- see
`vidax.models.cosmos2_5.dit`'s module docstring and
`refs/cosmos-predict2.5-main/cosmos_predict2/_src/predict2/tokenizers/
cosmos.py`, which is a re-export wrapper, not an independent implementation).
Import `vidax.models.wan.wan2_1.WanVAEDecoder`/`WanVAEEncoder` directly for
it, loaded via `vidax.translator.load_torch_checkpoint_to_jax(...,
model_type="wan2.1_vae")` against `checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth`
(shared by both sizes -- neither ships its own VAE checkpoint).
"""
from .dit import CosmosDiT, CosmosDiTBlock, CosmosFinalLayer
from .reason1 import (
    NUM_EMBEDDING_PADDING_TOKENS,
    Qwen2TextModel,
    Reason1Tokenizer,
    compute_reason1_embeddings,
    mean_normalize,
)

__all__ = [
    "CosmosDiT", "CosmosDiTBlock", "CosmosFinalLayer",
    "NUM_EMBEDDING_PADDING_TOKENS",
    "Qwen2TextModel",
    "Reason1Tokenizer",
    "compute_reason1_embeddings",
    "mean_normalize",
]
