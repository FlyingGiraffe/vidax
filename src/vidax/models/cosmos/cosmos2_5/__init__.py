"""Cosmos-Predict2.5 2B.

The VAE is the Wan2.1 causal VAE, reused verbatim (Cosmos-Predict2.5's own
tokenizer config wraps Wan's `WanVAE` unchanged -- see
`vidax.models.cosmos.cosmos2_5.dit`'s module docstring and
`refs/cosmos-predict2.5-main/cosmos_predict2/_src/predict2/tokenizers/
cosmos.py`, which is a re-export wrapper, not an independent implementation).
Import `vidax.models.wan.wan2_1.WanVAEDecoder`/`WanVAEEncoder` directly for
it, loaded via `vidax.translator.load_torch_checkpoint_to_jax(...,
model_type="wan2.1_vae")` against `checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth`.
"""
from .dit import CosmosDiT, CosmosDiTBlock, CosmosFinalLayer

__all__ = ["CosmosDiT", "CosmosDiTBlock", "CosmosFinalLayer"]
