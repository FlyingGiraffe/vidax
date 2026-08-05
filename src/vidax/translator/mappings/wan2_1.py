"""PyTorch state_dict -> Flax parameter tree key mappings specific to Wan2.1
(its DiT mapping is shared with Wan2.2 -- see `map_wan_dit_keys` in `.common`).
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy
from .common import _leaf_name, _set_nested_dict, map_vae_tower, map_wan_dit_keys

# Kept as an importable alias for callers that referred to this Wan2.1-specific
# name before the DiT mapper was generalized across Wan versions.
map_wan2_1_dit_keys = map_wan_dit_keys

# --------------------------------------------------------------------------
# WanVAE (Wan2.1's own causal VAE: encoder + decoder)
# --------------------------------------------------------------------------

def map_wan2_1_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 `WanVAE_` state_dict into a Flax param tree for
    `vidax.models.wan.wan2_1.vae.WanVAEDecoder`/`WanVAEEncoder`.

    Both towers' weights are mapped from the same checkpoint; T2V generation
    only needs the decoder ones (`conv2.*`, `decoder.*`), I2V's image
    conditioning also needs the encoder ones (`conv1.*`, `encoder.*`).
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key.startswith("conv2."):
            _set_nested_dict(jax_params, ["conv2", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("conv1."):
            _set_nested_dict(jax_params, ["conv1", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("decoder."):
            map_vae_tower(pt_key, pt_key[len("decoder."):], jax_tensor, jax_params,
                           "decoder", "upsamples")
        elif pt_key.startswith("encoder."):
            map_vae_tower(pt_key, pt_key[len("encoder."):], jax_tensor, jax_params,
                           "encoder", "downsamples")

    return {"params": jax_params}


# --------------------------------------------------------------------------
# CLIP (ViT-H/14) vision tower, for I2V image conditioning
# --------------------------------------------------------------------------

def map_wan2_1_clip_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 CLIP (`clip_xlm_roberta_vit_h_14`) checkpoint into
    a Flax param tree for `vidax.models.wan.wan2_1.clip_vision.ClipVisionTransformer`.

    Only `visual.*` keys are mapped -- the text tower (`textual.*`) and
    `log_scale` are unused by the I2V pipeline (see
    `vidax.models.wan.wan2_1.clip_vision`'s module docstring), as are
    `visual.transformer.31.*` (the 32nd layer), `visual.post_norm.*`, and
    `visual.head*` (the reference's own `visual(..., use_31_block=True)`
    call never reaches any of them either).
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith("visual."):
            continue
        sub_key = pt_key[len("visual."):]

        if sub_key == "cls_embedding":
            jax_params["cls_embedding"] = pt_tensor_to_numpy(pt_tensor)
            continue
        if sub_key == "pos_embedding":
            jax_params["pos_embedding"] = pt_tensor_to_numpy(pt_tensor)
            continue

        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if sub_key.startswith("patch_embedding."):
            _set_nested_dict(jax_params, ["patch_embedding", _leaf_name(sub_key)], jax_tensor)
            continue
        if sub_key.startswith("pre_norm."):
            leaf = "scale" if sub_key.endswith(".weight") else "bias"
            _set_nested_dict(jax_params, ["pre_norm", leaf], jax_tensor)
            continue

        match = re.match(r"transformer\.(\d+)\.(.*)", sub_key)
        if not match:
            continue  # post_norm.*, head* -- unused, see docstring.
        layer_idx, field = match.groups()
        if int(layer_idx) >= 31:
            continue  # the 32nd layer -- unused, see docstring.
        block_path = [f"transformer_{layer_idx}"]

        if field.startswith("norm1."):
            leaf = "scale" if field.endswith(".weight") else "bias"
            _set_nested_dict(jax_params, block_path + ["norm1", leaf], jax_tensor)
        elif field.startswith("norm2."):
            leaf = "scale" if field.endswith(".weight") else "bias"
            _set_nested_dict(jax_params, block_path + ["norm2", leaf], jax_tensor)
        elif field.startswith("attn.to_qkv."):
            _set_nested_dict(jax_params, block_path + ["attn_to_qkv", _leaf_name(field)], jax_tensor)
        elif field.startswith("attn.proj."):
            _set_nested_dict(jax_params, block_path + ["attn_proj", _leaf_name(field)], jax_tensor)
        elif field.startswith("mlp.0."):
            _set_nested_dict(jax_params, block_path + ["mlp_0", _leaf_name(field)], jax_tensor)
        elif field.startswith("mlp.2."):
            _set_nested_dict(jax_params, block_path + ["mlp_2", _leaf_name(field)], jax_tensor)

    return {"params": jax_params}
