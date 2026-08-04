"""PyTorch state_dict -> Flax parameter tree key mappings for Wan2.1."""
import os
import re
import safetensors.numpy
import jax.numpy as jnp
from typing import Any, Dict

from .converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy


def _set_nested_dict(d: dict, keys: list, value: Any):
    """Safely sets d[keys[0]][keys[1]]...[keys[-1]] = value without overwriting dicts."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _leaf_name(pt_key: str) -> str:
    return "bias" if pt_key.endswith(".bias") else "kernel"


# --------------------------------------------------------------------------
# WanDiT (text-to-video transformer backbone)
# --------------------------------------------------------------------------

# Maps a WanSelfAttention/WanT2VCrossAttention/WanI2VCrossAttention PyTorch
# submodule attribute to the corresponding vidax `_attend()` parameter
# prefix. k_img/v_img/norm_k_img only ever appear under `cross_attn.` (i2v
# checkpoints only); self_attn keys never match them.
_ATTN_SUBMODULE_NAMES = {
    "q": "q", "k": "k", "v": "v", "o": "o",
    "norm_q": "norm_q", "norm_k": "norm_k",
    "k_img": "k_img", "v_img": "v_img", "norm_k_img": "norm_k_img",
}


def _map_attention_submodule(pt_sub: str, jax_prefix: str, block_path: list,
                              jax_tensor, jax_params: dict):
    """Handles keys under `self_attn.` / `cross_attn.` within one DiT block."""
    for pt_name, jax_suffix in _ATTN_SUBMODULE_NAMES.items():
        if pt_sub.startswith(f"{pt_name}."):
            field = pt_sub[len(pt_name) + 1:]
            leaf = "scale" if field == "weight" and "norm" in pt_name else (
                "bias" if field == "bias" else "kernel")
            _set_nested_dict(
                jax_params, block_path + [f"{jax_prefix}_{jax_suffix}", leaf],
                jax_tensor)
            return True
    return False


def map_wan2_1_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 `WanModel` (t2v or i2v) state_dict into a Flax
    param tree for `vidax.models.wan.dit.WanDiT`. i2v-only keys
    (`img_emb.*`, `cross_attn.{k,v,norm_k}_img.*`) are mapped whenever
    present; loading a t2v checkpoint into a `model_type="i2v"` WanDiT (or
    vice versa) will simply leave those params unset/unused, matching
    however the state_dict and the model's `model_type` are actually paired.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key.startswith("blocks."):
            match = re.match(r"blocks\.(\d+)\.(.*)", pt_key)
            block_idx, sub_key = match.group(1), match.group(2)
            block_path = [f"blocks_{block_idx}"]

            if sub_key.startswith("self_attn."):
                _map_attention_submodule(
                    sub_key[len("self_attn."):], "self_attn", block_path,
                    jax_tensor, jax_params)
            elif sub_key.startswith("cross_attn."):
                _map_attention_submodule(
                    sub_key[len("cross_attn."):], "cross_attn", block_path,
                    jax_tensor, jax_params)
            elif sub_key.startswith("norm3."):
                leaf = "scale" if sub_key.endswith(".weight") else "bias"
                _set_nested_dict(jax_params, block_path + ["norm3", leaf], jax_tensor)
            elif sub_key.startswith("ffn.0."):
                _set_nested_dict(
                    jax_params, block_path + ["ffn_0", _leaf_name(sub_key)], jax_tensor)
            elif sub_key.startswith("ffn.2."):
                _set_nested_dict(
                    jax_params, block_path + ["ffn_2", _leaf_name(sub_key)], jax_tensor)
            elif sub_key == "modulation":
                _set_nested_dict(jax_params, block_path + ["modulation"], jax_tensor)

        elif pt_key.startswith("patch_embedding."):
            _set_nested_dict(jax_params, ["patch_embedding", _leaf_name(pt_key)], jax_tensor)

        elif pt_key.startswith("text_embedding."):
            match = re.match(r"text_embedding\.(\d+)\.", pt_key)
            _set_nested_dict(
                jax_params, [f"text_embedding_{match.group(1)}", _leaf_name(pt_key)],
                jax_tensor)

        elif pt_key.startswith("time_embedding."):
            match = re.match(r"time_embedding\.(\d+)\.", pt_key)
            _set_nested_dict(
                jax_params, [f"time_embedding_{match.group(1)}", _leaf_name(pt_key)],
                jax_tensor)

        elif pt_key.startswith("time_projection."):
            match = re.match(r"time_projection\.(\d+)\.", pt_key)
            _set_nested_dict(
                jax_params, [f"time_projection_{match.group(1)}", _leaf_name(pt_key)],
                jax_tensor)

        elif pt_key.startswith("head."):
            if pt_key == "head.modulation":
                _set_nested_dict(jax_params, ["head", "modulation"], jax_tensor)
            elif pt_key.startswith("head.head."):
                _set_nested_dict(jax_params, ["head", "head", _leaf_name(pt_key)], jax_tensor)

        elif pt_key.startswith("img_emb.proj."):
            # i2v checkpoints only. MLPProj.proj is an nn.Sequential:
            # 0=LayerNorm, 1=Linear, 2=GELU (no params), 3=Linear, 4=LayerNorm.
            match = re.match(r"img_emb\.proj\.(0|1|3|4)\.(weight|bias)$", pt_key)
            if match:
                idx, field = match.groups()
                leaf = "scale" if (idx in ("0", "4") and field == "weight") else _leaf_name(pt_key)
                _set_nested_dict(jax_params, ["img_emb", f"proj_{idx}", leaf], jax_tensor)

        # Any other keys (e.g. `img_emb.emb_pos`, flf2v-only) are
        # intentionally ignored: vidax's WanDiT only implements t2v and i2v.

    return {"params": jax_params}


# --------------------------------------------------------------------------
# WanVAE decoder
# --------------------------------------------------------------------------

# Wan2.1's ResidualBlock.residual is an nn.Sequential; these are its fixed
# index -> submodule-name assignments (see Wan2.1-main/wan/modules/vae.py).
_RESIDUAL_SEQ_NAMES = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}


def _map_vae_block_submodule(sub_key: str, block_path: list, jax_tensor, jax_params: dict) -> bool:
    """Handles keys inside one ResidualBlock, AttentionBlock, or Resample."""
    match = re.match(r"residual\.(\d+)\.(gamma|weight|bias)$", sub_key)
    if match:
        name = _RESIDUAL_SEQ_NAMES.get(match.group(1))
        if name is None:
            return True  # SiLU / Dropout entries carry no parameters.
        leaf = "scale" if match.group(2) == "gamma" else _leaf_name(sub_key)
        _set_nested_dict(jax_params, block_path + [name, leaf], jax_tensor)
        return True

    if sub_key.startswith("shortcut."):
        _set_nested_dict(jax_params, block_path + ["shortcut", _leaf_name(sub_key)], jax_tensor)
        return True

    if sub_key == "norm.gamma":
        _set_nested_dict(jax_params, block_path + ["norm", "scale"], jax_tensor)
        return True
    if sub_key.startswith("to_qkv."):
        _set_nested_dict(jax_params, block_path + ["to_qkv", _leaf_name(sub_key)], jax_tensor)
        return True
    if sub_key.startswith("proj."):
        _set_nested_dict(jax_params, block_path + ["proj", _leaf_name(sub_key)], jax_tensor)
        return True

    if sub_key.startswith("resample.1."):
        _set_nested_dict(jax_params, block_path + ["resample_1", _leaf_name(sub_key)], jax_tensor)
        return True
    if sub_key.startswith("time_conv."):
        _set_nested_dict(jax_params, block_path + ["time_conv", _leaf_name(sub_key)], jax_tensor)
        return True

    return False


def _map_vae_tower(
    pt_key: str, sub_key: str, jax_tensor, jax_params: dict,
    tower_name: str, resample_key: str,
) -> bool:
    """Handles keys inside `encoder.*` or `decoder.*` (`Encoder3d`/`Decoder3d`
    share this exact structure: conv1, middle.{0,1,2}, {resample_key}.{N},
    head.{0,2} -- only the resample-list's PT attribute name itself differs
    ("downsamples" vs "upsamples").
    """
    if sub_key.startswith("conv1."):
        _set_nested_dict(jax_params, [tower_name, "conv1", _leaf_name(sub_key)], jax_tensor)
        return True

    match = re.match(r"middle\.(\d+)\.(.*)", sub_key)
    if match:
        block_path = [tower_name, f"middle_{match.group(1)}"]
        return _map_vae_block_submodule(match.group(2), block_path, jax_tensor, jax_params)

    match = re.match(rf"{resample_key}\.(\d+)\.(.*)", sub_key)
    if match:
        block_path = [tower_name, f"{resample_key}_{match.group(1)}"]
        return _map_vae_block_submodule(match.group(2), block_path, jax_tensor, jax_params)

    match = re.match(r"head\.(0|2)\.(gamma|weight|bias)$", sub_key)
    if match:
        idx, field = match.groups()
        leaf = "scale" if field == "gamma" else _leaf_name(sub_key)
        _set_nested_dict(jax_params, [tower_name, f"head_{idx}", leaf], jax_tensor)
        return True

    return False


def map_wan2_1_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 `WanVAE_` state_dict into a Flax param tree for
    `vidax.models.wan.vae.WanVAEDecoder`/`WanVAEEncoder`.

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
            _map_vae_tower(pt_key, pt_key[len("decoder."):], jax_tensor, jax_params,
                            "decoder", "upsamples")
        elif pt_key.startswith("encoder."):
            _map_vae_tower(pt_key, pt_key[len("encoder."):], jax_tensor, jax_params,
                            "encoder", "downsamples")

    return {"params": jax_params}


# --------------------------------------------------------------------------
# UMT5-XXL text encoder
# --------------------------------------------------------------------------

def map_wan2_1_t5_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 `T5Encoder` (umt5_xxl, encoder-only) state_dict
    into a Flax param tree for `vidax.models.wan.t5.T5Encoder`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        # Embedding tables (`nn.Embedding`/`nn.Embed`) keep their PyTorch
        # (num_embeddings, features) layout as-is -- never transposed.
        is_embedding_table = (
            pt_key == "token_embedding.weight" or pt_key.endswith("pos_embedding.embedding.weight"))
        if is_embedding_table:
            jax_tensor = jnp.array(pt_tensor_to_numpy(pt_tensor))
        else:
            jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key == "token_embedding.weight":
            _set_nested_dict(jax_params, ["token_embedding", "embedding"], jax_tensor)
        elif pt_key == "norm.weight":
            _set_nested_dict(jax_params, ["norm", "scale"], jax_tensor)
        elif pt_key.startswith("blocks."):
            match = re.match(r"blocks\.(\d+)\.(.*)", pt_key)
            block_path = [f"blocks_{match.group(1)}"]
            sub_key = match.group(2)

            if sub_key in ("norm1.weight", "norm2.weight"):
                name = sub_key.split(".")[0]
                _set_nested_dict(jax_params, block_path + [name, "scale"], jax_tensor)
            elif sub_key.startswith("attn."):
                sub = sub_key[len("attn."):].split(".")[0]  # q, k, v, o
                _set_nested_dict(jax_params, block_path + [f"attn_{sub}", "kernel"], jax_tensor)
            elif sub_key == "ffn.gate.0.weight":
                _set_nested_dict(jax_params, block_path + ["ffn_gate_0", "kernel"], jax_tensor)
            elif sub_key in ("ffn.fc1.weight", "ffn.fc2.weight"):
                name = "ffn_" + sub_key.split(".")[1]
                _set_nested_dict(jax_params, block_path + [name, "kernel"], jax_tensor)
            elif sub_key == "pos_embedding.embedding.weight":
                _set_nested_dict(jax_params, block_path + ["pos_embedding", "embedding"], jax_tensor)

    return {"params": jax_params}


# --------------------------------------------------------------------------
# CLIP (ViT-H/14) vision tower, for I2V image conditioning
# --------------------------------------------------------------------------

def map_wan2_1_clip_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.1 CLIP (`clip_xlm_roberta_vit_h_14`) checkpoint into
    a Flax param tree for `vidax.models.wan.clip_vision.ClipVisionTransformer`.

    Only `visual.*` keys are mapped -- the text tower (`textual.*`) and
    `log_scale` are unused by the I2V pipeline (see
    `vidax.models.wan.clip_vision`'s module docstring), as are
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
            jax_params["cls_embedding"] = jnp.array(pt_tensor_to_numpy(pt_tensor))
            continue
        if sub_key == "pos_embedding":
            jax_params["pos_embedding"] = jnp.array(pt_tensor_to_numpy(pt_tensor))
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


def _load_pt_state_dict(checkpoint_path: str) -> Dict:
    """Loads a PyTorch state_dict from either a `.safetensors` or a `.pth`/`.pt`
    checkpoint (the released Wan2.1 checkpoints ship the DiT as `.safetensors`
    but the VAE and T5 text encoder as raw `.pth`).
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".safetensors":
        return safetensors.numpy.load_file(checkpoint_path)
    elif ext in (".pth", ".pt", ".bin"):
        import torch  # Optional dependency; install the `torch` extra.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return {k: v.detach().cpu() for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unrecognized checkpoint extension: '{ext}' ({checkpoint_path})")


def load_torch_checkpoint_to_jax(checkpoint_path: str, model_type: str = "wan2.1_dit") -> Dict:
    """Loads a Wan2.1 PyTorch checkpoint into a JAX/Flax parameter dict.

    Args:
        checkpoint_path: Path to the state_dict (`.safetensors`, `.pth`, or `.pt`).
        model_type: One of "wan2.1_dit" (WanDiT), "wan2.1_vae"
            (WanVAEDecoder/WanVAEEncoder), "wan2.1_t5" (T5Encoder), or
            "wan2.1_clip" (ClipVisionTransformer).
    """
    pt_state_dict = _load_pt_state_dict(checkpoint_path)

    if model_type == "wan2.1_dit":
        return map_wan2_1_dit_keys(pt_state_dict)
    elif model_type == "wan2.1_vae":
        return map_wan2_1_vae_keys(pt_state_dict)
    elif model_type == "wan2.1_t5":
        return map_wan2_1_t5_keys(pt_state_dict)
    elif model_type == "wan2.1_clip":
        return map_wan2_1_clip_keys(pt_state_dict)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
