"""PyTorch state_dict -> Flax parameter tree mapping helpers shared across
Wan versions (and, in future, other model families whose PT modules follow
the same nn.Sequential / named-submodule conventions).
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy


def _set_nested_dict(d: dict, keys: list, value: Any):
    """Safely sets d[keys[0]][keys[1]]...[keys[-1]] = value without overwriting dicts."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _leaf_name(pt_key: str) -> str:
    return "bias" if pt_key.endswith(".bias") else "kernel"


# --------------------------------------------------------------------------
# Causal-VAE building blocks (ResidualBlock / AttentionBlock / Resample),
# byte-for-byte identical between Wan2.1-main/wan/modules/vae.py and
# Wan2.2-main/wan/modules/vae2_2.py.
# --------------------------------------------------------------------------

# Wan's ResidualBlock.residual is an nn.Sequential; these are its fixed
# index -> submodule-name assignments (see Wan2.1-main/wan/modules/vae.py).
_RESIDUAL_SEQ_NAMES = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}


def map_vae_block_submodule(sub_key: str, block_path: list, jax_tensor, jax_params: dict) -> bool:
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


def map_vae_tower(
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
        return map_vae_block_submodule(match.group(2), block_path, jax_tensor, jax_params)

    match = re.match(rf"{resample_key}\.(\d+)\.(.*)", sub_key)
    if match:
        block_path = [tower_name, f"{resample_key}_{match.group(1)}"]
        return map_vae_block_submodule(match.group(2), block_path, jax_tensor, jax_params)

    match = re.match(r"head\.(0|2)\.(gamma|weight|bias)$", sub_key)
    if match:
        idx, field = match.groups()
        leaf = "scale" if field == "gamma" else _leaf_name(sub_key)
        _set_nested_dict(jax_params, [tower_name, f"head_{idx}", leaf], jax_tensor)
        return True

    return False


# --------------------------------------------------------------------------
# WanDiT (transformer backbone), shared across Wan versions: the reference's
# `WanModel` (Wan2.1-main/wan/modules/model.py and Wan2.2-main/wan/modules/
# model.py) uses identical attribute names for every non-i2v-specific
# submodule (`self_attn`, `cross_attn`, `norm1/2/3`, `ffn.{0,2}`,
# `modulation`, `patch_embedding`, `text_embedding.{0,2}`,
# `time_embedding.{0,2}`, `time_projection.1`, `head.*`); only Wan2.1's i2v
# checkpoints add the extra `img_emb.*`/`cross_attn.{k,v,norm_k}_img.*` keys
# this mapper also handles, which simply never appear in a Wan2.2 state_dict.
# --------------------------------------------------------------------------

# Maps a WanSelfAttention/WanT2VCrossAttention/WanI2VCrossAttention PyTorch
# submodule attribute to the corresponding vidax `attend()` parameter
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


def map_wan_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan `WanModel` state_dict (Wan2.1 t2v/i2v, or Wan2.2
    t2v/i2v/ti2v) into a Flax param tree for `WanDiT`
    (`vidax.models.wan.wan2_1.dit.WanDiT` or `vidax.models.wan.wan2_2.dit.WanDiT`).
    i2v-only keys (`img_emb.*`, `cross_attn.{k,v,norm_k}_img.*`) are mapped
    whenever present; they simply never appear in a Wan2.2 checkpoint.
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
        # intentionally ignored.

    return {"params": jax_params}


# --------------------------------------------------------------------------
# UMT5-XXL text encoder, byte-for-byte identical across Wan versions.
# --------------------------------------------------------------------------

def map_wan_t5_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan `T5Encoder` (umt5_xxl, encoder-only) state_dict into
    a Flax param tree for `vidax.models.wan.common.t5.T5Encoder`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        # Embedding tables (`nn.Embedding`/`nn.Embed`) keep their PyTorch
        # (num_embeddings, features) layout as-is -- never transposed.
        is_embedding_table = (
            pt_key == "token_embedding.weight" or pt_key.endswith("pos_embedding.embedding.weight"))
        if is_embedding_table:
            jax_tensor = pt_tensor_to_numpy(pt_tensor)
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
