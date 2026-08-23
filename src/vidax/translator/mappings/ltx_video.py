"""PyTorch state_dict -> Flax parameter tree mapping for LTX-Video 0.9.8.

Every released checkpoint (`ltxv-{2b,13b}-0.9.8-*.safetensors`) is a single
flat file bundling the DiT (`model.diffusion_model.*`, ComfyUI-style prefix
-- see `Transformer3DModel.load_state_dict`'s prefix-strip in the
reference) and the VAE (`vae.*`) together; `map_ltx_video_dit_keys`/
`map_ltx_video_vae_keys` each filter to their own prefix and ignore the
other, so both can be called on the exact same loaded state_dict.
"""
import re
from typing import Any, Dict

from .common import _leaf_name, _set_nested_dict
from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy

_DIT_PREFIX = "model.diffusion_model."
_VAE_PREFIX = "vae."

# `scale_shift_table` tensors are raw AdaLN modulation parameters (shape
# (6, dim), (2, dim), or (4, dim)) -- like Wan's `modulation`/`head.modulation`
# (see `vidax.translator.converter.convert_pt_tensor_to_jax`'s docstring),
# these must NOT go through the generic ndim==2 "Linear weight" transpose.
_NO_TRANSPOSE_SUFFIXES = ("scale_shift_table",)


def _convert(pt_key: str, pt_tensor) -> Any:
    if pt_key.endswith(_NO_TRANSPOSE_SUFFIXES):
        return pt_tensor_to_numpy(pt_tensor)
    return convert_pt_tensor_to_jax(pt_key, pt_tensor)


def _map_attention_submodule(sub_key: str, attn_path: list, jax_tensor, jax_params: dict) -> bool:
    """Handles keys under `attn1.`/`attn2.` (self-/cross-attention), matching
    `vidax.models.ltx_video.dit.LTXAttention`'s own submodule names.
    """
    if sub_key in ("q_norm.weight", "k_norm.weight"):
        name = "q_norm_scale" if sub_key.startswith("q_norm") else "k_norm_scale"
        _set_nested_dict(jax_params, attn_path + [name], jax_tensor)
        return True
    match = re.match(r"(to_q|to_k|to_v)\.(weight|bias)$", sub_key)
    if match:
        name, field = match.groups()
        _set_nested_dict(jax_params, attn_path + [name, "bias" if field == "bias" else "kernel"], jax_tensor)
        return True
    match = re.match(r"to_out\.0\.(weight|bias)$", sub_key)
    if match:
        field = match.group(1)
        _set_nested_dict(jax_params, attn_path + ["to_out_0", "bias" if field == "bias" else "kernel"], jax_tensor)
        return True
    return False


def map_ltx_video_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates LTX-Video's `Transformer3DModel` state_dict (filtered to
    its `model.diffusion_model.` prefix) into a Flax param tree for
    `vidax.models.ltx_video.dit.LTXDiT`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(_DIT_PREFIX):
            continue
        key = pt_key[len(_DIT_PREFIX):]
        jax_tensor = _convert(pt_key, pt_tensor)

        if key == "scale_shift_table":
            _set_nested_dict(jax_params, ["scale_shift_table"], jax_tensor)
        elif key.startswith("patchify_proj."):
            _set_nested_dict(jax_params, ["patchify_proj", _leaf_name(key)], jax_tensor)
        elif key.startswith("proj_out."):
            _set_nested_dict(jax_params, ["proj_out", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.emb.timestep_embedder.linear_1."):
            _set_nested_dict(jax_params, ["adaln_timestep_embedder_linear_1", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.emb.timestep_embedder.linear_2."):
            _set_nested_dict(jax_params, ["adaln_timestep_embedder_linear_2", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.linear."):
            _set_nested_dict(jax_params, ["adaln_linear", _leaf_name(key)], jax_tensor)
        elif key.startswith("caption_projection.linear_1."):
            _set_nested_dict(jax_params, ["caption_projection_linear_1", _leaf_name(key)], jax_tensor)
        elif key.startswith("caption_projection.linear_2."):
            _set_nested_dict(jax_params, ["caption_projection_linear_2", _leaf_name(key)], jax_tensor)
        elif key.startswith("transformer_blocks."):
            match = re.match(r"transformer_blocks\.(\d+)\.(.*)", key)
            block_idx, sub_key = match.group(1), match.group(2)
            block_path = [f"blocks_{block_idx}"]

            if sub_key == "scale_shift_table":
                _set_nested_dict(jax_params, block_path + ["scale_shift_table"], jax_tensor)
            elif sub_key.startswith("attn1."):
                _map_attention_submodule(sub_key[len("attn1."):], block_path + ["attn1"], jax_tensor, jax_params)
            elif sub_key.startswith("attn2."):
                _map_attention_submodule(sub_key[len("attn2."):], block_path + ["attn2"], jax_tensor, jax_params)
            elif sub_key.startswith("ff.net.0.proj."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_proj", _leaf_name(sub_key)], jax_tensor)
            elif sub_key.startswith("ff.net.2."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_out", _leaf_name(sub_key)], jax_tensor)
            # `norm1`/`norm2` carry no learnable params (elementwise_affine=False
            # for every released checkpoint) -- nothing to map.

    return {"params": jax_params}


def map_ltx_video_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates LTX-Video's `CausalVideoAutoencoder` state_dict (filtered
    to its `vae.` prefix) into a Flax param tree for
    `vidax.models.ltx_video.vae.LTXVAE`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(_VAE_PREFIX):
            continue
        if pt_key.startswith("vae.per_channel_statistics."):
            continue  # loaded separately, straight from the checkpoint, by the example script.
        key = pt_key[len(_VAE_PREFIX):]
        jax_tensor = _convert(pt_key, pt_tensor)

        match = re.match(r"(encoder|decoder)\.(.*)", key)
        if not match:
            continue
        tower, sub_key = match.groups()
        tower_path = [tower]

        if sub_key in ("conv_in.conv.weight", "conv_in.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_in", _leaf_name(sub_key)], jax_tensor)
        elif sub_key in ("conv_out.conv.weight", "conv_out.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_out", _leaf_name(sub_key)], jax_tensor)
        elif sub_key == "timestep_scale_multiplier":
            _set_nested_dict(jax_params, tower_path + ["timestep_scale_multiplier"], jax_tensor)
        elif sub_key == "last_scale_shift_table":
            _set_nested_dict(jax_params, tower_path + ["last_scale_shift_table"], jax_tensor)
        elif sub_key.startswith("last_time_embedder.timestep_embedder."):
            lin = re.match(r"last_time_embedder\.timestep_embedder\.(linear_[12])\.(weight|bias)$", sub_key)
            _set_nested_dict(
                jax_params, tower_path + [f"last_time_embedder_timestep_embedder_{lin.group(1)}", _leaf_name(sub_key)],
                jax_tensor)
        else:
            match = re.match(rf"(?:down|up)_blocks\.(\d+)\.(.*)", sub_key)
            if not match:
                continue
            block_idx, block_sub = match.groups()
            block_path = tower_path + [f"{'down' if tower == 'encoder' else 'up'}_blocks_{block_idx}"]

            if block_sub in ("conv.conv.weight", "conv.conv.bias"):
                # SpaceToDepthDownsample/DepthToSpaceUpsample's single conv.
                _set_nested_dict(jax_params, block_path + ["conv", _leaf_name(block_sub)], jax_tensor)
            elif block_sub == "scale_shift_table":
                # Shouldn't occur at this level (only per-resnet), but guard anyway.
                continue
            elif block_sub.startswith("time_embedder.timestep_embedder."):
                lin = re.match(r"time_embedder\.timestep_embedder\.(linear_[12])\.(weight|bias)$", block_sub)
                _set_nested_dict(
                    jax_params, block_path + [f"time_embedder_timestep_embedder_{lin.group(1)}", _leaf_name(block_sub)],
                    jax_tensor)
            else:
                res_match = re.match(r"res_blocks\.(\d+)\.(.*)", block_sub)
                if not res_match:
                    continue
                res_idx, res_sub = res_match.groups()
                res_path = block_path + [f"res_blocks_{res_idx}"]
                if res_sub == "scale_shift_table":
                    _set_nested_dict(jax_params, res_path + ["scale_shift_table"], jax_tensor)
                elif res_sub.startswith("conv1.conv."):
                    _set_nested_dict(jax_params, res_path + ["conv1", _leaf_name(res_sub)], jax_tensor)
                elif res_sub.startswith("conv2.conv."):
                    _set_nested_dict(jax_params, res_path + ["conv2", _leaf_name(res_sub)], jax_tensor)

    return {"params": jax_params}


def map_ltx_video_t5_keys(pt_state_dict: Dict) -> Dict:
    """Translates `PixArt-alpha/PixArt-XL-2-1024-MS`'s `text_encoder`
    (a plain HuggingFace `T5EncoderModel`) state_dict into a Flax param tree
    for `vidax.models.ltx_video.t5.T5Encoder`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        # Embedding tables (token embedding, relative-position-bias table)
        # keep their PyTorch (num_embeddings, features) layout as-is --
        # never transposed like an ordinary Linear weight would be (same
        # trap `map_wan_t5_keys` avoids for UMT5's embedding tables).
        is_embedding_table = pt_key == "shared.weight" or pt_key.endswith("relative_attention_bias.weight")
        jax_tensor = pt_tensor_to_numpy(pt_tensor) if is_embedding_table else convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key == "shared.weight":
            _set_nested_dict(jax_params, ["token_embedding", "embedding"], jax_tensor)
        elif pt_key == "encoder.final_layer_norm.weight":
            _set_nested_dict(jax_params, ["norm", "scale"], jax_tensor)
        elif pt_key == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight":
            _set_nested_dict(jax_params, ["relative_attention_bias", "embedding"], jax_tensor)
        elif pt_key.startswith("encoder.block."):
            match = re.match(r"encoder\.block\.(\d+)\.layer\.(0|1)\.(.*)", pt_key)
            if not match:
                continue
            block_idx, layer_idx, sub_key = match.groups()
            block_path = [f"blocks_{block_idx}"]

            if sub_key == "layer_norm.weight":
                norm_name = "norm1" if layer_idx == "0" else "norm2"
                _set_nested_dict(jax_params, block_path + [norm_name, "scale"], jax_tensor)
            elif layer_idx == "0" and sub_key.startswith("SelfAttention."):
                attn_sub = sub_key[len("SelfAttention."):]
                match2 = re.match(r"(q|k|v|o)\.weight$", attn_sub)
                if match2:
                    _set_nested_dict(jax_params, block_path + [f"attn_{match2.group(1)}", "kernel"], jax_tensor)
            elif layer_idx == "1" and sub_key.startswith("DenseReluDense."):
                ffn_sub = sub_key[len("DenseReluDense."):]
                ffn_name = {"wi_0.weight": "ffn_gate_0", "wi_1.weight": "ffn_fc1", "wo.weight": "ffn_fc2"}.get(ffn_sub)
                if ffn_name:
                    _set_nested_dict(jax_params, block_path + [ffn_name, "kernel"], jax_tensor)

    return {"params": jax_params}
