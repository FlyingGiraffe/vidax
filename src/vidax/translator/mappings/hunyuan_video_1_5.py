"""PyTorch -> Flax key mapping for HunyuanVideo-1.5's DiT and VAE.

Built from ``hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py``
and ``hyvideo/models/autoencoders/hunyuanvideo_15_vae.py``'s real
``nn.Module`` attribute names (== state_dict key prefixes), cross-checked
against the actual downloaded checkpoints' key sets (never guessed).
Target Flax module trees are ``vidax.models.hunyuan_video.hunyuan_video_1_5.
dit.HunyuanVideo15DiT`` (+ ``common/dit_layers.py``) and ``...vae.
HunyuanVideo15VAE``.
"""
import re
from typing import Dict

from .common import _leaf_name, _set_nested_dict
from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy


def _lin(jax_params: dict, path: list, pt_key: str, arr) -> None:
    _set_nested_dict(jax_params, path + [_leaf_name(pt_key)], convert_pt_tensor_to_jax(pt_key, arr))


def _rms(jax_params: dict, path: list, arr) -> None:
    # RMSNorm's PyTorch key is ``<prefix>.weight`` -> Flax ``scale`` (1D, no transpose).
    _set_nested_dict(jax_params, path + ["scale"], convert_pt_tensor_to_jax("weight", arr))


_DOUBLE_QKV_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_(q|k|v)\.(weight|bias)$")
_DOUBLE_QK_NORM_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_(q|k)_norm\.weight$")
_DOUBLE_PROJ_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_proj\.(weight|bias)$")
_DOUBLE_MOD_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_mod\.linear\.(weight|bias)$")
_DOUBLE_MLP_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_mlp\.(fc1|fc2)\.(weight|bias)$")

_SINGLE_LIN1_RE = re.compile(r"^single_blocks\.(\d+)\.linear1_(q|k|v|mlp)\.(weight|bias)$")
_SINGLE_QK_NORM_RE = re.compile(r"^single_blocks\.(\d+)\.(q|k)_norm\.weight$")
_SINGLE_LIN2_RE = re.compile(r"^single_blocks\.(\d+)\.linear2\.fc\.(weight|bias)$")
_SINGLE_MOD_RE = re.compile(r"^single_blocks\.(\d+)\.modulation\.linear\.(weight|bias)$")

_REFINER_BLOCK_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$")
_REFINER_QKV_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(weight|bias)$")
_REFINER_PROJ_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_proj\.(weight|bias)$")
_REFINER_MLP_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$")
_REFINER_ADALN_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.adaLN_modulation\.1\.(weight|bias)$")

_VISION_SEQ_NAMES = {"0": "ln_0", "1": "linear_1", "3": "linear_3", "4": "ln_4"}


def map_hunyuan_video_1_5_dit_keys(pt_state_dict: Dict) -> Dict:
    jax_params: Dict = {}

    for pt_key, arr in pt_state_dict.items():
        if pt_key == "img_in.proj.weight":
            # Conv3d (out, in, pt, ph, pw), kernel==stride patchify -> flatten
            # to a plain Dense kernel. PyTorch's own init flattens the same
            # way (`self.proj.weight[...].view(out, -1)`) -- row-major, so
            # (in, pt, ph, pw) collapses with `in` slowest, matching
            # `dit.py:_patchify`'s (c, pt, ph, pw) flatten order exactly.
            out_ch = arr.shape[0]
            flat = arr.reshape(out_ch, -1)  # (out, in*pt*ph*pw)
            _set_nested_dict(jax_params, ["img_in_proj", "kernel"], flat.T)
            continue
        if pt_key == "img_in.proj.bias":
            _set_nested_dict(jax_params, ["img_in_proj", "bias"], arr)
            continue

        m = _DOUBLE_QKV_RE.match(pt_key)
        if m:
            i, stream, qkv, _ = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_attn", f"attn_{qkv}"], pt_key, arr)
            continue
        m = _DOUBLE_QK_NORM_RE.match(pt_key)
        if m:
            i, stream, qk = m.groups()
            _rms(jax_params, ["double_blocks_" + i, f"{stream}_attn", f"attn_{qk}_norm"], arr)
            continue
        m = _DOUBLE_PROJ_RE.match(pt_key)
        if m:
            i, stream, _ = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_attn_proj"], pt_key, arr)
            continue
        m = _DOUBLE_MOD_RE.match(pt_key)
        if m:
            i, stream, _ = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_mod", "linear"], pt_key, arr)
            continue
        m = _DOUBLE_MLP_RE.match(pt_key)
        if m:
            i, stream, fc, _ = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_mlp", fc], pt_key, arr)
            continue

        m = _SINGLE_LIN1_RE.match(pt_key)
        if m:
            i, which, _ = m.groups()
            _lin(jax_params, ["single_blocks_" + i, f"linear1_{which}"], pt_key, arr)
            continue
        m = _SINGLE_QK_NORM_RE.match(pt_key)
        if m:
            i, qk = m.groups()
            _rms(jax_params, ["single_blocks_" + i, f"{qk}_norm"], arr)
            continue
        m = _SINGLE_LIN2_RE.match(pt_key)
        if m:
            i, _ = m.groups()
            _lin(jax_params, ["single_blocks_" + i, "linear2"], pt_key, arr)
            continue
        m = _SINGLE_MOD_RE.match(pt_key)
        if m:
            i, _ = m.groups()
            _lin(jax_params, ["single_blocks_" + i, "modulation", "linear"], pt_key, arr)
            continue

        if pt_key in ("txt_in.input_embedder.weight", "txt_in.input_embedder.bias"):
            _lin(jax_params, ["txt_in", "input_embedder"], pt_key, arr)
            continue
        if pt_key in ("txt_in.t_embedder.mlp.0.weight", "txt_in.t_embedder.mlp.0.bias"):
            _lin(jax_params, ["txt_in", "t_embedder", "mlp_0"], pt_key, arr)
            continue
        if pt_key in ("txt_in.t_embedder.mlp.2.weight", "txt_in.t_embedder.mlp.2.bias"):
            _lin(jax_params, ["txt_in", "t_embedder", "mlp_2"], pt_key, arr)
            continue
        if pt_key in ("txt_in.c_embedder.linear_1.weight", "txt_in.c_embedder.linear_1.bias"):
            _lin(jax_params, ["txt_in", "c_embedder_linear_1"], pt_key, arr)
            continue
        if pt_key in ("txt_in.c_embedder.linear_2.weight", "txt_in.c_embedder.linear_2.bias"):
            _lin(jax_params, ["txt_in", "c_embedder_linear_2"], pt_key, arr)
            continue
        m = _REFINER_BLOCK_RE.match(pt_key)
        if m:
            i, norm, _ = m.groups()
            leaf = "bias" if pt_key.endswith(".bias") else "scale"  # affine LayerNorm, not Dense
            _set_nested_dict(jax_params, ["txt_in", f"blocks_{i}", norm, leaf],
                              convert_pt_tensor_to_jax(pt_key, arr))
            continue
        m = _REFINER_QKV_RE.match(pt_key)
        if m:
            i, _ = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "self_attn_qkv"], pt_key, arr)
            continue
        m = _REFINER_PROJ_RE.match(pt_key)
        if m:
            i, _ = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "self_attn_proj"], pt_key, arr)
            continue
        m = _REFINER_MLP_RE.match(pt_key)
        if m:
            i, fc, _ = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "mlp", fc], pt_key, arr)
            continue
        m = _REFINER_ADALN_RE.match(pt_key)
        if m:
            i, _ = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "adaLN_modulation_1"], pt_key, arr)
            continue

        if pt_key in ("time_in.mlp.0.weight", "time_in.mlp.0.bias"):
            _lin(jax_params, ["time_in", "mlp_0"], pt_key, arr)
            continue
        if pt_key in ("time_in.mlp.2.weight", "time_in.mlp.2.bias"):
            _lin(jax_params, ["time_in", "mlp_2"], pt_key, arr)
            continue

        if pt_key in ("vector_in.in_layer.weight", "vector_in.in_layer.bias"):
            _lin(jax_params, ["vector_in", "in_layer"], pt_key, arr)
            continue
        if pt_key in ("vector_in.out_layer.weight", "vector_in.out_layer.bias"):
            _lin(jax_params, ["vector_in", "out_layer"], pt_key, arr)
            continue
        if pt_key in ("guidance_in.mlp.0.weight", "guidance_in.mlp.0.bias"):
            _lin(jax_params, ["guidance_in", "mlp_0"], pt_key, arr)
            continue
        if pt_key in ("guidance_in.mlp.2.weight", "guidance_in.mlp.2.bias"):
            _lin(jax_params, ["guidance_in", "mlp_2"], pt_key, arr)
            continue

        m = re.match(r"^vision_in\.proj\.(\d+)\.(weight|bias)$", pt_key)
        if m:
            idx, _ = m.groups()
            sub_name = _VISION_SEQ_NAMES[idx]
            if sub_name.startswith("ln_"):
                scale_or_bias = "bias" if pt_key.endswith(".bias") else "scale"
                _set_nested_dict(jax_params, ["vision_in", sub_name, scale_or_bias],
                                  convert_pt_tensor_to_jax(pt_key, arr))
            else:
                _lin(jax_params, ["vision_in", sub_name], pt_key, arr)
            continue

        if pt_key.startswith("byt5_in."):
            leaf = pt_key.split(".", 1)[1]  # e.g. "layernorm.weight" -> "layernorm", "weight"
            name, _leaf_kind = leaf.rsplit(".", 1)
            if name == "layernorm":
                scale_or_bias = "bias" if pt_key.endswith(".bias") else "scale"
                _set_nested_dict(jax_params, ["byt5_in", "layernorm", scale_or_bias],
                                  convert_pt_tensor_to_jax(pt_key, arr))
            else:
                _lin(jax_params, ["byt5_in", name], pt_key, arr)
            continue

        if pt_key == "final_layer.linear.weight" or pt_key == "final_layer.linear.bias":
            _lin(jax_params, ["final_layer", "linear"], pt_key, arr)
            continue
        if pt_key in ("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias"):
            _lin(jax_params, ["final_layer", "adaLN_modulation_1"], pt_key, arr)
            continue

        if pt_key == "cond_type_embedding.weight":
            # nn.Embedding.weight is (num_embeddings, features) in both
            # PyTorch and Flax (`nn.Embed`) -- no transpose, unlike a Linear.
            _set_nested_dict(jax_params, ["cond_type_embedding", "embedding"], pt_tensor_to_numpy(arr))
            continue

        # Note: img_norm1/2, txt_norm1/2, pre_norm, norm_final are all
        # `elementwise_affine=False` in the reference, so they never emit a
        # state_dict entry at all -- no skip-clause needed for them here.
        raise KeyError(f"Unrecognized HunyuanVideo-1.5 DiT key: {pt_key}")

    return {"params": jax_params}


# ---------------------------------------------------------------------------
# VAE (AutoencoderKLConv3D): Encoder/Decoder built from CausalConv3d/RMS_norm/
# ResnetBlock/AttnBlock/Downsample/Upsample -- see
# vidax.models.hunyuan_video.hunyuan_video_1_5.vae for the Flax-side names
# each regex below targets.
# ---------------------------------------------------------------------------

_VAE_DOWN_BLOCK_RE = re.compile(r"^encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$")
_VAE_DOWN_DOWNSAMPLE_RE = re.compile(r"^encoder\.down\.(\d+)\.downsample\.conv\.conv\.(weight|bias)$")
_VAE_UP_BLOCK_RE = re.compile(r"^decoder\.up\.(\d+)\.block\.(\d+)\.(.+)$")
_VAE_UP_UPSAMPLE_RE = re.compile(r"^decoder\.up\.(\d+)\.upsample\.conv\.conv\.(weight|bias)$")


def _map_resnet_block_submodule(sub_key: str, jax_block_path: list, jax_params: dict, pt_key: str, arr) -> bool:
    """Handles one ResnetBlock's own sub-keys (``norm1``/``conv1``/``norm2``/
    ``conv2``/``nin_shortcut``), appended under ``jax_block_path``."""
    if sub_key == "norm1.gamma":
        _set_nested_dict(jax_params, jax_block_path + ["norm1", "gamma"],
                          convert_pt_tensor_to_jax(pt_key, arr).reshape(-1))
        return True
    if sub_key == "norm2.gamma":
        _set_nested_dict(jax_params, jax_block_path + ["norm2", "gamma"],
                          convert_pt_tensor_to_jax(pt_key, arr).reshape(-1))
        return True
    if sub_key.startswith("conv1.conv."):
        _lin(jax_params, jax_block_path + ["conv1"], pt_key, arr)
        return True
    if sub_key.startswith("conv2.conv."):
        _lin(jax_params, jax_block_path + ["conv2"], pt_key, arr)
        return True
    if sub_key.startswith("nin_shortcut."):
        _lin(jax_params, jax_block_path + ["nin_shortcut"], pt_key, arr)
        return True
    return False


def _map_attn_block_submodule(sub_key: str, jax_block_path: list, jax_params: dict, pt_key: str, arr) -> bool:
    if sub_key == "norm.gamma":
        _set_nested_dict(jax_params, jax_block_path + ["norm", "gamma"],
                          convert_pt_tensor_to_jax(pt_key, arr).reshape(-1))
        return True
    for name in ("q", "k", "v", "proj_out"):
        if sub_key.startswith(f"{name}."):
            _lin(jax_params, jax_block_path + [name], pt_key, arr)
            return True
    return False


def map_hunyuan_video_1_5_vae_keys(pt_state_dict: Dict) -> Dict:
    jax_params: Dict = {}

    for pt_key, arr in pt_state_dict.items():
        if pt_key in ("encoder.conv_in.conv.weight", "encoder.conv_in.conv.bias"):
            _lin(jax_params, ["encoder", "conv_in"], pt_key, arr)
            continue
        if pt_key in ("encoder.norm_out.gamma",):
            _set_nested_dict(jax_params, ["encoder", "norm_out", "gamma"],
                              convert_pt_tensor_to_jax(pt_key, arr).reshape(-1))
            continue
        if pt_key in ("encoder.conv_out.conv.weight", "encoder.conv_out.conv.bias"):
            _lin(jax_params, ["encoder", "conv_out"], pt_key, arr)
            continue
        if pt_key.startswith("encoder.mid.block_1."):
            if _map_resnet_block_submodule(pt_key[len("encoder.mid.block_1."):],
                                            ["encoder", "mid_block_1"], jax_params, pt_key, arr):
                continue
        if pt_key.startswith("encoder.mid.block_2."):
            if _map_resnet_block_submodule(pt_key[len("encoder.mid.block_2."):],
                                            ["encoder", "mid_block_2"], jax_params, pt_key, arr):
                continue
        if pt_key.startswith("encoder.mid.attn_1."):
            if _map_attn_block_submodule(pt_key[len("encoder.mid.attn_1."):],
                                          ["encoder", "mid_attn_1"], jax_params, pt_key, arr):
                continue
        m = _VAE_DOWN_DOWNSAMPLE_RE.match(pt_key)
        if m:
            level, _ = m.groups()
            _lin(jax_params, ["encoder", f"down_{level}_downsample", "conv"], pt_key, arr)
            continue
        m = _VAE_DOWN_BLOCK_RE.match(pt_key)
        if m:
            level, block, sub_key = m.groups()
            path = ["encoder", f"down_{level}_block_{block}"]
            if _map_resnet_block_submodule(sub_key, path, jax_params, pt_key, arr):
                continue

        if pt_key in ("decoder.conv_in.conv.weight", "decoder.conv_in.conv.bias"):
            _lin(jax_params, ["decoder", "conv_in"], pt_key, arr)
            continue
        if pt_key == "decoder.norm_out.gamma":
            _set_nested_dict(jax_params, ["decoder", "norm_out", "gamma"],
                              convert_pt_tensor_to_jax(pt_key, arr).reshape(-1))
            continue
        if pt_key in ("decoder.conv_out.conv.weight", "decoder.conv_out.conv.bias"):
            _lin(jax_params, ["decoder", "conv_out"], pt_key, arr)
            continue
        if pt_key.startswith("decoder.mid.block_1."):
            if _map_resnet_block_submodule(pt_key[len("decoder.mid.block_1."):],
                                            ["decoder", "mid_block_1"], jax_params, pt_key, arr):
                continue
        if pt_key.startswith("decoder.mid.block_2."):
            if _map_resnet_block_submodule(pt_key[len("decoder.mid.block_2."):],
                                            ["decoder", "mid_block_2"], jax_params, pt_key, arr):
                continue
        if pt_key.startswith("decoder.mid.attn_1."):
            if _map_attn_block_submodule(pt_key[len("decoder.mid.attn_1."):],
                                          ["decoder", "mid_attn_1"], jax_params, pt_key, arr):
                continue
        m = _VAE_UP_UPSAMPLE_RE.match(pt_key)
        if m:
            level, _ = m.groups()
            _lin(jax_params, ["decoder", f"up_{level}_upsample", "conv"], pt_key, arr)
            continue
        m = _VAE_UP_BLOCK_RE.match(pt_key)
        if m:
            level, block, sub_key = m.groups()
            path = ["decoder", f"up_{level}_block_{block}"]
            if _map_resnet_block_submodule(sub_key, path, jax_params, pt_key, arr):
                continue

        raise KeyError(f"Unrecognized HunyuanVideo-1.5 VAE key: {pt_key}")

    return {"params": jax_params}


# ---------------------------------------------------------------------------
# byT5 (Glyph-SDXL-v2 on google/byt5-small): a bare `T5Stack` encoder
# (`T5ForConditionalGeneration(...).get_encoder()`, no `encoder.` wrapper
# prefix -- unlike ltx_video's full `T5EncoderModel` checkpoint, see
# `mappings/ltx_video.py:map_ltx_video_t5_keys`, whose `encoder.`-prefixed
# pattern this closely mirrors minus that prefix). Target: `vidax.models.
# ltx_video.t5.T5Encoder` (reused directly, see hunyuan_video_1_5/byt5.py).
# Checkpoint's own `module.text_tower.encoder.` prefix must already be
# stripped by the caller (matching the reference's own `create_byt5`
# loader) before calling this.
# ---------------------------------------------------------------------------

_BYT5_BLOCK_RE = re.compile(r"^block\.(\d+)\.layer\.(0|1)\.(.*)$")


def map_hunyuan_video_1_5_siglip_keys(pt_state_dict: Dict) -> Dict:
    """Translates a real `SiglipVisionModel` state_dict (``black-forest-labs/
    FLUX.1-Redux-dev``'s ``image_encoder`` subfolder -- gated, not yet
    downloaded/verified against, see ``hunyuan_video_1_5/siglip.py``'s
    module docstring) into a Flax param tree for ``SiglipVisionEncoder``.
    Ignores the (unused-by-HunyuanVideo-1.5) ``vision_model.head.*``
    pooling-head keys, since ``SiglipVisionEncoder`` doesn't port them.
    """
    jax_params: Dict = {}
    prefix = "vision_model."

    for pt_key, arr in pt_state_dict.items():
        if not pt_key.startswith(prefix):
            continue  # e.g. a stray top-level key from a different wrapper
        key = pt_key[len(prefix):]

        if key.startswith("head."):
            continue  # SiglipMultiheadAttentionPoolingHead -- not ported, unused

        if key == "embeddings.patch_embedding.weight":
            _set_nested_dict(jax_params, ["embeddings", "patch_embedding", "kernel"],
                              convert_pt_tensor_to_jax(pt_key, arr))
            continue
        if key == "embeddings.patch_embedding.bias":
            _set_nested_dict(jax_params, ["embeddings", "patch_embedding", "bias"], pt_tensor_to_numpy(arr))
            continue
        if key == "embeddings.position_embedding.weight":
            _set_nested_dict(jax_params, ["embeddings", "position_embedding", "embedding"], pt_tensor_to_numpy(arr))
            continue
        if key == "post_layernorm.weight":
            _set_nested_dict(jax_params, ["post_layernorm", "scale"], pt_tensor_to_numpy(arr))
            continue
        if key == "post_layernorm.bias":
            _set_nested_dict(jax_params, ["post_layernorm", "bias"], pt_tensor_to_numpy(arr))
            continue

        m = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", key)
        if m:
            i, sub_key = m.groups()
            path = [f"layers_{i}"]
            if sub_key in ("layer_norm1.weight", "layer_norm1.bias"):
                _set_nested_dict(jax_params, path + ["layer_norm1", "bias" if sub_key.endswith("bias") else "scale"],
                                  pt_tensor_to_numpy(arr))
                continue
            if sub_key in ("layer_norm2.weight", "layer_norm2.bias"):
                _set_nested_dict(jax_params, path + ["layer_norm2", "bias" if sub_key.endswith("bias") else "scale"],
                                  pt_tensor_to_numpy(arr))
                continue
            m2 = re.match(r"^self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$", sub_key)
            if m2:
                proj, _ = m2.groups()
                _lin(jax_params, path + ["self_attn", proj], pt_key, arr)
                continue
            m3 = re.match(r"^mlp\.(fc1|fc2)\.(weight|bias)$", sub_key)
            if m3:
                fc, _ = m3.groups()
                _lin(jax_params, path + ["mlp", fc], pt_key, arr)
                continue

        raise KeyError(f"Unrecognized HunyuanVideo-1.5 SigLIP key: {pt_key}")

    return {"params": jax_params}


def map_hunyuan_video_1_5_byt5_keys(pt_state_dict: Dict) -> Dict:
    jax_params: Dict = {}

    for pt_key, arr in pt_state_dict.items():
        if pt_key == "embed_tokens.weight":
            _set_nested_dict(jax_params, ["token_embedding", "embedding"], pt_tensor_to_numpy(arr))
            continue
        if pt_key == "final_layer_norm.weight":
            _set_nested_dict(jax_params, ["norm", "scale"], pt_tensor_to_numpy(arr))
            continue
        if pt_key == "block.0.layer.0.SelfAttention.relative_attention_bias.weight":
            _set_nested_dict(jax_params, ["relative_attention_bias", "embedding"], pt_tensor_to_numpy(arr))
            continue
        m = _BYT5_BLOCK_RE.match(pt_key)
        if m:
            block_idx, layer_idx, sub_key = m.groups()
            block_path = [f"blocks_{block_idx}"]
            if sub_key == "layer_norm.weight":
                norm_name = "norm1" if layer_idx == "0" else "norm2"
                _set_nested_dict(jax_params, block_path + [norm_name, "scale"], pt_tensor_to_numpy(arr))
                continue
            if layer_idx == "0" and sub_key.startswith("SelfAttention."):
                attn_sub = sub_key[len("SelfAttention."):]
                m2 = re.match(r"(q|k|v|o)\.weight$", attn_sub)
                if m2:
                    _set_nested_dict(jax_params, block_path + [f"attn_{m2.group(1)}", "kernel"],
                                      convert_pt_tensor_to_jax(pt_key, arr))
                    continue
            if layer_idx == "1" and sub_key.startswith("DenseReluDense."):
                ffn_sub = sub_key[len("DenseReluDense."):]
                ffn_name = {"wi_0.weight": "ffn_gate_0", "wi_1.weight": "ffn_fc1", "wo.weight": "ffn_fc2"}.get(ffn_sub)
                if ffn_name:
                    _set_nested_dict(jax_params, block_path + [ffn_name, "kernel"],
                                      convert_pt_tensor_to_jax(pt_key, arr))
                    continue

        raise KeyError(f"Unrecognized HunyuanVideo-1.5 byT5 key: {pt_key}")

    return {"params": jax_params}


# Note: Glyph-SDXL-v2/checkpoints/byt5_mapper.pt is *not* the DiT's
# `byt5_in` (`ByT5Mapper`) -- checked directly (its keys are a small
# 4-block T5-style tower + `channel_mapper`/`final_layer_norm`, nothing
# resembling `layernorm`/`fc1`/`fc2`/`fc3`). It's a Glyph-SDXL-v2-internal
# artifact for that project's own SDXL LoRA glyph pipeline, unrelated to
# HunyuanVideo-1.5. The real `byt5_in.*` weights the DiT actually uses live
# inside the main DiT checkpoint itself (`transformer/<variant>/
# diffusion_pytorch_model.safetensors`, `byt5_in.{layernorm,fc1,fc2,fc3}.*`
# keys -- already handled by `map_hunyuan_video_1_5_dit_keys` above and
# confirmed present against the real checkpoint). No separate mapper/loader
# needed for it.
