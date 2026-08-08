"""PyTorch state_dict -> Flax parameter tree mapping for Wan2.2's VAE as
shipped by `diffusers` (`AutoencoderKLWan`, e.g. under Cosmos3-Nano's
`vae/diffusion_pytorch_model.safetensors`) -- a different checkpoint *key
layout* than `vidax.translator.mappings.wan2_2.map_wan2_2_vae_keys` handles
(the original Wan repo's own release format), for the *same* underlying
architecture (confirmed directly: `vae/config.json`'s `base_dim=160`,
`decoder_base_dim=256`, `z_dim=48`, `dim_mult=[1,2,4,4]` match
`vidax.models.wan.wan2_2.vae`'s defaults exactly).

Differences from the original-repo layout, confirmed against the checkpoint's
actual keys (not just diffusers' source):
  - Each resolution stage is `{encoder,decoder}.{down,up}_blocks.{i}` with
    its resnets directly at `.resnets.{j}.*` (not the original repo's
    doubly-nested `downsamples.{i}.downsamples.{j}.*` naming -- the *nesting
    depth* in our own Flax port still matches the original repo, since our
    port was written against that layout; only this checkpoint's *key
    strings* differ).
  - Each stage's `Resample` submodule is a separate `.downsampler.`/
    `.upsampler.` child (`resample.1.*`, `time_conv.*` beneath it) rather
    than being the last element of the resnets index range.
  - Resnet submodule names are already `conv1`/`conv2`/`norm1`/`norm2`/
    `conv_shortcut` (our port's own names, coincidentally, apart from
    `conv_shortcut` -> `shortcut`) rather than the original repo's
    `residual.{0,2,3,6}` `nn.Sequential` indices.
  - `quant_conv`/`post_quant_conv` (both trivial `z_dim*2 -> z_dim*2` /
    `z_dim -> z_dim` 1x1 convs) replace the original repo's fused `conv1`/
    `conv2` naming for the same role -- our port already keeps these as
    separate top-level `conv1`/`conv2` submodules (see
    `WanVAEEncoder`/`WanVAEDecoder`'s `setup()`), so this is a rename, not a
    structural difference.
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax
from .common import _leaf_name, _set_nested_dict


def _map_diffusers_vae_block_submodule(sub_key: str, block_path: list, jax_tensor, jax_params: dict) -> bool:
    """Handles keys inside one diffusers-format ResnetBlock or Attention."""
    if sub_key in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
        name = sub_key.split(".")[0]
        _set_nested_dict(jax_params, block_path + [name, _leaf_name(sub_key)], jax_tensor)
        return True
    if sub_key in ("norm1.gamma", "norm2.gamma"):
        name = sub_key.split(".")[0]
        _set_nested_dict(jax_params, block_path + [name, "scale"], jax_tensor)
        return True
    if sub_key.startswith("conv_shortcut."):
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


def _map_diffusers_vae_tower(
    sub_key: str, jax_tensor, jax_params: dict, tower_name: str, stage_key: str, num_res_blocks: int,
) -> bool:
    """Handles keys inside `encoder.*`/`decoder.*`.

    `stage_key` is "down_blocks"/"downsampler" for the encoder,
    "up_blocks"/"upsampler" for the decoder. `num_res_blocks` is this tower's
    per-stage resnet count (`num_res_blocks` for the encoder,
    `num_res_blocks + 1` for the decoder -- matching `Encoder3d`/`Decoder3d`'s
    own construction) -- the diffusers `Resample` submodule's position in
    our port's flat `{downsamples,upsamples}_{i}` numbering is
    `num_res_blocks` (it comes right after that many resnets).
    """
    blocks_key, resample_key, flat_prefix = stage_key

    if sub_key.startswith("conv_in."):
        _set_nested_dict(jax_params, [tower_name, "conv1", _leaf_name(sub_key)], jax_tensor)
        return True
    if sub_key.startswith("conv_out."):
        _set_nested_dict(jax_params, [tower_name, "head_2", _leaf_name(sub_key)], jax_tensor)
        return True
    if sub_key.startswith("norm_out."):
        _set_nested_dict(jax_params, [tower_name, "head_0", "scale"], jax_tensor)
        return True

    match = re.match(r"mid_block\.resnets\.(\d+)\.(.*)", sub_key)
    if match:
        # middle_0 / middle_2 -- middle_1 is the (separately-matched) attention block.
        idx, field = match.groups()
        block_path = [tower_name, f"middle_{int(idx) * 2}"]
        return _map_diffusers_vae_block_submodule(field, block_path, jax_tensor, jax_params)
    match = re.match(r"mid_block\.attentions\.0\.(.*)", sub_key)
    if match:
        return _map_diffusers_vae_block_submodule(match.group(1), [tower_name, "middle_1"], jax_tensor, jax_params)

    match = re.match(rf"{blocks_key}\.(\d+)\.resnets\.(\d+)\.(.*)", sub_key)
    if match:
        stage_idx, inner_idx, field = match.groups()
        block_path = [tower_name, f"{flat_prefix}_{stage_idx}", f"{flat_prefix}_{inner_idx}"]
        return _map_diffusers_vae_block_submodule(field, block_path, jax_tensor, jax_params)

    match = re.match(rf"{blocks_key}\.(\d+)\.{resample_key}\.(.*)", sub_key)
    if match:
        stage_idx, field = match.groups()
        block_path = [tower_name, f"{flat_prefix}_{stage_idx}", f"{flat_prefix}_{num_res_blocks}"]
        return _map_diffusers_vae_block_submodule(field, block_path, jax_tensor, jax_params)

    return False


def map_wan2_2_vae_diffusers_keys(pt_state_dict: Dict) -> Dict:
    """Translates a diffusers-format `AutoencoderKLWan` state_dict into a
    Flax param tree for `vidax.models.wan.wan2_2.vae.WanVAEDecoder`/`WanVAEEncoder`.
    """
    jax_params: Dict[str, Any] = {}
    # (blocks_key, resample_key, flat-name-prefix) per tower.
    encoder_keys = ("down_blocks", "downsampler", "downsamples")
    decoder_keys = ("up_blocks", "upsampler", "upsamples")

    for pt_key, pt_tensor in pt_state_dict.items():
        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key.startswith("quant_conv."):
            _set_nested_dict(jax_params, ["conv1", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("post_quant_conv."):
            _set_nested_dict(jax_params, ["conv2", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("encoder."):
            _map_diffusers_vae_tower(
                pt_key[len("encoder."):], jax_tensor, jax_params, "encoder", encoder_keys,
                num_res_blocks=2)
        elif pt_key.startswith("decoder."):
            _map_diffusers_vae_tower(
                pt_key[len("decoder."):], jax_tensor, jax_params, "decoder", decoder_keys,
                num_res_blocks=3)

    return {"params": jax_params}
