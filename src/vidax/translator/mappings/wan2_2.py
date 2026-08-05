"""PyTorch state_dict -> Flax parameter tree key mappings specific to Wan2.2.

Wan2.2's DiT reuses `vidax.translator.mappings.common.map_wan_dit_keys`
unchanged (see that function's docstring for why); only the VAE needs its
own mapper here, since `Encoder3d`/`Decoder3d` wrap each resolution stage in
a `Down_ResidualBlock`/`Up_ResidualBlock` (an extra level of nn.Sequential
nesting Wan2.1 doesn't have: PT paths look like
`encoder.downsamples.{i}.downsamples.{j}.*` instead of Wan2.1's flat
`encoder.downsamples.{j}.*`), and `avg_shortcut`/`AvgDown3D`/`DupUp3D` have
no learnable parameters, so no keys for them ever appear.
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax
from .common import _leaf_name, _set_nested_dict, map_vae_block_submodule


def _map_vae2_2_tower(
    pt_key: str, sub_key: str, jax_tensor, jax_params: dict,
    tower_name: str, stage_key: str,
) -> bool:
    """Handles keys inside `encoder.*` or `decoder.*`.

    `stage_key` is "downsamples" for the encoder, "upsamples" for the
    decoder -- the reference's per-stage attribute name, which (unlike
    Wan2.1) wraps a whole `Down_ResidualBlock`/`Up_ResidualBlock`, hence the
    extra `.{j}` level below it.
    """
    if sub_key.startswith("conv1."):
        _set_nested_dict(jax_params, [tower_name, "conv1", _leaf_name(sub_key)], jax_tensor)
        return True

    match = re.match(r"middle\.(\d+)\.(.*)", sub_key)
    if match:
        block_path = [tower_name, f"middle_{match.group(1)}"]
        return map_vae_block_submodule(match.group(2), block_path, jax_tensor, jax_params)

    match = re.match(rf"{stage_key}\.(\d+)\.{stage_key}\.(\d+)\.(.*)", sub_key)
    if match:
        stage_idx, inner_idx, field = match.groups()
        block_path = [tower_name, f"{stage_key}_{stage_idx}", f"{stage_key}_{inner_idx}"]
        return map_vae_block_submodule(field, block_path, jax_tensor, jax_params)

    match = re.match(r"head\.(0|2)\.(gamma|weight|bias)$", sub_key)
    if match:
        idx, field = match.groups()
        leaf = "scale" if field == "gamma" else _leaf_name(sub_key)
        _set_nested_dict(jax_params, [tower_name, f"head_{idx}", leaf], jax_tensor)
        return True

    return False


def map_wan2_2_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Wan2.2 `WanVAE_` state_dict into a Flax param tree for
    `vidax.models.wan.wan2_2.vae.WanVAEDecoder`/`WanVAEEncoder`.

    Both towers' weights are mapped from the same checkpoint, same as
    Wan2.1's `map_wan2_1_vae_keys` -- see that function's docstring.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key.startswith("conv2."):
            _set_nested_dict(jax_params, ["conv2", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("conv1."):
            _set_nested_dict(jax_params, ["conv1", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("decoder."):
            _map_vae2_2_tower(pt_key, pt_key[len("decoder."):], jax_tensor, jax_params,
                               "decoder", "upsamples")
        elif pt_key.startswith("encoder."):
            _map_vae2_2_tower(pt_key, pt_key[len("encoder."):], jax_tensor, jax_params,
                               "encoder", "downsamples")

    return {"params": jax_params}
