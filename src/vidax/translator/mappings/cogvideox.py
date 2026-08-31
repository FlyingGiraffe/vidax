"""PyTorch state_dict -> Flax parameter tree mapping for CogVideoX.

Two entry points, one per model_type in
`vidax.translator.mappings.load_torch_checkpoint_to_jax`:

* `map_cogvideox_dit_keys`  -- diffusers `CogVideoXTransformer3DModel`
  (`transformer/diffusion_pytorch_model*.safetensors`) ->
  `vidax.models.cogvideo.dit.CogVideoXDiT`.
* `map_cogvideox_vae_keys`  -- diffusers `AutoencoderKLCogVideoX`
  (`vae/diffusion_pytorch_model.safetensors`) ->
  `vidax.models.cogvideo.vae.CogVideoXVAE`.

The T5 text encoder is a plain `T5EncoderModel` identical to LTX-Video's
PixArt encoder -- load it with `model_type="ltx_video_t5"`
(`map_ltx_video_t5_keys`), there is nothing CogVideoX-specific there.

Style follows `mappings/ltx_video.py`: iterate the state_dict, run each
tensor through `convert_pt_tensor_to_jax` (which already applies the
Conv2d/Conv3d/Linear layout transposes), and `_set_nested_dict` into the
Flax tree. LayerNorm weight/bias and 1-D norm scales need no transpose;
`patch_embed.pos_embedding` is a `(1, N, dim)` buffer kept verbatim.
"""
import re
from typing import Any, Dict

from .common import _leaf_name, _set_nested_dict
from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy


# --------------------------------------------------------------------------
# DiT
# --------------------------------------------------------------------------

def _dense(jax_params, path, pt_key, tensor):
    _set_nested_dict(jax_params, path + [_leaf_name(pt_key)], tensor)


def map_cogvideox_dit_keys(pt_state_dict: Dict) -> Dict:
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        # `patch_embed.pos_embedding` is a positional-embedding buffer
        # (shape (1, 226 + num_patches, inner_dim)) -- not a Linear weight,
        # never transposed.
        if pt_key == "patch_embed.pos_embedding":
            _set_nested_dict(jax_params, ["pos_embedding"], pt_tensor_to_numpy(pt_tensor))
            continue

        t = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key.startswith("patch_embed.text_proj."):
            _dense(jax_params, ["patch_embed", "text_proj"], pt_key, t)
        elif pt_key.startswith("patch_embed.proj."):
            _dense(jax_params, ["patch_embed", "proj"], pt_key, t)
        elif pt_key.startswith("time_embedding.linear_1."):
            _dense(jax_params, ["time_embedding_linear_1"], pt_key, t)
        elif pt_key.startswith("time_embedding.linear_2."):
            _dense(jax_params, ["time_embedding_linear_2"], pt_key, t)
        elif pt_key.startswith("ofs_embedding.linear_1."):
            _dense(jax_params, ["ofs_embedding_linear_1"], pt_key, t)
        elif pt_key.startswith("ofs_embedding.linear_2."):
            _dense(jax_params, ["ofs_embedding_linear_2"], pt_key, t)
        elif pt_key == "norm_final.weight":
            _set_nested_dict(jax_params, ["norm_final_scale"], t)
        elif pt_key == "norm_final.bias":
            _set_nested_dict(jax_params, ["norm_final_bias"], t)
        elif pt_key.startswith("norm_out.linear."):
            _dense(jax_params, ["norm_out_linear"], pt_key, t)
        elif pt_key == "norm_out.norm.weight":
            _set_nested_dict(jax_params, ["norm_out_norm_scale"], t)
        elif pt_key == "norm_out.norm.bias":
            _set_nested_dict(jax_params, ["norm_out_norm_bias"], t)
        elif pt_key.startswith("proj_out."):
            _dense(jax_params, ["proj_out"], pt_key, t)
        elif pt_key.startswith("transformer_blocks."):
            m = re.match(r"transformer_blocks\.(\d+)\.(.*)", pt_key)
            bi, sub = m.group(1), m.group(2)
            bp = [f"transformer_blocks_{bi}"]

            if m2 := re.match(r"(norm1|norm2)\.linear\.(weight|bias)$", sub):
                _dense(jax_params, bp + [m2.group(1), "linear"], pt_key, t)
            elif m2 := re.match(r"(norm1|norm2)\.norm\.(weight|bias)$", sub):
                leaf = "norm_scale" if m2.group(2) == "weight" else "norm_bias"
                _set_nested_dict(jax_params, bp + [m2.group(1), leaf], t)
            elif m2 := re.match(r"attn1\.(to_q|to_k|to_v)\.(weight|bias)$", sub):
                _dense(jax_params, bp + ["attn1", m2.group(1)], pt_key, t)
            elif m2 := re.match(r"attn1\.to_out\.0\.(weight|bias)$", sub):
                _dense(jax_params, bp + ["attn1", "to_out_0"], pt_key, t)
            elif m2 := re.match(r"attn1\.(norm_q|norm_k)\.(weight|bias)$", sub):
                leaf = f"{m2.group(1)}_scale" if m2.group(2) == "weight" else f"{m2.group(1)}_bias"
                _set_nested_dict(jax_params, bp + ["attn1", leaf], t)
            elif sub.startswith("ff.net.0.proj."):
                _dense(jax_params, bp + ["ff", "net_0_proj"], pt_key, t)
            elif sub.startswith("ff.net.2."):
                _dense(jax_params, bp + ["ff", "net_2"], pt_key, t)
            else:
                raise KeyError(f"Unmapped CogVideoX DiT block key: {pt_key}")
        else:
            raise KeyError(f"Unmapped CogVideoX DiT key: {pt_key}")

    return {"params": jax_params}


# --------------------------------------------------------------------------
# VAE
# --------------------------------------------------------------------------

_VAE_INDEX_SUBST = (
    ("resnets.", "resnets_"),
    ("down_blocks.", "down_blocks_"),
    ("up_blocks.", "up_blocks_"),
    ("downsamplers.", "downsamplers_"),
    ("upsamplers.", "upsamplers_"),
)

_GROUPNORM_LEAVES = {"norm1", "norm2", "norm_out", "norm_layer"}


def _vae_flax_path(pt_key: str) -> list:
    """diffusers `AutoencoderKLCogVideoX` key -> `CogVideoXVAE` Flax path.

    Near-mechanical: `resnets.0` -> `resnets_0` etc.; a trailing `conv.weight`
    / `conv_shortcut.weight` becomes `.../kernel` (Flax `nn.Conv`); a trailing
    `<groupnorm>.weight` gains a `"norm"` level and `weight->scale`
    (the `GroupNorm` wrapper module in `vae.py`).
    """
    k = pt_key
    for a, b in _VAE_INDEX_SUBST:
        k = k.replace(a, b)
    toks = k.split(".")
    leaf = toks[-1]  # weight | bias
    parent = toks[-2]

    if parent in ("conv", "conv_shortcut"):
        return toks[:-1] + ["kernel" if leaf == "weight" else "bias"]
    if parent in _GROUPNORM_LEAVES:
        return toks[:-1] + ["norm", "scale" if leaf == "weight" else "bias"]
    raise KeyError(f"Unmapped CogVideoX VAE key: {pt_key}")


def map_cogvideox_vae_keys(pt_state_dict: Dict) -> Dict:
    jax_params: Dict[str, Any] = {}
    for pt_key, pt_tensor in pt_state_dict.items():
        if not (pt_key.startswith("encoder.") or pt_key.startswith("decoder.")):
            raise KeyError(f"Unexpected CogVideoX VAE key: {pt_key}")
        t = convert_pt_tensor_to_jax(pt_key, pt_tensor)
        _set_nested_dict(jax_params, _vae_flax_path(pt_key), t)
    return {"params": jax_params}
