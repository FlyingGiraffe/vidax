"""PyTorch checkpoint -> Flax parameter tree loading, dispatched by model type.

One module per model version (`wan2_1`, `wan2_2`, ...) holds that version's
key-mapping functions; `common` holds helpers/mappers that are provably
identical across versions (see its module docstring). This `__init__` is the
single place that knows the full model_type -> mapping-function table, so
adding a new version only means adding new `elif` branches here plus a new
sibling module -- callers just keep using `load_torch_checkpoint_to_jax`.
"""
import json
import os
from typing import Dict

import safetensors.numpy

from .common import map_wan_dit_keys, map_wan_t5_keys
from .cogvideox import map_cogvideox_dit_keys, map_cogvideox_vae_keys
from .cosmos2_5 import map_cosmos2_5_dit_keys
from .cosmos3 import map_cosmos3_dit_keys
from .ltx_video import map_ltx_video_dit_keys, map_ltx_video_t5_keys, map_ltx_video_vae_keys
from .ltx2_5 import (
    map_gemma4_text_keys,
    map_ltx2_5_connector_keys,
    map_ltx2_5_dit_keys,
    map_ltx2_5_diffusion_decoder_keys,
    map_ltx2_5_vae_keys,
)
from .reason1 import map_reason1_text_encoder_keys
from .hunyuan_video1_5 import (
    map_hunyuan_video1_5_byt5_keys,
    map_hunyuan_video1_5_dit_keys,
    map_hunyuan_video1_5_siglip_keys,
    map_hunyuan_video1_5_vae_keys,
)
from .hunyuan_video import (
    map_hunyuan_video_clip_text_keys,
    map_hunyuan_video_clip_vision_keys,
    map_hunyuan_video_dit_keys,
    map_hunyuan_video_llama_text_keys,
    map_hunyuan_video_llava_llama_text_keys,
    map_hunyuan_video_llava_projector_keys,
    map_hunyuan_video_vae_keys,
)
from .wan2_1 import map_wan2_1_clip_keys, map_wan2_1_dit_keys, map_wan2_1_vae_keys
from .wan2_2 import map_wan2_2_vae_keys
from .wan2_2_diffusers import map_wan2_2_vae_diffusers_keys

__all__ = [
    "load_torch_checkpoint_to_jax",
    "map_wan_dit_keys",
    "map_wan_t5_keys",
    "map_wan2_1_dit_keys",
    "map_wan2_1_vae_keys",
    "map_wan2_1_clip_keys",
    "map_wan2_2_vae_keys",
    "map_cosmos2_5_dit_keys",
    "map_cogvideox_dit_keys",
    "map_cogvideox_vae_keys",
    "map_reason1_text_encoder_keys",
    "map_hunyuan_video1_5_dit_keys",
    "map_hunyuan_video1_5_vae_keys",
    "map_hunyuan_video1_5_byt5_keys",
    "map_hunyuan_video1_5_siglip_keys",
    "map_hunyuan_video_dit_keys",
    "map_hunyuan_video_vae_keys",
    "map_hunyuan_video_llama_text_keys",
    "map_hunyuan_video_clip_text_keys",
    "map_hunyuan_video_clip_vision_keys",
    "map_hunyuan_video_llava_llama_text_keys",
    "map_hunyuan_video_llava_projector_keys",
    "map_cosmos3_dit_keys",
    "map_wan2_2_vae_diffusers_keys",
    "map_ltx_video_dit_keys",
    "map_ltx_video_vae_keys",
    "map_ltx_video_t5_keys",
    "map_ltx2_5_dit_keys",
    "map_ltx2_5_connector_keys",
    "map_ltx2_5_vae_keys",
    "map_ltx2_5_diffusion_decoder_keys",
    "map_gemma4_text_keys",
]


def _load_pt_state_dict(checkpoint_path: str) -> Dict:
    """Loads a PyTorch state_dict from a `.safetensors`/`.safetensors.index.json`
    or a `.pth`/`.pt` checkpoint (the released Wan checkpoints ship the DiT
    as `.safetensors` -- sharded across multiple files with a
    `.safetensors.index.json` manifest for the larger models (e.g. Wan2.2's
    5B/14B DiTs), single-file for the smaller ones -- but the VAE and T5 text
    encoder as raw `.pth`).
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".json":
        # A `*.safetensors.index.json` manifest: load every shard it
        # references (paths relative to the manifest's own directory) and
        # merge into one state_dict.
        with open(checkpoint_path) as f:
            index = json.load(f)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        state_dict = {}
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(safetensors.numpy.load_file(os.path.join(checkpoint_dir, shard_file)))
        return state_dict
    elif ext == ".safetensors":
        return safetensors.numpy.load_file(checkpoint_path)
    elif ext in (".pth", ".pt", ".bin"):
        import io
        import torch  # Optional dependency; install the `torch` extra.
        # Cosmos-Predict2.5 checkpoints stash TransformerEngine bookkeeping
        # blobs in `*._extra_state` as raw `io.BytesIO` objects (skipped
        # entirely by `cosmos2_5.map_cosmos2_5_dit_keys` -- see its module
        # docstring) -- `weights_only=True`'s unpickler rejects that type by
        # default. Allow-listing it is safe (it holds opaque bytes, not
        # executable state) without falling back to `weights_only=False`
        # (unrestricted unpickling) for the whole checkpoint.
        with torch.serialization.safe_globals([io.BytesIO]):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # `io.BytesIO`'s own `.detach()` (inherited from `IOBase`, meant for
        # layered streams) raises `UnsupportedOperation` if called -- an
        # unrelated method that happens to share the tensor API's name, not
        # a tensor-like `.detach()`. Only real tensors need detaching.
        return {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }
    else:
        raise ValueError(f"Unrecognized checkpoint extension: '{ext}' ({checkpoint_path})")


def load_torch_checkpoint_to_jax(checkpoint_path: str, model_type: str = "wan2.1_dit") -> Dict:
    """Loads a Wan PyTorch checkpoint into a JAX/Flax parameter dict.

    Args:
        checkpoint_path: Path to the state_dict (`.safetensors`, `.pth`, or `.pt`).
        model_type: One of "wan_dit" (WanDiT, either Wan2.1 or Wan2.2 -- the
            two versions' DiT state_dicts use identical key names, see
            `map_wan_dit_keys`), "wan2.1_vae" (WanVAEDecoder/WanVAEEncoder),
            "wan2.1_clip" (ClipVisionTransformer), "wan2.2_vae"
            (Wan2.2's WanVAEDecoder/WanVAEEncoder), or "wan_t5" (T5Encoder --
            shared, byte-identical checkpoint format across Wan2.1 and Wan2.2).
            "wan2.1_dit" is kept as a backward-compatible alias for "wan_dit".
    """
    pt_state_dict = _load_pt_state_dict(checkpoint_path)

    if model_type in ("wan_dit", "wan2.1_dit", "wan2.2_dit"):
        return map_wan_dit_keys(pt_state_dict)
    elif model_type == "cosmos2.5_dit":
        return map_cosmos2_5_dit_keys(pt_state_dict)
    elif model_type == "cogvideox_dit":
        return map_cogvideox_dit_keys(pt_state_dict)
    elif model_type == "cogvideox_vae":
        return map_cogvideox_vae_keys(pt_state_dict)
    elif model_type == "wan2.1_vae":
        return map_wan2_1_vae_keys(pt_state_dict)
    elif model_type == "wan2.1_clip":
        return map_wan2_1_clip_keys(pt_state_dict)
    elif model_type == "wan2.2_vae":
        return map_wan2_2_vae_keys(pt_state_dict)
    elif model_type == "wan2.2_vae_diffusers":
        return map_wan2_2_vae_diffusers_keys(pt_state_dict)
    elif model_type == "wan_t5":
        return map_wan_t5_keys(pt_state_dict)
    elif model_type == "cosmos3_dit":
        return map_cosmos3_dit_keys(pt_state_dict)
    elif model_type == "ltx_video_dit":
        return map_ltx_video_dit_keys(pt_state_dict)
    elif model_type == "ltx_video_vae":
        return map_ltx_video_vae_keys(pt_state_dict)
    elif model_type == "ltx_video_t5":
        return map_ltx_video_t5_keys(pt_state_dict)
    elif model_type == "ltx2_5_dit":
        return map_ltx2_5_dit_keys(pt_state_dict)
    elif model_type == "ltx2_5_connector":
        return map_ltx2_5_connector_keys(pt_state_dict)
    elif model_type == "ltx2_5_vae":
        return map_ltx2_5_vae_keys(pt_state_dict)
    elif model_type == "ltx2_5_diffusion_decoder":
        return map_ltx2_5_diffusion_decoder_keys(pt_state_dict)
    elif model_type == "gemma4_text":
        return map_gemma4_text_keys(pt_state_dict)
    elif model_type == "reason1_text_encoder":
        # Cosmos-Predict2.5-2B's text encoder (Reason1-finetuned
        # Qwen2.5-VL-7B-Instruct, text tower only -- see
        # vidax.models.cosmos2_5.reason1). Ships as its own separate
        # HuggingFace-format repo (nvidia/Cosmos-Reason1-7B), sharded
        # `model-NNNNN-of-NNNNN.safetensors` + a `model.safetensors.index.json`
        # manifest -- pass the `.json` manifest's path as `checkpoint_path`
        # (handled by the ordinary `.json`-sharded branch above, same as
        # Wan2.2's DiT). Confirmed against the real checkpoint: exact 1:1
        # parameter-tree match against `Qwen2TextModel`'s init'd params.
        return map_reason1_text_encoder_keys(pt_state_dict)
    elif model_type == "hunyuan_video1_5_dit":
        return map_hunyuan_video1_5_dit_keys(pt_state_dict)
    elif model_type == "hunyuan_video1_5_vae":
        return map_hunyuan_video1_5_vae_keys(pt_state_dict)
    elif model_type == "hunyuan_video1_5_byt5":
        return map_hunyuan_video1_5_byt5_keys(pt_state_dict)
    elif model_type == "hunyuan_video1_5_siglip":
        return map_hunyuan_video1_5_siglip_keys(pt_state_dict)
    elif model_type == "hunyuan_video_dit":
        # HunyuanVideo 1.0's DiT (T2V only) -- cross-checked against the
        # real downloaded checkpoint. See
        # `hunyuan_video.map_hunyuan_video_dit_keys`'s module docstring.
        return map_hunyuan_video_dit_keys(pt_state_dict)
    elif model_type == "hunyuan_video_vae":
        return map_hunyuan_video_vae_keys(pt_state_dict)
    elif model_type == "hunyuan_video_llama_text":
        return map_hunyuan_video_llama_text_keys(pt_state_dict)
    elif model_type == "hunyuan_video_clip_text":
        return map_hunyuan_video_clip_text_keys(pt_state_dict)
    elif model_type == "hunyuan_video_llava_llama_text":
        return map_hunyuan_video_llava_llama_text_keys(pt_state_dict)
    elif model_type == "hunyuan_video_clip_vision":
        return map_hunyuan_video_clip_vision_keys(pt_state_dict)
    elif model_type == "hunyuan_video_llava_projector":
        return map_hunyuan_video_llava_projector_keys(pt_state_dict)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
