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
from .wan2_1 import map_wan2_1_clip_keys, map_wan2_1_dit_keys, map_wan2_1_vae_keys
from .wan2_2 import map_wan2_2_vae_keys

__all__ = [
    "load_torch_checkpoint_to_jax",
    "map_wan_dit_keys",
    "map_wan_t5_keys",
    "map_wan2_1_dit_keys",
    "map_wan2_1_vae_keys",
    "map_wan2_1_clip_keys",
    "map_wan2_2_vae_keys",
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
        import torch  # Optional dependency; install the `torch` extra.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return {k: v.detach().cpu() for k, v in state_dict.items()}
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
    elif model_type == "wan2.1_vae":
        return map_wan2_1_vae_keys(pt_state_dict)
    elif model_type == "wan2.1_clip":
        return map_wan2_1_clip_keys(pt_state_dict)
    elif model_type == "wan2.2_vae":
        return map_wan2_2_vae_keys(pt_state_dict)
    elif model_type == "wan_t5":
        return map_wan_t5_keys(pt_state_dict)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
