"""PyTorch state_dict -> Flax parameter tree key mapping for Cosmos-Predict2.5's
DiT (`vidax.models.cosmos2_5.dit.CosmosDiT`).

The released checkpoint (`model_ema_bf16.pt`/`model_ema_fp32.pt`) is a flat
`dict[str, Tensor]`, every key prefixed `net.` (the EMA net's state_dict,
saved with that prefix regardless). Buffers with no analogue in this port
are silently skipped:
  - `net.pos_embedder.*` -- RoPE lookup tables, regenerated fresh every
    forward pass by `vidax.models.cosmos2_5.rope.create_cosmos_rope3d_freqs`,
    not learned.
  - `net.accum_*` -- training-loop bookkeeping (sample/iteration counters).
  - `net.*._extra_state` -- TransformerEngine FP8/internal bookkeeping blobs.
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax
from .common import _leaf_name, _set_nested_dict

# self_attn/cross_attn submodule attribute -> vidax `cosmos_attend` name
# suffix (identical naming, kept as an explicit table for clarity/symmetry
# with `translator.mappings.common._ATTN_SUBMODULE_NAMES`).
_ATTN_SUBMODULE_NAMES = ("q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm")

_SKIP_SUFFIXES = ("._extra_state",)
_SKIP_PREFIXES = ("net.pos_embedder.", "net.accum_")


def _map_attention_submodule(pt_sub: str, jax_prefix: str, block_path: list,
                              jax_tensor, jax_params: dict) -> bool:
    for name in _ATTN_SUBMODULE_NAMES:
        if pt_sub == f"{name}.weight":
            leaf = "scale" if "norm" in name else "kernel"
            _set_nested_dict(jax_params, block_path + [f"{jax_prefix}_{name}", leaf], jax_tensor)
            return True
    return False


def map_cosmos2_5_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Cosmos-Predict2.5 `MinimalV1LVGDiT` state_dict (2B base
    checkpoint) into a Flax param tree for `CosmosDiT`.
    """
    jax_params: Dict[str, Any] = {}

    for raw_key, pt_tensor in pt_state_dict.items():
        if not raw_key.startswith("net."):
            continue  # Not part of the DiT's own state (shouldn't happen for this checkpoint).
        pt_key = raw_key[len("net."):]

        if any(raw_key.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(pt_key.endswith(s) for s in _SKIP_SUFFIXES):
            continue

        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key == "x_embedder.proj.1.weight":
            _set_nested_dict(jax_params, ["x_embedder_proj_1", "kernel"], jax_tensor)

        elif pt_key in ("t_embedder.1.linear_1.weight", "t_embedder.1.linear_2.weight"):
            idx = "1" if pt_key.endswith("linear_1.weight") else "2"
            _set_nested_dict(jax_params, [f"t_embedder_1_linear_{idx}", "kernel"], jax_tensor)

        elif pt_key == "t_embedding_norm.weight":
            _set_nested_dict(jax_params, ["t_embedding_norm", "scale"], jax_tensor)

        elif pt_key.startswith("crossattn_proj.0."):
            _set_nested_dict(jax_params, ["crossattn_proj_0", _leaf_name(pt_key)], jax_tensor)

        elif pt_key.startswith("final_layer."):
            sub = pt_key[len("final_layer."):]
            if sub == "linear.weight":
                _set_nested_dict(jax_params, ["final_layer", "linear", "kernel"], jax_tensor)
            elif sub.startswith("adaln_modulation."):
                match = re.match(r"adaln_modulation\.(1|2)\.weight$", sub)
                if match:
                    _set_nested_dict(
                        jax_params, ["final_layer", f"adaln_modulation_{match.group(1)}", "kernel"],
                        jax_tensor)

        elif pt_key.startswith("blocks."):
            match = re.match(r"blocks\.(\d+)\.(.*)", pt_key)
            block_idx, sub_key = match.group(1), match.group(2)
            block_path = [f"blocks_{block_idx}"]

            if sub_key.startswith("adaln_modulation_self_attn."):
                m = re.match(r"adaln_modulation_self_attn\.(1|2)\.weight$", sub_key)
                if m:
                    _set_nested_dict(
                        jax_params, block_path + [f"adaln_modulation_self_attn_{m.group(1)}", "kernel"],
                        jax_tensor)
            elif sub_key.startswith("adaln_modulation_cross_attn."):
                m = re.match(r"adaln_modulation_cross_attn\.(1|2)\.weight$", sub_key)
                if m:
                    _set_nested_dict(
                        jax_params, block_path + [f"adaln_modulation_cross_attn_{m.group(1)}", "kernel"],
                        jax_tensor)
            elif sub_key.startswith("adaln_modulation_mlp."):
                m = re.match(r"adaln_modulation_mlp\.(1|2)\.weight$", sub_key)
                if m:
                    _set_nested_dict(
                        jax_params, block_path + [f"adaln_modulation_mlp_{m.group(1)}", "kernel"],
                        jax_tensor)
            elif sub_key.startswith("self_attn."):
                _map_attention_submodule(
                    sub_key[len("self_attn."):], "self_attn", block_path, jax_tensor, jax_params)
            elif sub_key.startswith("cross_attn."):
                _map_attention_submodule(
                    sub_key[len("cross_attn."):], "cross_attn", block_path, jax_tensor, jax_params)
            elif sub_key == "mlp.layer1.weight":
                _set_nested_dict(jax_params, block_path + ["mlp_layer1", "kernel"], jax_tensor)
            elif sub_key == "mlp.layer2.weight":
                _set_nested_dict(jax_params, block_path + ["mlp_layer2", "kernel"], jax_tensor)

        # Any other keys are intentionally ignored (see module docstring).

    return {"params": jax_params}
