"""PyTorch state_dict -> Flax parameter tree key mapping for Cosmos3's DiT
(`vidax.models.cosmos3.dit.Cosmos3Transformer`), shared by both Cosmos3-Nano
and Cosmos3-Edge checkpoints.

The released checkpoints (`transformer/diffusion_pytorch_model.safetensors.index.json`)
are already a *flat* layout (`layers.N.self_attn.to_q.weight`, no leading
`model.`/`net.` prefix to strip -- unlike Cosmos-Predict2.5's `net.blocks.N...`).
Keys with no analogue in this T2V/I2V-only port are silently skipped:
  - `lm_head.weight` -- next-token prediction head, never used for generation.
  - `norm.weight` -- final norm of the "und" (text/understanding) pathway,
    only consumed by `lm_head`; `gen`'s cross-attention reads `und`'s
    per-layer keys/values directly, never this final normed output.
  - `action_*`/`audio_*` -- action- and sound-generation heads, out of scope
    (T2V/I2V only).
Keys present only for some checkpoints (e.g. `self_attn.norm_q`/`norm_k`,
absent when `qk_norm_for_text=False`; `self_attn.k_norm_und_for_gen`, present
only when `use_und_k_norm_for_gen=True`) map normally when present and are
simply never encountered in state_dicts that don't have them.
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy
from .common import _leaf_name, _set_nested_dict

# self_attn submodule attribute -> (jax name, is a per-head RMSNorm scale).
_ATTN_LEAF_NAMES = (
    "to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "norm_q", "norm_k", "norm_added_q", "norm_added_k", "k_norm_und_for_gen",
)

_SKIP_PREFIXES = ("lm_head.", "action_", "audio_")


def _map_attention_submodule(pt_sub: str, block_path: list, jax_tensor, jax_params: dict) -> bool:
    for name in _ATTN_LEAF_NAMES:
        if pt_sub == f"{name}.weight":
            leaf = "scale" if "norm" in name else "kernel"
            _set_nested_dict(jax_params, block_path + ["self_attn", name, leaf], jax_tensor)
            return True
    return False


def map_cosmos3_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Cosmos3-Nano `Cosmos3OmniTransformer` state_dict into a
    Flax param tree for `Cosmos3Transformer`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if pt_key == "norm.weight" or any(pt_key.startswith(p) for p in _SKIP_PREFIXES):
            continue

        if pt_key == "embed_tokens.weight":
            # nn.Embedding/nn.Embed keep the (vocab, features) layout as-is --
            # never transposed (matches Wan's own `token_embedding.weight` handling).
            _set_nested_dict(jax_params, ["embed_tokens", "embedding"], pt_tensor_to_numpy(pt_tensor))
            continue

        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if pt_key == "norm_moe_gen.weight":
            _set_nested_dict(jax_params, ["norm_moe_gen", "scale"], jax_tensor)

        elif pt_key.startswith("proj_in."):
            _set_nested_dict(jax_params, ["proj_in", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("proj_out."):
            _set_nested_dict(jax_params, ["proj_out", _leaf_name(pt_key)], jax_tensor)

        elif pt_key.startswith("time_embedder.linear_1."):
            _set_nested_dict(jax_params, ["time_embedder_linear_1", _leaf_name(pt_key)], jax_tensor)
        elif pt_key.startswith("time_embedder.linear_2."):
            _set_nested_dict(jax_params, ["time_embedder_linear_2", _leaf_name(pt_key)], jax_tensor)

        elif pt_key.startswith("layers."):
            match = re.match(r"layers\.(\d+)\.(.*)", pt_key)
            layer_idx, sub_key = match.group(1), match.group(2)
            block_path = [f"layers_{layer_idx}"]

            if sub_key.startswith("self_attn."):
                _map_attention_submodule(sub_key[len("self_attn."):], block_path, jax_tensor, jax_params)
            elif sub_key in ("input_layernorm.weight", "input_layernorm_moe_gen.weight",
                              "post_attention_layernorm.weight", "post_attention_layernorm_moe_gen.weight"):
                name = sub_key[:-len(".weight")]
                _set_nested_dict(jax_params, block_path + [name, "scale"], jax_tensor)
            elif sub_key.startswith("mlp."):
                proj = sub_key[len("mlp."):].split(".")[0]  # gate_proj / up_proj / down_proj
                _set_nested_dict(jax_params, block_path + ["mlp", proj, "kernel"], jax_tensor)
            elif sub_key.startswith("mlp_moe_gen."):
                proj = sub_key[len("mlp_moe_gen."):].split(".")[0]
                _set_nested_dict(jax_params, block_path + ["mlp_moe_gen", proj, "kernel"], jax_tensor)

        # Any other keys are intentionally ignored (see module docstring).

    return {"params": jax_params}
