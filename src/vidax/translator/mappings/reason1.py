"""PyTorch state_dict -> Flax parameter tree mapping for the Reason1 text
encoder (`vidax.models.cosmos2_5.reason1.Qwen2TextModel`).

Key layout (`cosmos_predict2/_src/reason1/models/vlm_qwen.py`'s `QwenModel`
wraps a local `Qwen2_5_VLModel` (`.../networks/qwen2_5_vl.py:1157`) under the
`model` attribute, alongside a separate `visual` (vision tower, not ported --
never invoked for text-only calls) and `lm_head` (not ported -- only
hidden_states are needed, not logits)):

  model.embed_tokens.weight
  model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
  model.layers.{i}.self_attn.o_proj.weight
  model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.post_attention_layernorm.weight
  model.norm.weight
  visual.*        (skipped -- vision encoder/patch-merger)
  lm_head.weight  (skipped -- vocab projection, dead weight for this use case)

`model.rotary_emb.inv_freq` is a non-persistent buffer (recomputed from
config, not learned) and never appears in a saved state_dict, so there is
nothing to map for it -- `Qwen2TextModel` recomputes the same table from
`rope_theta` directly.
"""
import re
from typing import Any, Dict

from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy
from .common import _leaf_name, _set_nested_dict


def map_reason1_text_encoder_keys(pt_state_dict: Dict) -> Dict:
    """Translates a Reason1/Qwen2.5-VL text-tower state_dict into a Flax
    param tree for `vidax.models.cosmos2_5.reason1.Qwen2TextModel`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if pt_key.startswith("visual.") or pt_key == "lm_head.weight" or not pt_key.startswith("model."):
            continue

        sub_key = pt_key[len("model."):]

        if sub_key == "embed_tokens.weight":
            # nn.Embedding table: (vocab_size, hidden_size), same layout as
            # Flax's nn.Embed -- must NOT go through convert_pt_tensor_to_jax,
            # whose generic 2D rule assumes an nn.Linear weight and would
            # wrongly transpose it (see map_wan_t5_keys for the same guard).
            _set_nested_dict(jax_params, ["embed_tokens", "embedding"], pt_tensor_to_numpy(pt_tensor))
            continue

        jax_tensor = convert_pt_tensor_to_jax(pt_key, pt_tensor)

        if sub_key == "norm.weight":
            _set_nested_dict(jax_params, ["norm", "scale"], jax_tensor)
            continue

        match = re.match(r"layers\.(\d+)\.(.*)", sub_key)
        if not match:
            continue  # e.g. `rotary_emb.*`, if ever present -- no learned params.
        block_path = [f"layers_{match.group(1)}"]
        layer_sub = match.group(2)

        if layer_sub in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            name = layer_sub.split(".")[0]
            _set_nested_dict(jax_params, block_path + [name, "scale"], jax_tensor)
        elif layer_sub.startswith("self_attn."):
            proj = layer_sub[len("self_attn."):].split(".")[0]  # q_proj, k_proj, v_proj, o_proj
            _set_nested_dict(jax_params, block_path + ["self_attn", proj, _leaf_name(layer_sub)], jax_tensor)
        elif layer_sub.startswith("mlp."):
            proj = layer_sub[len("mlp."):].split(".")[0]  # gate_proj, up_proj, down_proj
            _set_nested_dict(jax_params, block_path + ["mlp", proj, _leaf_name(layer_sub)], jax_tensor)

    return {"params": jax_params}
