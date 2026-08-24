"""PyTorch state_dict -> Flax parameter tree mapping for LTX-2.5 (video-only).

Four separate checkpoints, each with its own map function:

- The DiT checkpoint (`ltx-2.5-22b-{dev,distilled}-transformer-bf16.
  safetensors`) is a single flat file with a `model.diffusion_model.`
  prefix bundling the video DiT, the video *and* audio embeddings
  connectors, and every audio-branch weight -- `map_ltx2_5_dit_keys`/
  `map_ltx2_5_connector_keys` each filter to their own video-only subset and
  skip every `audio_*`/`av_ca_*`/`*_a2v_*`/`*_v2a_*` key (this port is
  video-only, see `vidax.models.ltx2_5.dit`'s module docstring).
- The VAE checkpoint (`ltx-2.5-video-vae-conv-bf16.safetensors`) is its own
  file, `encoder.`/`decoder.` prefixed, same shape as
  `vidax.translator.mappings.ltx_video.map_ltx_video_vae_keys` minus the
  timestep-conditioning keys (real checkpoint has none -- see
  `vidax.models.ltx2_5.vae`).
- The Gemma checkpoint (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`) is
  its own file, `model.` prefixed (standard HF `Gemma4UnifiedTextModel`
  naming) plus a top-level `text_embedding_projection.video_aggregate_embed`
  Linear -- `map_gemma4_text_keys` maps the former,
  `map_gemma4_video_projection_keys` the latter (audio's `vision_model.*`/
  `audio_projector.*`/`multi_modal_projector.*`/`text_embedding_projection.
  audio_aggregate_embed` are all skipped, this port is text-only).
"""
import re
from typing import Any, Dict

from .common import _leaf_name, _set_nested_dict
from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy

_DIT_PREFIX = "model.diffusion_model."
_VAE_PREFIX = ""  # VAE checkpoint keys are already bare `encoder.`/`decoder.`/...
_GEMMA_PREFIX = "model."

# Raw AdaLN modulation tables ((9,dim)/(6,dim)/(2,dim)/(1,dim)) and the
# connector's learnable registers -- never go through the generic ndim==2
# "Linear weight" transpose (same trap as LTX-Video's `scale_shift_table`,
# see `vidax.translator.mappings.ltx_video`).
_NO_TRANSPOSE_SUFFIXES = (
    "scale_shift_table", "prompt_scale_shift_table", "learnable_registers", "keyframes_abs_pos_embedding",
)


def _convert(pt_key: str, pt_tensor) -> Any:
    if pt_key.endswith(_NO_TRANSPOSE_SUFFIXES):
        return pt_tensor_to_numpy(pt_tensor)
    return convert_pt_tensor_to_jax(pt_key, pt_tensor)


def _is_audio_or_av(key: str) -> bool:
    return (
        key.startswith("audio_") or "audio_embeddings_connector" in key
        or key.startswith("av_ca_") or "a2v" in key or "v2a" in key
        or key.endswith("scale_shift_table_a2v_ca_video") or key.endswith("scale_shift_table_a2v_ca_audio")
    )


def _map_gated_attention_submodule(sub_key: str, attn_path: list, jax_tensor, jax_params: dict) -> bool:
    """`attn1`/`attn2` (DiT) or the connector's `attn1` -- adds
    `to_gate_logits` on top of `vidax.translator.mappings.ltx_video.
    _map_attention_submodule`'s handling (same `q_norm`/`k_norm`/`to_q`/
    `to_k`/`to_v`/`to_out` shape, this port's checkpoints always have
    `apply_gated_attention=True`).
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
    match = re.match(r"to_gate_logits\.(weight|bias)$", sub_key)
    if match:
        field = match.group(1)
        _set_nested_dict(jax_params, attn_path + ["to_gate_logits", "bias" if field == "bias" else "kernel"], jax_tensor)
        return True
    return False


def map_ltx2_5_dit_keys(pt_state_dict: Dict) -> Dict:
    """Translates LTX-2.5's `LTXModel` (video-only) state_dict (filtered to
    `model.diffusion_model.`, audio/AV keys skipped) into a Flax param tree
    for `vidax.models.ltx2_5.dit.LTXDiT`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(_DIT_PREFIX):
            continue
        key = pt_key[len(_DIT_PREFIX):]
        if _is_audio_or_av(key) or key.startswith("video_embeddings_connector") or key.startswith(
                "audio_embeddings_connector"):
            continue
        jax_tensor = _convert(pt_key, pt_tensor)

        if key == "scale_shift_table":
            _set_nested_dict(jax_params, ["scale_shift_table"], jax_tensor)
        elif key == "keyframes_abs_pos_embedding":
            _set_nested_dict(jax_params, ["keyframes_abs_pos_embedding"], jax_tensor)
        elif key.startswith("patchify_proj."):
            _set_nested_dict(jax_params, ["patchify_proj", _leaf_name(key)], jax_tensor)
        elif key.startswith("proj_out."):
            _set_nested_dict(jax_params, ["proj_out", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.emb.timestep_embedder.linear_1."):
            _set_nested_dict(jax_params, ["adaln_single_emb_timestep_embedder_linear_1", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.emb.timestep_embedder.linear_2."):
            _set_nested_dict(jax_params, ["adaln_single_emb_timestep_embedder_linear_2", _leaf_name(key)], jax_tensor)
        elif key.startswith("adaln_single.linear."):
            _set_nested_dict(jax_params, ["adaln_linear", _leaf_name(key)], jax_tensor)
        elif key.startswith("prompt_adaln_single.emb.timestep_embedder.linear_1."):
            _set_nested_dict(
                jax_params, ["prompt_adaln_single_emb_timestep_embedder_linear_1", _leaf_name(key)], jax_tensor)
        elif key.startswith("prompt_adaln_single.emb.timestep_embedder.linear_2."):
            _set_nested_dict(
                jax_params, ["prompt_adaln_single_emb_timestep_embedder_linear_2", _leaf_name(key)], jax_tensor)
        elif key.startswith("prompt_adaln_single.linear."):
            _set_nested_dict(jax_params, ["prompt_adaln_single_linear", _leaf_name(key)], jax_tensor)
        elif key.startswith("transformer_blocks."):
            match = re.match(r"transformer_blocks\.(\d+)\.(.*)", key)
            block_idx, sub_key = match.group(1), match.group(2)
            block_path = [f"blocks_{block_idx}"]

            if sub_key == "scale_shift_table":
                _set_nested_dict(jax_params, block_path + ["scale_shift_table"], jax_tensor)
            elif sub_key == "prompt_scale_shift_table":
                _set_nested_dict(jax_params, block_path + ["prompt_scale_shift_table"], jax_tensor)
            elif sub_key.startswith("attn1."):
                _map_gated_attention_submodule(sub_key[len("attn1."):], block_path + ["attn1"], jax_tensor, jax_params)
            elif sub_key.startswith("attn2."):
                _map_gated_attention_submodule(sub_key[len("attn2."):], block_path + ["attn2"], jax_tensor, jax_params)
            elif sub_key.startswith("ff.net.0.proj."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_proj", _leaf_name(sub_key)], jax_tensor)
            elif sub_key.startswith("ff.net.2."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_out", _leaf_name(sub_key)], jax_tensor)

    return {"params": jax_params}


def map_ltx2_5_connector_keys(pt_state_dict: Dict) -> Dict:
    """Translates the DiT checkpoint's `video_embeddings_connector.*` state
    (its weights live inside the DiT file -- see module docstring) into a
    Flax param tree for `vidax.models.ltx2_5.connector.Embeddings1DConnector`.
    """
    jax_params: Dict[str, Any] = {}
    prefix = _DIT_PREFIX + "video_embeddings_connector."

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(prefix):
            continue
        key = pt_key[len(prefix):]
        jax_tensor = _convert(pt_key, pt_tensor)

        if key == "learnable_registers":
            _set_nested_dict(jax_params, ["learnable_registers"], jax_tensor)
        else:
            match = re.match(r"transformer_1d_blocks\.(\d+)\.(.*)", key)
            if not match:
                continue
            block_idx, sub_key = match.groups()
            block_path = [f"transformer_1d_blocks_{block_idx}"]

            if sub_key.startswith("attn1."):
                _map_gated_attention_submodule(sub_key[len("attn1."):], block_path + ["attn1"], jax_tensor, jax_params)
            elif sub_key.startswith("ff.net.0.proj."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_proj", _leaf_name(sub_key)], jax_tensor)
            elif sub_key.startswith("ff.net.2."):
                _set_nested_dict(jax_params, block_path + ["ff", "ff_out", _leaf_name(sub_key)], jax_tensor)

    return {"params": jax_params}


def map_ltx2_5_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates LTX-2.5's conv-decoder `CausalVideoAutoencoder` state_dict
    into a Flax param tree for `vidax.models.ltx2_5.vae.LTXVAE`. No
    `model.`/`vae.` prefix -- the VAE checkpoint's own top-level keys are
    already bare `encoder.`/`decoder.`/`per_channel_statistics.`.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if pt_key.startswith("per_channel_statistics."):
            # A single top-level buffer pair, duplicated into *both*
            # `encoder.per_channel_statistics_{mean,std}` and
            # `decoder.per_channel_statistics_{mean,std}` -- each of
            # `vidax.models.ltx2_5.vae.Encoder`/`Decoder` owns its own copy
            # and applies it internally (normalize on encode, un-normalize
            # on decode), matching the real reference's `VideoEncoder`/
            # `ConvVideoDecoder`, each of which has its own
            # `PerChannelStatistics` submodule loaded from this same
            # checkpoint key (see `VAE_ENCODER_COMFY_KEYS_FILTER`/
            # `VAE_DECODER_COMFY_KEYS_FILTER` in the reference).
            stat = "mean" if pt_key.endswith("mean-of-means") else "std"
            stat_array = pt_tensor_to_numpy(pt_tensor)
            _set_nested_dict(jax_params, ["encoder", f"per_channel_statistics_{stat}"], stat_array)
            _set_nested_dict(jax_params, ["decoder", f"per_channel_statistics_{stat}"], stat_array)
            continue
        jax_tensor = _convert(pt_key, pt_tensor)

        match = re.match(r"(encoder|decoder)\.(.*)", pt_key)
        if not match:
            continue
        tower, sub_key = match.groups()
        tower_path = [tower]

        if sub_key in ("conv_in.conv.weight", "conv_in.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_in", _leaf_name(sub_key)], jax_tensor)
        elif sub_key in ("conv_out.conv.weight", "conv_out.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_out", _leaf_name(sub_key)], jax_tensor)
        else:
            match = re.match(r"(?:down|up)_blocks\.(\d+)\.(.*)", sub_key)
            if not match:
                continue
            block_idx, block_sub = match.groups()
            block_path = tower_path + [f"{'down' if tower == 'encoder' else 'up'}_blocks_{block_idx}"]

            if block_sub in ("conv.conv.weight", "conv.conv.bias"):
                _set_nested_dict(jax_params, block_path + ["conv", _leaf_name(block_sub)], jax_tensor)
            else:
                res_match = re.match(r"res_blocks\.(\d+)\.(.*)", block_sub)
                if not res_match:
                    continue
                res_idx, res_sub = res_match.groups()
                res_path = block_path + [f"res_blocks_{res_idx}"]
                if res_sub.startswith("conv1.conv."):
                    _set_nested_dict(jax_params, res_path + ["conv1", _leaf_name(res_sub)], jax_tensor)
                elif res_sub.startswith("conv2.conv."):
                    _set_nested_dict(jax_params, res_path + ["conv2", _leaf_name(res_sub)], jax_tensor)

    return {"params": jax_params}


def map_gemma4_text_keys(pt_state_dict: Dict) -> Dict:
    """Translates the Gemma checkpoint's `model.*` (text tower) state into a
    Flax param tree for `vidax.models.ltx2_5.gemma4.Gemma4TextModel`. Skips
    `vision_model.*`/`audio_projector.*`/`multi_modal_projector.*` (this
    port is text-only) and any `hf_asset__*`/`tokenizer_json.*` metadata-as-
    tensor entries some checkpoint dumps carry.
    """
    jax_params: Dict[str, Any] = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(_GEMMA_PREFIX):
            continue
        key = pt_key[len(_GEMMA_PREFIX):]

        if key == "embed_tokens.weight":
            _set_nested_dict(jax_params, ["embed_tokens", "embedding"], pt_tensor_to_numpy(pt_tensor))
        elif key == "norm.weight":
            _set_nested_dict(jax_params, ["norm_scale"], pt_tensor_to_numpy(pt_tensor))
        elif key.startswith("layers."):
            match = re.match(r"layers\.(\d+)\.(.*)", key)
            if not match:
                continue
            layer_idx, sub_key = match.groups()
            layer_path = [f"layers_{layer_idx}"]

            if sub_key == "layer_scalar":
                _set_nested_dict(jax_params, layer_path + ["layer_scalar"], pt_tensor_to_numpy(pt_tensor))
            elif sub_key in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            ):
                # e.g. "input_layernorm.weight" -> "input_layernorm_scale"
                name = sub_key[: -len(".weight")] + "_scale"
                _set_nested_dict(jax_params, layer_path + [name], pt_tensor_to_numpy(pt_tensor))
            elif sub_key.startswith("self_attn."):
                attn_sub = sub_key[len("self_attn."):]
                if attn_sub in ("q_norm.weight", "k_norm.weight"):
                    name = "q_norm_scale" if attn_sub.startswith("q_norm") else "k_norm_scale"
                    _set_nested_dict(jax_params, layer_path + ["self_attn", name], pt_tensor_to_numpy(pt_tensor))
                else:
                    match2 = re.match(r"(q|k|v|o)_proj\.weight$", attn_sub)
                    if match2:
                        name = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}[match2.group(1)]
                        _set_nested_dict(
                            jax_params, layer_path + ["self_attn", name, "kernel"],
                            convert_pt_tensor_to_jax(pt_key, pt_tensor))
            elif sub_key.startswith("mlp."):
                mlp_sub = sub_key[len("mlp."):]
                match2 = re.match(r"(gate|up|down)_proj\.weight$", mlp_sub)
                if match2:
                    name = f"{match2.group(1)}_proj"
                    _set_nested_dict(
                        jax_params, layer_path + ["mlp", name, "kernel"], convert_pt_tensor_to_jax(pt_key, pt_tensor))

    return {"params": jax_params}


def load_gemma4_video_aggregate_embed(pt_state_dict: Dict):
    """Returns `(kernel, bias)` numpy arrays for `text_embedding_projection.
    video_aggregate_embed` (the `FeatureExtractorV2` Linear, see
    `vidax.models.ltx2_5.gemma4.extract_video_features`) -- not a Flax
    param tree, since that function takes the projection as plain arrays
    rather than owning a Flax module (it runs once per prompt, outside any
    `nn.Module`, same reasoning as `vidax.models.ltx_video.configs.
    load_ltx_vae_per_channel_stats`).
    """
    kernel = pt_tensor_to_numpy(pt_state_dict["text_embedding_projection.video_aggregate_embed.weight"]).T
    bias = pt_tensor_to_numpy(pt_state_dict["text_embedding_projection.video_aggregate_embed.bias"])
    return kernel, bias
