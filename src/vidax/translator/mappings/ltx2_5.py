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


def _map_conv_encoder_keys(pt_state_dict: Dict, encoder_prefix: str = "encoder.") -> Dict:
    """Translates one `Encoder` tower's state (conv-in/res-blocks/down-
    blocks/conv-out) into a `vidax.models.ltx2_5.vae.Encoder` param subtree
    -- shared by both VAE variants (the conv checkpoint's `encoder.*` and
    the diffusion checkpoint's `encoder.*`), which ship the same encoder
    architecture/key layout (confirmed from both checkpoints' embedded
    `config.vae.encoder`, see `vidax.models.ltx2_5.diffusion_vae`'s module
    docstring). Returns the `["encoder", ...]`-rooted subtree only --
    callers merge it into their own `jax_params`.
    """
    jax_params: Dict[str, Any] = {}
    for pt_key, pt_tensor in pt_state_dict.items():
        if not pt_key.startswith(encoder_prefix):
            continue
        sub_key = pt_key[len(encoder_prefix):]
        jax_tensor = _convert(pt_key, pt_tensor)
        tower_path = ["encoder"]

        if sub_key in ("conv_in.conv.weight", "conv_in.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_in", _leaf_name(sub_key)], jax_tensor)
        elif sub_key in ("conv_out.conv.weight", "conv_out.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_out", _leaf_name(sub_key)], jax_tensor)
        else:
            match = re.match(r"down_blocks\.(\d+)\.(.*)", sub_key)
            if not match:
                continue
            block_idx, block_sub = match.groups()
            block_path = tower_path + [f"down_blocks_{block_idx}"]

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
    return jax_params


def map_ltx2_5_vae_keys(pt_state_dict: Dict) -> Dict:
    """Translates LTX-2.5's conv-decoder `CausalVideoAutoencoder` state_dict
    into a Flax param tree for `vidax.models.ltx2_5.vae.LTXVAE`. No
    `model.`/`vae.` prefix -- the VAE checkpoint's own top-level keys are
    already bare `encoder.`/`decoder.`/`per_channel_statistics.`.
    """
    jax_params: Dict[str, Any] = _map_conv_encoder_keys(pt_state_dict)

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
        if not pt_key.startswith("decoder."):
            continue
        jax_tensor = _convert(pt_key, pt_tensor)
        sub_key = pt_key[len("decoder."):]
        tower_path = ["decoder"]

        if sub_key in ("conv_in.conv.weight", "conv_in.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_in", _leaf_name(sub_key)], jax_tensor)
        elif sub_key in ("conv_out.conv.weight", "conv_out.conv.bias"):
            _set_nested_dict(jax_params, tower_path + ["conv_out", _leaf_name(sub_key)], jax_tensor)
        else:
            match = re.match(r"up_blocks\.(\d+)\.(.*)", sub_key)
            if not match:
                continue
            block_idx, block_sub = match.groups()
            block_path = tower_path + [f"up_blocks_{block_idx}"]

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


def _map_dense_submodule(sub_key: str, path: list, jax_tensor, jax_params: dict, name: str) -> bool:
    """`name.weight`/`name.bias` -> Flax `Dense`'s `kernel`/`bias`."""
    match = re.match(rf"{re.escape(name)}\.(weight|bias)$", sub_key)
    if not match:
        return False
    field = match.group(1)
    _set_nested_dict(jax_params, path + [name, "bias" if field == "bias" else "kernel"], jax_tensor)
    return True


def _map_na_attention_submodule(sub_key: str, attn_path: list, pt_key: str, pt_tensor, jax_params: dict) -> bool:
    """`vidax.models.ltx2_5.diffusion_vae.NeighborhoodAttention3D`'s own
    `to_q`/`to_k`/`to_v`/`to_out`/`q_norm`/`k_norm` -- splits the
    checkpoint's *fused* `qkv.{weight,bias}` into thirds (q, k, v order,
    confirmed from the reference's own `_split_fused_qkv_param`) since this
    checkpoint ships them fused (see module docstring). The checkpoint's own
    raw key for the output projection is `attn.proj.*` -- mapped to this
    port's `to_out` (not `proj`), see `NeighborhoodAttention3D`'s own
    docstring for why (Megatron-TP sharding name reuse + a same-file naming
    collision to avoid).
    """
    if sub_key in ("q_norm.weight", "k_norm.weight"):
        name = "q_norm" if sub_key.startswith("q_norm") else "k_norm"
        _set_nested_dict(jax_params, attn_path + [name, "scale"], pt_tensor_to_numpy(pt_tensor))
        return True
    if sub_key in ("qkv.weight", "qkv.bias"):
        field = "weight" if sub_key.endswith("weight") else "bias"
        d = pt_tensor.shape[0] // 3
        for i, name in enumerate(("to_q", "to_k", "to_v")):
            piece_key = pt_key  # only used by `_convert` to decide transpose; same rule for all thirds
            piece = pt_tensor[i * d:(i + 1) * d]
            jax_piece = _convert(piece_key, piece)
            _set_nested_dict(jax_params, attn_path + [name, "bias" if field == "bias" else "kernel"], jax_piece)
        return True
    if sub_key in ("proj.weight", "proj.bias"):
        field = "bias" if sub_key.endswith("bias") else "kernel"
        _set_nested_dict(jax_params, attn_path + ["to_out", field], _convert(pt_key, pt_tensor))
        return True
    return False


def _map_swiglu_submodule(sub_key: str, path: list, jax_tensor, jax_params: dict) -> bool:
    """`vidax.models.ltx2_5.diffusion_vae.SwiGLU`'s `w_gate`/`w_up`/`w_down`
    -- each an unbiased `Dense`, so only `.weight` keys ever appear."""
    match = re.match(r"(w_gate|w_up|w_down)\.weight$", sub_key)
    if not match:
        return False
    _set_nested_dict(jax_params, path + ["mlp", match.group(1), "kernel"], jax_tensor)
    return True


def map_ltx2_5_diffusion_decoder_keys(pt_state_dict: Dict) -> Dict:
    """Translates the diffusion (NATTEN) VAE checkpoint
    (`ltx-2.5-video-vae-bf16.safetensors`, `config.vae._class_name ==
    "CausalDiffusionVAE"`) into a Flax param tree for
    `vidax.models.ltx2_5.diffusion_vae.DiffusionVideoDecoder` (its
    `encoder.*`/`per_channel_statistics.*` keys are shared with
    `map_ltx2_5_vae_keys` via `_map_conv_encoder_keys` -- only this
    checkpoint's own `decoder.*` diffusion-decoder keys are handled here).
    `decoder.type_emb` is intentionally skipped -- see module docstring for
    why (unreferenced anywhere in the real reference's own loader/model
    code).
    """
    jax_params: Dict[str, Any] = {"encoder": _map_conv_encoder_keys(pt_state_dict)["encoder"]}

    for pt_key, pt_tensor in pt_state_dict.items():
        if pt_key.startswith("per_channel_statistics."):
            # One shared top-level buffer pair -- the real checkpoint has no
            # separate `decoder.per_channel_statistics.*` (confirmed from its
            # own keys), so this duplicates into *both* `encoder.
            # per_channel_statistics_{mean,std}` and the decoder's own
            # top-level `per_channel_statistics_{mean,std}` (matching the
            # reference's `DiffusionVideoDecoder.__init__` constructing its
            # own separate `PerChannelStatistics` submodule from this same
            # checkpoint key) -- same duplication pattern as
            # `map_ltx2_5_vae_keys`.
            stat = "mean" if pt_key.endswith("mean-of-means") else "std"
            stat_array = pt_tensor_to_numpy(pt_tensor)
            _set_nested_dict(jax_params, ["encoder", f"per_channel_statistics_{stat}"], stat_array)
            _set_nested_dict(jax_params, [f"per_channel_statistics_{stat}"], stat_array)
            continue
        if not pt_key.startswith("decoder."):
            continue
        sub_key = pt_key[len("decoder."):]
        if sub_key == "type_emb":
            continue

        if _map_dense_submodule(sub_key, [], _convert(pt_key, pt_tensor), jax_params, "conv_in"):
            continue
        if _map_dense_submodule(sub_key, [], _convert(pt_key, pt_tensor), jax_params, "conv_in_x_t"):
            continue
        if _map_dense_submodule(sub_key, [], _convert(pt_key, pt_tensor), jax_params, "conv_out"):
            continue
        if sub_key == "norm_out.weight":
            _set_nested_dict(jax_params, ["norm_out", "scale"], _convert(pt_key, pt_tensor))
            continue
        if sub_key.startswith("t_embedder.mlp.0.") or sub_key.startswith("t_embedder.mlp.2."):
            name = "t_embedder_linear_1" if "mlp.0." in sub_key else "t_embedder_linear_2"
            field = "bias" if sub_key.endswith("bias") else "kernel"
            _set_nested_dict(jax_params, [name, field], _convert(pt_key, pt_tensor))
            continue
        if sub_key.startswith("shared_adaln.proj."):
            _map_dense_submodule(
                sub_key[len("shared_adaln."):], ["shared_adaln"], _convert(pt_key, pt_tensor), jax_params, "proj")
            continue

        match = re.match(r"det_stages\.(\d+)\.(\d+)\.(.*)", sub_key)
        if match:
            stage_idx, block_idx, block_sub = match.groups()
            block_path = [f"det_stages_{stage_idx}_{block_idx}"]
            if block_sub.startswith("attn."):
                _map_na_attention_submodule(block_sub[len("attn."):], block_path + ["attn"], pt_key, pt_tensor, jax_params)
            elif block_sub == "norm1.weight":
                _set_nested_dict(jax_params, block_path + ["norm1", "scale"], _convert(pt_key, pt_tensor))
            elif block_sub == "norm2.weight":
                _set_nested_dict(jax_params, block_path + ["norm2", "scale"], _convert(pt_key, pt_tensor))
            elif block_sub.startswith("mlp."):
                _map_swiglu_submodule(block_sub[len("mlp."):], block_path, _convert(pt_key, pt_tensor), jax_params)
            continue

        match = re.match(r"upsamples\.(\d+)\.(.*)", sub_key)
        if match:
            up_idx, up_sub = match.groups()
            _map_dense_submodule(up_sub, [f"upsamples_{up_idx}"], _convert(pt_key, pt_tensor), jax_params, "proj")
            continue

        match = re.match(r"diff_blocks\.(\d+)\.(.*)", sub_key)
        if match:
            block_idx, block_sub = match.groups()
            block_path = [f"diff_blocks_{block_idx}"]
            if block_sub.startswith("attn."):
                _map_na_attention_submodule(block_sub[len("attn."):], block_path + ["attn"], pt_key, pt_tensor, jax_params)
            elif block_sub == "norm1.weight":
                _set_nested_dict(jax_params, block_path + ["norm1", "scale"], _convert(pt_key, pt_tensor))
            elif block_sub == "norm2.weight":
                _set_nested_dict(jax_params, block_path + ["norm2", "scale"], _convert(pt_key, pt_tensor))
            elif block_sub.startswith("mlp."):
                _map_swiglu_submodule(block_sub[len("mlp."):], block_path, _convert(pt_key, pt_tensor), jax_params)
            elif block_sub == "scale_shift_table":
                _set_nested_dict(jax_params, block_path + ["scale_shift_table"], pt_tensor_to_numpy(pt_tensor))
            elif block_sub.startswith("context_proj."):
                _map_dense_submodule(
                    block_sub, block_path, _convert(pt_key, pt_tensor), jax_params, "context_proj")
            continue

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
