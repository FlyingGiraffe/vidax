"""PyTorch -> Flax key mapping for HunyuanVideo 1.0's DiT, VAE, Llama text
tower, and CLIP-L pooled text encoder.

**DiT mapper cross-checked against the real downloaded checkpoint**
(`tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/
mp_rank_00_model_states.pt`, `ckpt["module"]`, 856 leaves, 20 double blocks
/ 40 single blocks / 2 token-refiner blocks, `guidance_in.*` present i.e.
the `"HYVideo-T/2-cfgdistill"` variant) -- every regex below (fused-QKV
splitting included) matches the real key names byte-for-byte; the earlier
`TODO(real-checkpoint)` markers are resolved. Target Flax module tree:
`vidax.models.hunyuan_video.hunyuan_video_1_0.dit.HunyuanVideo10DiT` (+ shared
`common/dit_layers.py`).

**Key structural difference from `hunyuan_video_1_5.py`'s DiT mapper**:
1.0's real checkpoint stores **fused** QKV/QKV+MLP-in Linears
(`img_attn_qkv`/`txt_attn_qkv`: `Linear(hidden, 3*hidden)`;
`single_blocks.N.linear1`: `Linear(hidden, 3*hidden+mlp_hidden)`), not
1.5's already-split `img_attn_q`/`img_attn_k`/`img_attn_v` naming -- see the
plan's architecture-diff section. This mapper therefore **splits** the
fused weight into contiguous chunks before writing to the Flax param tree
(`_split_fused_linear`), unlike 1.5's mapper which copies each Q/K/V weight
straight across.

**VAE/text-encoder mappers**, added this session, all cross-checked against
real downloaded checkpoints:
- `map_hunyuan_video_1_0_vae_keys`: `AutoencoderKLCausal3D`'s real
  `pytorch_model.pt` (248 leaves) -- see `hunyuan_video_1_0.vae`'s module
  docstring for the exact key/name correspondence.
- `map_hunyuan_video_1_0_llama_text_keys`: the extracted
  `xtuner/llava-llama-3-8b-v1_1-transformers`'s `.language_model`
  (`LlamaModel`, 290 leaves, no `model.` prefix since only the bare
  sub-module was saved).
- `map_hunyuan_video_1_0_clip_text_keys`: `openai/clip-vit-large-patch14`'s
  `text_model.*` (197 leaves), pooled-output-only (no vision tower).
"""
import re
from typing import Dict

from .common import _leaf_name, _set_nested_dict
from ..converter import convert_pt_tensor_to_jax, pt_tensor_to_numpy


def _lin(jax_params: dict, path: list, pt_key: str, arr) -> None:
    _set_nested_dict(jax_params, path + [_leaf_name(pt_key)], convert_pt_tensor_to_jax(pt_key, arr))


def _rms(jax_params: dict, path: list, arr) -> None:
    _set_nested_dict(jax_params, path + ["scale"], convert_pt_tensor_to_jax("weight", arr))


def _ln(jax_params: dict, path: list, pt_key: str, arr) -> None:
    # Flax `nn.LayerNorm`'s affine params are named `scale`/`bias`, not
    # `kernel`/`bias` -- unlike `_lin`, which is only for `nn.Dense`.
    leaf = "bias" if pt_key.endswith(".bias") else "scale"
    _set_nested_dict(jax_params, path + [leaf], convert_pt_tensor_to_jax(pt_key, arr))


def _split_fused_linear(arr, sizes, axis=0):
    """Splits a fused PyTorch `Linear` weight/bias (out-features-first) into
    contiguous chunks of `sizes` along `axis`. `arr` is `(sum(sizes), in)`
    for a weight or `(sum(sizes),)` for a bias -- PyTorch `Linear` stores
    `(out_features, in_features)`, so `axis=0` splits the *output* (fused)
    dimension, matching `torch.split(..., dim=-1)` on the *input*
    activation the reference performs, which is equivalent to slicing the
    weight's output rows (`W @ x` where `W`'s rows are grouped by output
    chunk). Confirmed against the real `img_attn_qkv`/`single_blocks.N.
    linear1` weight shapes (`(3*3072, 3072)`/`(3*3072+mlp_hidden, 3072)`) --
    fused-Linear output-chunking is the standard PyTorch convention
    (`models.py`'s `rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3)`).
    """
    import numpy as np
    offsets = np.cumsum([0] + list(sizes))
    return [arr[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]


_DOUBLE_QKV_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_qkv\.(weight|bias)$")
_DOUBLE_QK_NORM_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_(q|k)_norm\.weight$")
_DOUBLE_PROJ_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn_proj\.(weight|bias)$")
_DOUBLE_MOD_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_mod\.linear\.(weight|bias)$")
_DOUBLE_NORM_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_norm(1|2)\.(weight|bias)$")
_DOUBLE_MLP_RE = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_mlp\.(fc1|fc2)\.(weight|bias)$")

_SINGLE_LINEAR1_RE = re.compile(r"^single_blocks\.(\d+)\.linear1\.(weight|bias)$")
_SINGLE_QK_NORM_RE = re.compile(r"^single_blocks\.(\d+)\.(q|k)_norm\.weight$")
_SINGLE_LINEAR2_RE = re.compile(r"^single_blocks\.(\d+)\.linear2\.(weight|bias)$")
_SINGLE_MOD_RE = re.compile(r"^single_blocks\.(\d+)\.modulation\.linear\.(weight|bias)$")
_SINGLE_PRENORM_RE = re.compile(r"^single_blocks\.(\d+)\.pre_norm\.(weight|bias)$")

_REFINER_BLOCK_NORM_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$")
_REFINER_QKV_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(weight|bias)$")
_REFINER_PROJ_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_proj\.(weight|bias)$")
_REFINER_MLP_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc(1|2)\.(weight|bias)$")
_REFINER_ADALN_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.adaLN_modulation\.1\.(weight|bias)$")

_FINAL_ADALN_RE = re.compile(r"^final_layer\.adaLN_modulation\.1\.(weight|bias)$")


def map_hunyuan_video_1_0_dit_keys(pt_state_dict: Dict) -> Dict:
    """Cross-checked against the real downloaded checkpoint -- see module
    docstring. `mp_rank_00_model_states.pt` (this model predates broad
    safetensors adoption -- a raw DeepSpeed-style `.pt`, confirmed via
    `config.py`'s `--dit-weight` default path) wraps the actual state_dict
    one level down under a `"module"` key (alongside DeepSpeed's own
    optimizer/lr-scheduler bookkeeping keys, which aren't weights and are
    never matched by any regex below) -- unwrapped here rather than by the
    generic `_load_pt_state_dict` loader, since this nesting is specific to
    this one checkpoint's save format, not a general `.pt` convention.
    """
    if set(pt_state_dict.keys()) == {"module"}:
        pt_state_dict = pt_state_dict["module"]

    jax_params: Dict = {}
    # Inferred from `img_in.proj.weight`'s output channel count up front
    # (a dict pre-scan, not reliant on `pt_state_dict`'s iteration order --
    # `single_blocks.N.linear1`'s fused-QKV+MLP split needs `hidden_size`
    # but can't infer it alone, unlike the double-blocks' fused QKV-only
    # split, which has its own same-shape fallback).
    hidden_size = None
    if "img_in.proj.weight" in pt_state_dict:
        hidden_size = pt_tensor_to_numpy(pt_state_dict["img_in.proj.weight"]).shape[0]

    for pt_key, arr in pt_state_dict.items():
        if pt_key == "img_in.proj.weight":
            # Conv3d (out, in, pt, ph, pw), kernel==stride patchify -> flatten
            # to a plain Dense kernel, same reasoning as
            # `hunyuan_video_1_5.py`'s identical branch (this file's
            # `_patchify` uses the same (c, pt, ph, pw) flatten order).
            arr_np = pt_tensor_to_numpy(arr)
            out_ch = arr_np.shape[0]
            flat = arr_np.reshape(out_ch, -1)
            _set_nested_dict(jax_params, ["img_in_proj", "kernel"], flat.T)
            hidden_size = out_ch
            continue
        if pt_key == "img_in.proj.bias":
            _set_nested_dict(jax_params, ["img_in_proj", "bias"], pt_tensor_to_numpy(arr))
            continue

        if pt_key in ("time_in.mlp.0.weight", "time_in.mlp.0.bias",
                      "time_in.mlp.2.weight", "time_in.mlp.2.bias"):
            idx = pt_key.split(".")[2]
            sub = "mlp_0" if idx == "0" else "mlp_2"
            _lin(jax_params, ["time_in", sub], pt_key, arr)
            continue
        if pt_key in ("guidance_in.mlp.0.weight", "guidance_in.mlp.0.bias",
                      "guidance_in.mlp.2.weight", "guidance_in.mlp.2.bias"):
            idx = pt_key.split(".")[2]
            sub = "mlp_0" if idx == "0" else "mlp_2"
            _lin(jax_params, ["guidance_in", sub], pt_key, arr)
            continue
        if pt_key.startswith("vector_in.in_layer."):
            _lin(jax_params, ["vector_in", "in_layer"], pt_key, arr)
            continue
        if pt_key.startswith("vector_in.out_layer."):
            _lin(jax_params, ["vector_in", "out_layer"], pt_key, arr)
            continue

        m = _DOUBLE_QKV_RE.match(pt_key)
        if m:
            i, stream, kind = m.groups()
            hs = hidden_size or (arr.shape[0] // 3)
            chunks = _split_fused_linear(pt_tensor_to_numpy(arr), [hs, hs, hs], axis=0)
            for name, chunk in zip(("attn_q", "attn_k", "attn_v"), chunks):
                _set_nested_dict(
                    jax_params, ["double_blocks_" + i, f"{stream}_attn", name,
                                 "kernel" if kind == "weight" else "bias"],
                    chunk.T if kind == "weight" else chunk)
            continue
        m = _DOUBLE_QK_NORM_RE.match(pt_key)
        if m:
            i, stream, qk = m.groups()
            _rms(jax_params, ["double_blocks_" + i, f"{stream}_attn", f"attn_{qk}_norm"], arr)
            continue
        m = _DOUBLE_PROJ_RE.match(pt_key)
        if m:
            i, stream, kind = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_attn_proj"], pt_key, arr)
            continue
        m = _DOUBLE_MOD_RE.match(pt_key)
        if m:
            i, stream, kind = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_mod", "linear"], pt_key, arr)
            continue
        m = _DOUBLE_MLP_RE.match(pt_key)
        if m:
            i, stream, fc, kind = m.groups()
            _lin(jax_params, ["double_blocks_" + i, f"{stream}_mlp", fc], pt_key, arr)
            continue
        # img_norm1/2, txt_norm1/2: no learnable affine (elementwise_affine=False
        # in the reference), so no state_dict keys exist for them at all --
        # nothing to map, matching `common/dit_layers.py`'s `use_bias=False,
        # use_scale=False` LayerNorm calls (no params).

        m = _SINGLE_LINEAR1_RE.match(pt_key)
        if m:
            i, kind = m.groups()
            hs = hidden_size
            arr_np = pt_tensor_to_numpy(arr)
            mlp_hidden = arr_np.shape[0] - 3 * hs
            chunks = _split_fused_linear(arr_np, [hs, hs, hs, mlp_hidden], axis=0)
            for name, chunk in zip(("linear1_q", "linear1_k", "linear1_v", "linear1_mlp"), chunks):
                _set_nested_dict(
                    jax_params, ["single_blocks_" + i, name, "kernel" if kind == "weight" else "bias"],
                    chunk.T if kind == "weight" else chunk)
            continue
        m = _SINGLE_QK_NORM_RE.match(pt_key)
        if m:
            i, qk = m.groups()
            _rms(jax_params, ["single_blocks_" + i, f"{qk}_norm"], arr)
            continue
        m = _SINGLE_LINEAR2_RE.match(pt_key)
        if m:
            i, kind = m.groups()
            # No split needed: linear2's input is already [attn, mlp_act]
            # concatenated (hidden+mlp_hidden -> hidden), matching
            # `common/dit_layers.py:MMSingleStreamBlock`'s `linear2` shape
            # exactly.
            _lin(jax_params, ["single_blocks_" + i, "linear2"], pt_key, arr)
            continue
        m = _SINGLE_MOD_RE.match(pt_key)
        if m:
            i, kind = m.groups()
            _lin(jax_params, ["single_blocks_" + i, "modulation", "linear"], pt_key, arr)
            continue
        m = _SINGLE_PRENORM_RE.match(pt_key)
        if m:
            # elementwise_affine=False in the reference -- no params, skip
            # (present here only so the loop's fallthrough doesn't warn).
            continue

        m = _REFINER_BLOCK_NORM_RE.match(pt_key)
        if m:
            i, norm, kind = m.groups()
            _ln(jax_params, ["txt_in", f"blocks_{i}", norm], pt_key, arr)
            continue
        m = _REFINER_QKV_RE.match(pt_key)
        if m:
            # No split needed: common/dit_layers.py's
            # IndividualTokenRefinerBlock already uses one fused
            # `self_attn_qkv` Dense (matching the reference's own fused
            # `self_attn_qkv` Linear directly, unlike the main double/
            # single-stream blocks) -- copy straight across.
            i, kind = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "self_attn_qkv"], pt_key, arr)
            continue
        m = _REFINER_PROJ_RE.match(pt_key)
        if m:
            i, kind = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "self_attn_proj"], pt_key, arr)
            continue
        m = _REFINER_MLP_RE.match(pt_key)
        if m:
            i, fc, kind = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "mlp", f"fc{fc}"], pt_key, arr)
            continue
        m = _REFINER_ADALN_RE.match(pt_key)
        if m:
            i, kind = m.groups()
            _lin(jax_params, ["txt_in", f"blocks_{i}", "adaLN_modulation_1"], pt_key, arr)
            continue
        if pt_key.startswith("txt_in.input_embedder."):
            _lin(jax_params, ["txt_in", "input_embedder"], pt_key, arr)
            continue
        if pt_key.startswith("txt_in.t_embedder.mlp."):
            idx = pt_key.split(".")[3]
            sub = "mlp_0" if idx == "0" else "mlp_2"
            _lin(jax_params, ["txt_in", "t_embedder", sub], pt_key, arr)
            continue
        if pt_key.startswith("txt_in.c_embedder.linear_1."):
            _lin(jax_params, ["txt_in", "c_embedder_linear_1"], pt_key, arr)
            continue
        if pt_key.startswith("txt_in.c_embedder.linear_2."):
            _lin(jax_params, ["txt_in", "c_embedder_linear_2"], pt_key, arr)
            continue

        m = _FINAL_ADALN_RE.match(pt_key)
        if m:
            (kind,) = m.groups()
            _lin(jax_params, ["final_layer", "adaLN_modulation_1"], pt_key, arr)
            continue
        if pt_key.startswith("final_layer.linear."):
            _lin(jax_params, ["final_layer", "linear"], pt_key, arr)
            continue
        # final_layer.norm_final: elementwise_affine=False, no params.

        # Unrecognized key (e.g. an EMA/optimizer-state key, or a real
        # checkpoint layout difference this skeleton didn't anticipate) --
        # left unmapped rather than silently guessed at.

    return {"params": jax_params}


# ---------------------------------------------------------------------------
# VAE ("884-16c-hy", `AutoencoderKLCausal3D`)
# ---------------------------------------------------------------------------

def _conv(jax_params: dict, path: list, pt_key: str, arr) -> None:
    """`nn.Conv` (used for every causal conv *and* the 1x1 GroupNorm-free
    shortcut/quant convs in `hunyuan_video_1_0.vae`) -- Flax param name
    `kernel`, not `weight`. `convert_pt_tensor_to_jax` already transposes
    5D (Conv3d) arrays to Flax's (T,H,W,In,Out) layout.
    """
    leaf = "bias" if pt_key.endswith(".bias") else "kernel"
    _set_nested_dict(jax_params, path + [leaf], convert_pt_tensor_to_jax(pt_key, arr))


def _gn(jax_params: dict, path: list, pt_key: str, arr) -> None:
    """`nn.GroupNorm`'s Flax param name is `scale`, not `weight` (same
    `weight`/`bias` -> `scale`/`bias` rename as `_ln`, for GroupNorm instead
    of LayerNorm)."""
    leaf = "bias" if pt_key.endswith(".bias") else "scale"
    _set_nested_dict(jax_params, path + [leaf], convert_pt_tensor_to_jax(pt_key, arr))


_VAE_LEVEL_RESNET_RE = re.compile(
    r"^(encoder|decoder)\.(down_blocks|up_blocks)\.(\d+)\.resnets\.(\d+)\."
    r"(norm1|norm2|conv1|conv2|conv_shortcut)(?:\.conv)?\.(weight|bias)$")
_VAE_MID_RESNET_RE = re.compile(
    r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\."
    r"(norm1|norm2|conv1|conv2|conv_shortcut)(?:\.conv)?\.(weight|bias)$")
_VAE_DOWNSAMPLE_RE = re.compile(
    r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.conv\.(weight|bias)$")
_VAE_UPSAMPLE_RE = re.compile(
    r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.conv\.(weight|bias)$")
_VAE_ATTN_RE = re.compile(
    r"^(encoder|decoder)\.mid_block\.attentions\.0\.(group_norm|to_q|to_k|to_v|to_out\.0)\.(weight|bias)$")
_VAE_CONV_IN_OUT_RE = re.compile(r"^(encoder|decoder)\.(conv_in|conv_out)\.conv\.(weight|bias)$")
_VAE_NORM_OUT_RE = re.compile(r"^(encoder|decoder)\.conv_norm_out\.(weight|bias)$")
_VAE_QUANT_CONV_RE = re.compile(r"^(quant_conv|post_quant_conv)\.(weight|bias)$")


def map_hunyuan_video_1_0_vae_keys(pt_state_dict: Dict) -> Dict:
    """`AutoencoderKLCausal3D`'s real `pytorch_model.pt` (248 leaves,
    cross-checked directly against `tencent/HunyuanVideo/hunyuan-video-
    t2v-720p/vae/pytorch_model.pt`) -> `vidax.models.hunyuan_video.
    hunyuan_video_1_0.vae.HunyuanVideo10VAE`'s param tree. See that module's
    docstring for the encoder/decoder/mid-block naming correspondence this
    mapper relies on (e.g. `encoder.down_blocks.I.resnets.J.*` ->
    `["encoder", f"down_blocks_{I}_resnets_{J}", ...]`).
    """
    jax_params: Dict = {}
    for pt_key, arr in pt_state_dict.items():
        m = _VAE_LEVEL_RESNET_RE.match(pt_key)
        if m:
            enc_dec, block_kind, i_level, i_block, sub, kind = m.groups()
            name = f"{block_kind}_{i_level}_resnets_{i_block}"
            if sub in ("norm1", "norm2"):
                _gn(jax_params, [enc_dec, name, sub], pt_key, arr)
            else:
                _conv(jax_params, [enc_dec, name, sub], pt_key, arr)
            continue
        m = _VAE_MID_RESNET_RE.match(pt_key)
        if m:
            enc_dec, i_block, sub, kind = m.groups()
            name = f"mid_block_resnets_{i_block}"
            if sub in ("norm1", "norm2"):
                _gn(jax_params, [enc_dec, name, sub], pt_key, arr)
            else:
                _conv(jax_params, [enc_dec, name, sub], pt_key, arr)
            continue
        m = _VAE_DOWNSAMPLE_RE.match(pt_key)
        if m:
            i_level, kind = m.groups()
            _conv(jax_params, ["encoder", f"down_blocks_{i_level}_downsamplers_0"], pt_key, arr)
            continue
        m = _VAE_UPSAMPLE_RE.match(pt_key)
        if m:
            i_level, kind = m.groups()
            _conv(jax_params, ["decoder", f"up_blocks_{i_level}_upsamplers_0_conv"], pt_key, arr)
            continue
        m = _VAE_ATTN_RE.match(pt_key)
        if m:
            enc_dec, sub, kind = m.groups()
            attn_name = f"mid_block_attentions_0"
            if sub == "group_norm":
                _gn(jax_params, [enc_dec, attn_name, "group_norm"], pt_key, arr)
            else:
                dense_name = {"to_q": "to_q", "to_k": "to_k", "to_v": "to_v", "to_out.0": "to_out_0"}[sub]
                _lin(jax_params, [enc_dec, attn_name, dense_name], pt_key, arr)
            continue
        m = _VAE_CONV_IN_OUT_RE.match(pt_key)
        if m:
            enc_dec, which, kind = m.groups()
            _conv(jax_params, [enc_dec, which], pt_key, arr)
            continue
        m = _VAE_NORM_OUT_RE.match(pt_key)
        if m:
            enc_dec, kind = m.groups()
            _gn(jax_params, [enc_dec, "conv_norm_out"], pt_key, arr)
            continue
        m = _VAE_QUANT_CONV_RE.match(pt_key)
        if m:
            which, kind = m.groups()
            _conv(jax_params, [which], pt_key, arr)
            continue
        # Unrecognized key -- left unmapped rather than silently guessed at.

    return {"params": jax_params}


# ---------------------------------------------------------------------------
# Llama text tower (`xtuner/llava-llama-3-8b-v1_1-transformers`'s extracted
# `.language_model`, plain `LlamaModel`)
# ---------------------------------------------------------------------------

_LLAMA_LAYER_RE = re.compile(
    r"^layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$")
_LLAMA_ATTN_RE = re.compile(r"^layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$")
_LLAMA_MLP_RE = re.compile(r"^layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")


def _rmsnorm(jax_params: dict, path: list, arr) -> None:
    _set_nested_dict(jax_params, path + ["scale"], convert_pt_tensor_to_jax("weight", arr))


def map_hunyuan_video_1_0_llama_text_keys(pt_state_dict: Dict) -> Dict:
    """The extracted `xtuner/llava-llama-3-8b-v1_1-transformers`'s
    `.language_model` (plain HF `LlamaModel`, 290 leaves, no `model.`
    prefix -- see `hunyuan_video_1_0.llama_text`'s module docstring) -> that
    module's `LlamaTextModel` param tree. All Dense layers are bias-free
    (`attention_bias=False`, `mlp_bias=False`), confirmed against the real
    checkpoint (no `*.bias` keys for any q/k/v/o/gate/up/down proj).
    """
    jax_params: Dict = {}
    for pt_key, arr in pt_state_dict.items():
        if pt_key == "embed_tokens.weight":
            # nn.Embed's weight is already (vocab, hidden) in PyTorch, same
            # layout Flax expects -- unlike a Linear, no transpose.
            _set_nested_dict(jax_params, ["embed_tokens", "embedding"], pt_tensor_to_numpy(arr))
            continue
        if pt_key == "norm.weight":
            _rmsnorm(jax_params, ["norm"], arr)
            continue
        m = _LLAMA_LAYER_RE.match(pt_key)
        if m:
            i, sub = m.groups()
            _rmsnorm(jax_params, [f"layers_{i}", sub], arr)
            continue
        m = _LLAMA_ATTN_RE.match(pt_key)
        if m:
            i, which = m.groups()
            name = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}[which]
            _set_nested_dict(
                jax_params, [f"layers_{i}", "self_attn", name, "kernel"], convert_pt_tensor_to_jax(pt_key, arr))
            continue
        m = _LLAMA_MLP_RE.match(pt_key)
        if m:
            i, which = m.groups()
            name = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[which]
            _set_nested_dict(
                jax_params, [f"layers_{i}", "mlp", name, "kernel"], convert_pt_tensor_to_jax(pt_key, arr))
            continue
        # Unrecognized key -- left unmapped rather than silently guessed at.

    return {"params": jax_params}


# ---------------------------------------------------------------------------
# CLIP-L pooled text encoder (`openai/clip-vit-large-patch14`'s `text_model`)
# ---------------------------------------------------------------------------

_CLIP_LAYER_NORM_RE = re.compile(r"^text_model\.encoder\.layers\.(\d+)\.(layer_norm1|layer_norm2)\.(weight|bias)$")
_CLIP_ATTN_RE = re.compile(
    r"^text_model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$")
_CLIP_MLP_RE = re.compile(r"^text_model\.encoder\.layers\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$")


def map_hunyuan_video_1_0_clip_text_keys(pt_state_dict: Dict) -> Dict:
    """`openai/clip-vit-large-patch14`'s `text_model.*` (197 leaves,
    pooled-output-only -- no `vision_model`/`visual_projection`/`logit_scale`
    keys are mapped, since only the text tower's pooled output is ever used,
    see `hunyuan_video_1_0.clip_text`'s module docstring) -> that module's
    `ClipTextModel` param tree.
    """
    jax_params: Dict = {}
    for pt_key, arr in pt_state_dict.items():
        if pt_key == "text_model.embeddings.token_embedding.weight":
            _set_nested_dict(
                jax_params, ["embeddings_token_embedding", "embedding"], pt_tensor_to_numpy(arr))
            continue
        if pt_key == "text_model.embeddings.position_embedding.weight":
            _set_nested_dict(
                jax_params, ["embeddings_position_embedding", "embedding"], pt_tensor_to_numpy(arr))
            continue
        if pt_key == "text_model.final_layer_norm.weight" or pt_key == "text_model.final_layer_norm.bias":
            _ln(jax_params, ["final_layer_norm"], pt_key, arr)
            continue
        m = _CLIP_LAYER_NORM_RE.match(pt_key)
        if m:
            i, sub, kind = m.groups()
            _ln(jax_params, [f"layers_{i}", sub], pt_key, arr)
            continue
        m = _CLIP_ATTN_RE.match(pt_key)
        if m:
            i, sub, kind = m.groups()
            _lin(jax_params, [f"layers_{i}", "self_attn", sub], pt_key, arr)
            continue
        m = _CLIP_MLP_RE.match(pt_key)
        if m:
            i, sub, kind = m.groups()
            _lin(jax_params, [f"layers_{i}", "mlp", sub], pt_key, arr)
            continue
        # Unrecognized key (e.g. vision_model.*/visual_projection.*/
        # logit_scale/position_ids buffer) -- left unmapped, matches
        # ClipTextModel's text-tower-only scope.

    return {"params": jax_params}
