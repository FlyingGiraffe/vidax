"""Checkpoint-metadata loaders and kwargs builders for HunyuanVideo-1.5.

Unlike ``ltx2_5``'s safetensors-embedded metadata, HunyuanVideo-1.5 ships
standard HF-diffusers-style ``config.json`` files per subfolder
(``transformer/<variant>/config.json``, ``vae/config.json``,
``scheduler/scheduler_config.json``) -- read directly from disk, not from
safetensors metadata.

Real values transcribed here (2025-08, ``tencent/HunyuanVideo-1.5`` on the
Hub, all 4 core T2V/I2V variants -- confirmed identical across all 4):
  hidden_size=2048, heads_num=16 (head_dim=128), mm_double_blocks_depth=54,
  mm_single_blocks_depth=0 (**no single-stream blocks in these checkpoints
  -- do not assume the reference ctor's default 20/40 split**), patch_size
  =[1,1,1] (no additional DiT-side patchify beyond the VAE's own
  compression), in_channels=out_channels=32 (== VAE latent_channels;
  ``concat_condition=True`` doubles this to 65 *inside* ``PatchEmbed``, see
  ``dit.py``), text_states_dim=3584 (Qwen2.5-VL-7B), vision_states_dim=1152
  (SigLIP), rope_theta=256, rope_dim_list=[16,56,56], guidance_embed=False,
  text_pool_type=None (no secondary pooled-text vector), glyph_byT5_v2=True,
  use_cond_type_embedding=True, vision_projection="linear",
  text_projection="single_refiner". Only ``ideal_resolution``/
  ``ideal_task`` differ per variant (metadata only, not consumed by the
  model itself -- used here to pick default ``shift``).
"""
import json
import os
from typing import Any, Dict

# Per-checkpoint default flow-match `shift`, from
# `hyvideo/commons/__init__.py`'s `PIPELINE_CONFIGS` in the reference --
# confirm against the downloaded `scheduler/scheduler_config.json` (shared
# across variants; the per-task shift is applied by the pipeline, not baked
# into the scheduler config) before trusting this table for a new release.
DEFAULT_SHIFT = {
    ("480p", "t2v"): 5.0,
    ("480p", "i2v"): 5.0,
    ("720p", "t2v"): 9.0,
    ("720p", "i2v"): 7.0,
}


def load_hunyuan_video_1_5_transformer_config(transformer_dir: str) -> Dict[str, Any]:
    """Reads ``<transformer_dir>/config.json`` (e.g. ``.../transformer/480p_t2v``)."""
    with open(os.path.join(transformer_dir, "config.json")) as f:
        return json.load(f)


def load_hunyuan_video_1_5_vae_config(vae_dir: str) -> Dict[str, Any]:
    """Reads ``<vae_dir>/config.json`` (e.g. ``.../vae``)."""
    with open(os.path.join(vae_dir, "config.json")) as f:
        return json.load(f)


def dit_kwargs_from_transformer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Builds ``HunyuanVideo15DiT`` constructor kwargs from a real ``config.json``.

    Every field is read from the config dict directly (required fields via
    ``config[...]``, optional/rarely-varying ones via ``config.get(...,
    default)`` with the default matching the reference ctor's own default)
    -- never hardcoded, per this repo's standing discipline.
    """
    patch_size = tuple(config.get("patch_size", [1, 2, 2]))
    rope_dim_list = tuple(config.get("rope_dim_list", [16, 56, 56]))
    return dict(
        patch_size=patch_size,
        in_channels=config["in_channels"],
        out_channels=config.get("out_channels") or config["in_channels"],
        concat_condition=config.get("concat_condition", True),
        is_reshape_temporal_channels=config.get("is_reshape_temporal_channels", False),
        hidden_size=config["hidden_size"],
        heads_num=config["heads_num"],
        mlp_width_ratio=config.get("mlp_width_ratio", 4.0),
        mlp_act_type=config.get("mlp_act_type", "gelu_tanh"),
        mm_double_blocks_depth=config["mm_double_blocks_depth"],
        mm_single_blocks_depth=config["mm_single_blocks_depth"],
        rope_dim_list=rope_dim_list,
        rope_theta=config.get("rope_theta", 256),
        qkv_bias=config.get("qkv_bias", True),
        qk_norm=config.get("qk_norm", True),
        guidance_embed=config.get("guidance_embed", False),
        text_projection=config.get("text_projection", "single_refiner"),
        use_attention_mask=config.get("use_attention_mask", True),
        text_states_dim=config["text_states_dim"],
        text_pool_type=config.get("text_pool_type"),
        text_states_dim_2=config.get("text_states_dim_2"),
        glyph_byT5_v2=config.get("glyph_byT5_v2", False),
        vision_projection=config.get("vision_projection", "none"),
        vision_states_dim=config.get("vision_states_dim", 1280),
        use_cond_type_embedding=config.get("use_cond_type_embedding", False),
    )


def default_shift_for(resolution: str, task: str) -> float:
    """Per-(resolution, task) flow-match shift, from ``PIPELINE_CONFIGS``."""
    return DEFAULT_SHIFT[(resolution, task)]


def vae_kwargs_from_vae_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Builds ``HunyuanVideo15VAE`` constructor kwargs from ``vae/config.json``."""
    return dict(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        ffactor_spatial=config["ffactor_spatial"],
        ffactor_temporal=config["ffactor_temporal"],
        downsample_match_channel=config.get("downsample_match_channel", True),
        upsample_match_channel=config.get("upsample_match_channel", True),
    )
