"""Named hyperparameter presets for HunyuanVideo 1.0 (T2V only).

Unlike HunyuanVideo-1.5, this repo's own `refs/HunyuanVideo-main/` ships no
downloaded-checkpoint `config.json` to read from -- its hyperparameters live
embedded in Python source (`hyvideo/modules/models.py`'s
`HUNYUAN_VIDEO_CONFIG` dict, `hyvideo/config.py`'s argparse defaults,
`hyvideo/constants.py`). This module transcribes those directly (values
confirmed by reading the reference source, cited inline), matching this
repo's `wan2_1.py`/`wan2_2.py`-style *named preset* pattern rather than
`ltx2_5`'s / `hunyuan_video1_5`'s checkpoint-embedded-metadata pattern --
revisit once a real `tencent/HunyuanVideo` checkpoint is downloaded (it may
ship its own `config.json`/`args.json` inside the `mp_rank_00_model_states.pt`
that should take precedence over these transcribed defaults; not checked
this pass, see docs/lessons/hunyuan_video_debugging.md).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class HunyuanVideoDiTConfig:
    """Mirrors `HYVideoDiffusionTransformer.__init__`'s real defaults
    (`hyvideo/modules/models.py`) for the `"HYVideo-T/2"` (real CFG) /
    `"HYVideo-T/2-cfgdistill"` (`guidance_embed=True`) presets -- both share
    every field except `guidance_embed`.
    """
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16  # VAE.config.latent_channels for "884-16c-hy".
    out_channels: Optional[int] = None  # None -> in_channels, per reference.
    hidden_size: int = 3072
    heads_num: int = 24
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    mm_double_blocks_depth: int = 20
    mm_single_blocks_depth: int = 40
    rope_dim_list: Tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0  # config.py's `--rope-theta` CLI default.
    qkv_bias: bool = True
    qk_norm: bool = True
    guidance_embed: bool = False
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True
    text_states_dim: int = 4096  # "llm" text encoder hidden size.
    text_states_dim_2: int = 768  # "clipL" pooled hidden size.


# "HYVideo-T/2": real classifier-free guidance, no embedded/distilled guidance.
HYVIDEO_T2_CONFIG = HunyuanVideoDiTConfig(guidance_embed=False)

# "HYVideo-T/2-cfgdistill": the config.py argparse default and the most
# widely released T2V checkpoint variant -- embedded/distilled guidance.
HYVIDEO_T2_CFGDISTILL_CONFIG = HunyuanVideoDiTConfig(guidance_embed=True)

DIT_CONFIGS = {
    "HYVideo-T/2": HYVIDEO_T2_CONFIG,
    "HYVideo-T/2-cfgdistill": HYVIDEO_T2_CFGDISTILL_CONFIG,
}

# `config.py`'s `--flow-shift` default -- one value (unlike HunyuanVideo-1.5's
# per-resolution/task table), since this batch covers one T2V resolution class.
DEFAULT_SHIFT = 7.0


def dit_kwargs_from_config(config: HunyuanVideoDiTConfig) -> dict:
    """Builds `HunyuanVideoDiT` constructor kwargs from a named preset."""
    return dict(
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels or config.in_channels,
        hidden_size=config.hidden_size,
        heads_num=config.heads_num,
        mlp_width_ratio=config.mlp_width_ratio,
        mlp_act_type=config.mlp_act_type,
        mm_double_blocks_depth=config.mm_double_blocks_depth,
        mm_single_blocks_depth=config.mm_single_blocks_depth,
        rope_dim_list=config.rope_dim_list,
        rope_theta=config.rope_theta,
        qkv_bias=config.qkv_bias,
        qk_norm=config.qk_norm,
        guidance_embed=config.guidance_embed,
        text_projection=config.text_projection,
        use_attention_mask=config.use_attention_mask,
        text_states_dim=config.text_states_dim,
        text_states_dim_2=config.text_states_dim_2,
    )


# VAE ("884-16c-hy"): 8x8 spatial / 4x temporal compression, 16 latent
# channels -- from `hyvideo/constants.py`'s `VAE_PATH` naming convention
# (`"<down><down><down>-<latent_channels>c-hy"`, read literally: 8x8x4).
# Real `block_out_channels`/`layers_per_block`/etc. are read from the
# downloaded checkpoint's own `vae/config.json` via
# `load_hunyuan_video_vae_config`/`vae_kwargs_from_vae_config` below (same
# pattern as `hunyuan_video1_5.configs`), not hardcoded here -- confirmed
# real values (2025-08, `tencent/HunyuanVideo`'s
# `hunyuan-video-t2v-720p/vae/config.json`): `block_out_channels=
# [128,256,512,512]`, `layers_per_block=2`, `latent_channels=16`,
# `norm_num_groups=32`, `in_channels=out_channels=3`,
# `time_compression_ratio=4`, `scaling_factor=0.476986`, no `shift_factor`
# key (unlike HunyuanVideo-1.5's VAE, this one never applies one). No
# `spatial_compression_ratio` key in the file itself -- the reference's own
# `AutoencoderKLCausal3D.__init__` default (8) applies, transcribed as
# `VAE_SPATIAL_COMPRESSION_RATIO` below since it's a ctor default, not a
# per-checkpoint config value.
VAE_FFACTOR_SPATIAL = 8
VAE_FFACTOR_TEMPORAL = 4
VAE_LATENT_CHANNELS = 16
VAE_SPATIAL_COMPRESSION_RATIO = 8  # AutoencoderKLCausal3D.__init__'s own default.


def load_hunyuan_video_vae_config(vae_dir: str) -> Dict[str, Any]:
    """Reads ``<vae_dir>/config.json`` (e.g. ``.../hunyuan-video-t2v-720p/vae``)."""
    with open(os.path.join(vae_dir, "config.json")) as f:
        return json.load(f)


def vae_kwargs_from_vae_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Builds ``HunyuanVideoVAE`` constructor kwargs from a real ``vae/config.json``."""
    return dict(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["latent_channels"],
        block_out_channels=tuple(config["block_out_channels"]),
        layers_per_block=config["layers_per_block"],
        spatial_compression_ratio=config.get("spatial_compression_ratio", VAE_SPATIAL_COMPRESSION_RATIO),
        time_compression_ratio=config["time_compression_ratio"],
        mid_block_add_attention=config.get("mid_block_add_attention", True),
    )
