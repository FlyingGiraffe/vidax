"""Named `LTXDiT`/`LTXVAE` hyperparameter presets for LTX-Video 0.9.8's
released checkpoints.

Unlike Wan's presets (plain hand-written dicts cross-checked against each
checkpoint's own `config.json`), LTX-Video's single-file `.safetensors`
checkpoints embed their own architecture config directly as file metadata
(`safetensors.safe_open(path).metadata()["config"]`, a JSON blob with
`"transformer"`/`"vae"`/`"scheduler"` sections -- see `Transformer3DModel.
from_pretrained`'s single-file branch in `refs/LTX-Video-main/ltx_video/
models/transformers/transformer3d.py`). `load_ltx_checkpoint_metadata`
reads that directly, so there's no hardcoded per-variant dict to fall out of
sync -- these two named dicts are just documentation of the dims actually
read from the two checkpoints this repo has downloaded (`ltxv-13b-0.9.8-*`,
`ltxv-2b-0.9.8-distilled`), for reference/tests.
"""
import json

import safetensors

# `qk_norm="rms_norm"`, `standardization_norm="rms_norm"`,
# `norm_elementwise_affine=False`, `activation_fn="gelu-approximate"`, and
# `attention_bias=True` are true of every released 0.9.8 checkpoint (2B and
# 13B alike) and are hardcoded into `vidax.models.ltx_video.dit.LTXDiT`
# rather than exposed as fields -- only the dims below actually vary.
DIT_13B_CONFIG = dict(
    num_attention_heads=32, attention_head_dim=128, in_channels=128, out_channels=128,
    num_layers=48, cross_attention_dim=4096, caption_channels=4096,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=(20, 2048, 2048), timestep_scale_multiplier=1000,
)

DIT_2B_CONFIG = dict(
    num_attention_heads=32, attention_head_dim=64, in_channels=128, out_channels=128,
    num_layers=28, cross_attention_dim=2048, caption_channels=4096,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=(20, 2048, 2048), timestep_scale_multiplier=1000,
)

# Both released variants (2B and 13B) share the same VAE architecture --
# only the DiT scales.
VAE_CONFIG = dict(
    latent_channels=128,
    encoder_blocks=(
        ("res_x", {"num_layers": 4}),
        ("compress_space_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_time_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_all_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
        ("compress_all_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
    ),
    decoder_blocks=(
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
    ),
    patch_size=4, norm_layer="pixel_norm", latent_log_var="uniform",
    timestep_conditioning=True, base_channels=128,
)


def load_ltx_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Reads the `{"transformer": ..., "vae": ..., "scheduler": ...}` config
    blob embedded as safetensors file metadata in a released LTX-Video
    checkpoint -- the authoritative source of architecture hyperparameters
    (see this module's docstring).
    """
    with safetensors.safe_open(checkpoint_path, framework="numpy") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def load_ltx_vae_per_channel_stats(checkpoint_path: str):
    """Reads the VAE's `mean-of-means`/`std-of-means` per-channel latent
    statistics (shape `(latent_channels,)` each), used by the *pipeline*
    (not `LTXVAE` itself -- see `examples/generate_ltx_video.py`) to
    normalize latents into the space the DiT was trained on before
    sampling, and un-normalize them back before VAE decode
    (`vae_encode.normalize_latents`/`un_normalize_latents` in the
    reference; `vidax.translator.mappings.ltx_video.map_ltx_video_vae_keys`
    deliberately skips these two keys since they're not `LTXVAE` model
    parameters).

    Returns:
        (mean, std), each a `(latent_channels,)` numpy array.
    """
    import numpy as np
    with safetensors.safe_open(checkpoint_path, framework="numpy") as f:
        mean = np.array(f.get_tensor("vae.per_channel_statistics.mean-of-means"))
        std = np.array(f.get_tensor("vae.per_channel_statistics.std-of-means"))
    return mean, std


def vae_scale_factors(vae_config: dict) -> "tuple[int, int]":
    """Computes `(temporal_downscale_factor, spatial_downscale_factor)` from
    an encoder-blocks list the same way the reference's
    `CausalVideoAutoencoder.spatial_downscale_factor`/
    `temporal_downscale_factor` properties do, generalized beyond the two
    specific released configs (rather than hardcoding `(8, 32)`).
    """
    encoder_blocks = vae_config["encoder_blocks"]
    spatial_compress_names = ("compress_space_res", "compress_all_res", "compress_space", "compress_all")
    temporal_compress_names = ("compress_time_res", "compress_all_res", "compress_time", "compress_all")
    num_spatial = sum(1 for name, _ in encoder_blocks if name in spatial_compress_names)
    num_temporal = sum(1 for name, _ in encoder_blocks if name in temporal_compress_names)
    spatial_scale = (2 ** num_spatial) * vae_config["patch_size"]
    temporal_scale = 2 ** num_temporal
    return temporal_scale, spatial_scale
