"""Named `CogVideoXDiT` / `CogVideoXVAE` hyperparameter presets for every
released CogVideoX checkpoint (THUDM / ZhipuAI).

Hand-written dicts, cross-checked against each HuggingFace repo's
`transformer/config.json` and `vae/config.json` (the Wan-style approach --
CogVideoX's diffusers checkpoints do *not* embed their own architecture
config as file metadata the way LTX's single-file checkpoints do).

CogVideoX 1.0 and 1.5 share a single transformer class in diffusers
(`CogVideoXTransformer3DModel`); 1.5 only toggles `patch_size_t` (temporal
patchifying) and the RoPE grid type (`"slice"` instead of `"linspace"`), so
one config-driven `dit.py` covers all five variants. The VAE
(`AutoencoderKLCogVideoX`) is byte-for-byte the same architecture across
every variant -- only the per-checkpoint `scaling_factor` /
`invert_scale_latents` differ, and those live on the preset, not in
`VAE_CONFIG`.

Invariants hardcoded into `dit.py` / `vae.py` (true of every checkpoint, not
exposed as fields): `attention_head_dim=64`, `text_embed_dim=4096`,
`time_embed_dim=512`, `max_text_seq_length=226`, `patch_size=2`,
`attention_bias=True`, `timestep_activation_fn="silu"`,
`activation_fn="gelu-approximate"`, `norm_elementwise_affine=True`,
`norm_eps=1e-5` (DiT) / `1e-6` (VAE), `temporal_compression_ratio=4`,
`flip_sin_to_cos=True`, `freq_shift=0`, `spatial_interpolation_scale=1.875`,
`temporal_interpolation_scale=1.0`.
"""

# Recommended (width, height) per checkpoint -- from
# refs/CogVideo-main/inference/cli_demo.py's RESOLUTION_MAP.
RESOLUTION_MAP = {
    "2b": (720, 480),
    "5b": (720, 480),
    "5b-i2v": (720, 480),
    "1.5-5b": (1360, 768),
    "1.5-5b-i2v": (1360, 768),
}

# `AutoencoderKLCogVideoX` -- identical for every released checkpoint.
VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    block_out_channels=(128, 256, 256, 512),
    latent_channels=16,
    layers_per_block=3,
    norm_num_groups=32,
    norm_eps=1e-6,
    temporal_compression_ratio=4,
)

# --- DiT presets -------------------------------------------------------------
# `snr_shift_scale`, `scaling_factor`, `invert_scale_latents` are consumed by
# the scheduler / example script, not `CogVideoXDiT`.

CONFIG_2B = dict(
    num_layers=30,
    num_attention_heads=30,
    in_channels=16,
    out_channels=16,
    use_rotary_positional_embeddings=False,
    use_learned_positional_embeddings=False,  # -> pos_embedding recomputed, not in checkpoint
    patch_size_t=None,
    patch_bias=True,
    ofs_embed_dim=None,
    sample_width=90,
    sample_height=60,
    sample_frames=49,
    # scheduler / latent-scale
    snr_shift_scale=3.0,
    scaling_factor=1.15258426,
    invert_scale_latents=False,
)

CONFIG_5B = dict(
    num_layers=42,
    num_attention_heads=48,
    in_channels=16,
    out_channels=16,
    use_rotary_positional_embeddings=True,
    use_learned_positional_embeddings=False,
    patch_size_t=None,
    patch_bias=True,
    ofs_embed_dim=None,
    sample_width=90,
    sample_height=60,
    sample_frames=49,
    snr_shift_scale=1.0,
    scaling_factor=0.7,
    invert_scale_latents=False,
)

CONFIG_5B_I2V = dict(
    num_layers=42,
    num_attention_heads=48,
    in_channels=32,  # image latent concatenated on the channel axis
    out_channels=16,
    use_rotary_positional_embeddings=True,
    use_learned_positional_embeddings=True,  # persistent pos_embedding buffer in the checkpoint
    patch_size_t=None,
    patch_bias=True,
    ofs_embed_dim=None,
    sample_width=90,
    sample_height=60,
    sample_frames=49,
    snr_shift_scale=1.0,
    scaling_factor=0.7,
    invert_scale_latents=False,
)

CONFIG_1_5_5B = dict(
    num_layers=42,
    num_attention_heads=48,
    in_channels=16,
    out_channels=16,
    use_rotary_positional_embeddings=True,  # "slice" grid (see dit.rope)
    use_learned_positional_embeddings=False,
    patch_size_t=2,
    patch_bias=False,
    ofs_embed_dim=None,
    sample_width=170,
    sample_height=96,
    sample_frames=81,
    snr_shift_scale=1.0,
    scaling_factor=0.7,
    invert_scale_latents=True,
)

CONFIG_1_5_5B_I2V = dict(
    num_layers=42,
    num_attention_heads=48,
    in_channels=32,
    out_channels=16,
    use_rotary_positional_embeddings=True,
    use_learned_positional_embeddings=False,
    patch_size_t=2,
    patch_bias=False,
    ofs_embed_dim=512,  # "ofs" (offset) embedding, added to the timestep embedding
    sample_width=170,
    sample_height=96,
    sample_frames=81,
    snr_shift_scale=1.0,
    scaling_factor=0.7,
    invert_scale_latents=True,
)

CONFIGS = {
    "2b": CONFIG_2B,
    "5b": CONFIG_5B,
    "5b-i2v": CONFIG_5B_I2V,
    "1.5-5b": CONFIG_1_5_5B,
    "1.5-5b-i2v": CONFIG_1_5_5B_I2V,
}

# Fields of a preset that are DiT constructor kwargs (everything else is
# scheduler / pipeline glue).
_DIT_FIELDS = (
    "num_layers", "num_attention_heads", "in_channels", "out_channels",
    "use_rotary_positional_embeddings", "use_learned_positional_embeddings",
    "patch_size_t", "patch_bias", "ofs_embed_dim",
    "sample_width", "sample_height", "sample_frames",
)


def dit_kwargs(variant: str) -> dict:
    """The subset of a preset that `CogVideoXDiT(**...)` accepts."""
    cfg = CONFIGS[variant]
    return {k: cfg[k] for k in _DIT_FIELDS}
