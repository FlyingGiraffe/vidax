"""Named `LTXDiT`/`LTXVAE` hyperparameter presets for LTX-2.5's released
video-only checkpoints, and metadata/stats readers.

Same pattern as `vidax.models.ltx_video.configs`: LTX-2.5's DiT/VAE
checkpoints are single-file `.safetensors` with their architecture config
embedded as file metadata (`safetensors.safe_open(path).metadata()
["config"]`) -- the authoritative source, read directly by
`load_ltx2_5_dit_metadata`/`load_ltx2_5_vae_metadata` rather than hardcoded.
`VAE_CONFIG` below is a verified transcription of the real
`ltx-2.5-video-vae-conv-bf16.safetensors` checkpoint's own embedded
`config.vae` (read during this port), kept as documentation/test fixture
the same way `vidax.models.ltx_video.configs.VAE_CONFIG` is -- **not** used
to build the model at runtime.

`DIT_22B_CONFIG` is filled in from the real
`ltx-2.5-22b-{dev,distilled}-transformer-bf16.safetensors` checkpoints'
embedded metadata once downloaded (see this module's TODO once verified);
until then, load DiT config via `load_ltx2_5_dit_metadata` directly rather
than trusting this dict's dims.
"""
import json

import safetensors

# Read directly from `ltx-2.5-video-vae-conv-bf16.safetensors`'s embedded
# `config.vae` -- confirmed real, not assumed from the LTX-Video precedent.
# Two real deltas from `vidax.models.ltx_video.configs.VAE_CONFIG`:
# - `timestep_conditioning=False` here (LTX-Video's VAE is always
#   timestep-conditioned) -- `vidax.models.ltx2_5.vae` threads this as a
#   real conditional, not hardcoded like the LTX-Video port.
# - The block list itself differs slightly (one `res_x` layer count, and
#   the last `compress_all_res`/`compress_all` pair has `multiplier=1`,
#   i.e. spatial/temporal downsampling with no channel-width change at
#   that step) -- `vidax.models.ltx2_5.vae`'s Encoder/Decoder read
#   `multiplier`/`residual` from each block's own dict generically, same
#   as the LTX-Video port, so no code change is needed for this, only the
#   config values below.
VAE_CONFIG = dict(
    latent_channels=128,
    encoder_blocks=(
        ("res_x", {"num_layers": 4}),
        ("compress_space_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_time_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 4}),
        ("compress_all_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
        ("compress_all_res", {"multiplier": 1}),
        ("res_x", {"num_layers": 2}),
    ),
    decoder_blocks=(
        ("res_x", {"num_layers": 4}),
        ("compress_space", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_time", {"multiplier": 2}),
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 2}),
        ("compress_all", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
    ),
    patch_size=4, norm_layer="pixel_norm", latent_log_var="uniform",
    causal_decoder=False, timestep_conditioning=False, base_channels=128,
    spatial_padding_mode="zeros",
)


# Read directly from `ltx-2.5-22b-distilled-transformer-bf16.safetensors`'s
# embedded `config.transformer` -- documentation/test fixture only, same
# caveat as `VAE_CONFIG` above (load via `load_ltx2_5_metadata` at runtime,
# don't trust this dict). `model_type: LTXModelType.VideoOnly` drops every
# `audio_*` field entirely -- listed here for completeness/documentation
# only, `vidax.models.ltx2_5.dit.LTXDiT` never reads them.
DIT_22B_CONFIG = dict(
    num_attention_heads=32, attention_head_dim=128, in_channels=128, out_channels=128,
    num_layers=48, cross_attention_dim=4096,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=(20, 2048, 2048),
    timestep_scale_multiplier=1000, ff_bias=False,
    cross_attention_adaln=True, apply_gated_attention=True,
    use_keyframes_abs_pos_embedding=True, double_precision_rope=True,
)

# `video_embeddings_connector.*` config, read from the same DiT checkpoint's
# `config.transformer` -- see `vidax.models.ltx2_5.connector`.
CONNECTOR_CONFIG = dict(
    num_attention_heads=32, attention_head_dim=128, num_layers=8,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=(4096,),
    num_learnable_registers=128, apply_gated_attention=True, ff_bias=True,
    double_precision_rope=True,
)


def connector_kwargs_from_transformer_config(transformer_config: dict) -> dict:
    """Builds `Embeddings1DConnector` constructor kwargs from a DiT
    checkpoint's own `config.transformer` (see `load_ltx2_5_metadata`).
    """
    return dict(
        num_attention_heads=transformer_config.get("connector_num_attention_heads", 32),
        attention_head_dim=transformer_config.get("connector_attention_head_dim", 128),
        num_layers=transformer_config.get("connector_num_layers", 8),
        positional_embedding_theta=transformer_config.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=tuple(transformer_config.get("connector_positional_embedding_max_pos", [4096])),
        num_learnable_registers=transformer_config.get("connector_num_learnable_registers", 128),
        apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
        ff_bias=transformer_config.get("connector_ff_bias", True),
        double_precision_rope=transformer_config.get("frequencies_precision", False) == "float64",
    )


def dit_kwargs_from_transformer_config(transformer_config: dict) -> dict:
    """Builds `LTXDiT` constructor kwargs from a DiT checkpoint's own
    `config.transformer` (see `load_ltx2_5_metadata`) -- reads every field
    `vidax.models.ltx2_5.dit.LTXDiT` needs directly from the checkpoint
    rather than trusting `DIT_22B_CONFIG`.
    """
    return dict(
        num_attention_heads=transformer_config["num_attention_heads"],
        attention_head_dim=transformer_config["attention_head_dim"],
        in_channels=transformer_config["in_channels"], out_channels=transformer_config["out_channels"],
        num_layers=transformer_config["num_layers"], cross_attention_dim=transformer_config["cross_attention_dim"],
        positional_embedding_theta=transformer_config.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=tuple(transformer_config.get("positional_embedding_max_pos", [20, 2048, 2048])),
        timestep_scale_multiplier=transformer_config.get("timestep_scale_multiplier", 1000),
        ff_bias=transformer_config.get("ff_bias", True),
        cross_attention_adaln=transformer_config.get("cross_attention_adaln", False),
        apply_gated_attention=transformer_config.get("apply_gated_attention", False),
        use_keyframes_abs_pos_embedding=transformer_config.get("use_keyframes_abs_pos_embedding", False),
        double_precision_rope=transformer_config.get("frequencies_precision", False) == "float64",
        eps=transformer_config.get("norm_eps", 1e-6),
    )


def load_ltx2_5_metadata(checkpoint_path: str) -> dict:
    """Reads the `{"transformer": ..., "vae": ...}` (or a single-section)
    config blob embedded as safetensors file metadata in a released LTX-2.5
    component checkpoint -- the authoritative source of architecture
    hyperparameters (see module docstring).
    """
    with safetensors.safe_open(checkpoint_path, framework="numpy") as f:
        metadata = f.metadata()
    return json.loads(metadata["config"])


def load_gemma4_config(checkpoint_path: str) -> dict:
    """Reads the Gemma checkpoint's own embedded `gemma_config` metadata
    (a full HF `Gemma4UnifiedConfig.to_dict()`, distinct key name from
    `load_ltx2_5_metadata`'s `"config"`) -- see
    `vidax.models.ltx2_5.gemma4`'s docstring for the fields this port
    actually uses (`["text_config"]`).
    """
    with safetensors.safe_open(checkpoint_path, framework="numpy") as f:
        metadata = f.metadata()
    return json.loads(metadata["gemma_config"])


def gemma4_text_model_kwargs(gemma_config: dict) -> dict:
    """Builds `Gemma4TextModel` constructor kwargs from a checkpoint's real
    `gemma_config["text_config"]` (see `load_gemma4_config`) -- verified
    against the real `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`:
    `hidden_size=3840, num_hidden_layers=48, num_attention_heads=16,
    num_key_value_heads=8, head_dim=256, global_head_dim=512,
    num_global_key_value_heads=1, intermediate_size=15360,
    sliding_window=1024`, `layer_types` a 48-entry list (5 `sliding_
    attention` : 1 `full_attention`, last layer always `full_attention`),
    `rope_parameters={"sliding_attention": {"rope_theta": 10000.0,
    "rope_type": "default"}, "full_attention": {"rope_theta": 1000000.0,
    "rope_type": "proportional", "partial_rotary_factor": 0.25}}`.
    """
    tc = gemma_config["text_config"]
    rope = tc["rope_parameters"]
    return dict(
        vocab_size=tc["vocab_size"], hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"], global_head_dim=tc["global_head_dim"],
        num_global_key_value_heads=tc["num_global_key_value_heads"],
        layer_types=tuple(tc["layer_types"]), sliding_window=tc["sliding_window"],
        rope_theta_sliding=rope["sliding_attention"]["rope_theta"],
        rope_theta_full=rope["full_attention"]["rope_theta"],
        partial_rotary_factor_full=rope["full_attention"]["partial_rotary_factor"],
        eps=tc["rms_norm_eps"],
    )


def vae_scale_factors(vae_config: dict) -> "tuple[int, int]":
    """`(temporal_downscale_factor, spatial_downscale_factor)` -- same
    computation as `vidax.models.ltx_video.configs.vae_scale_factors`,
    generalized the same way (counts compression blocks rather than
    hardcoding a specific total).
    """
    encoder_blocks = vae_config["encoder_blocks"]
    spatial_compress_names = ("compress_space_res", "compress_all_res", "compress_space", "compress_all")
    temporal_compress_names = ("compress_time_res", "compress_all_res", "compress_time", "compress_all")
    num_spatial = sum(1 for name, _ in encoder_blocks if name in spatial_compress_names)
    num_temporal = sum(1 for name, _ in encoder_blocks if name in temporal_compress_names)
    spatial_scale = (2 ** num_spatial) * vae_config["patch_size"]
    temporal_scale = 2 ** num_temporal
    return temporal_scale, spatial_scale
