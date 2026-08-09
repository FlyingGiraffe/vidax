"""Named `CosmosDiT` hyperparameter presets for Cosmos-Predict2.5's released
checkpoints. `CosmosDiT` itself is architecture-only and config-driven; these
dicts are just its constructor kwargs for each released size, cross-checked
against each checkpoint's own tensor shapes (`net.x_embedder.proj.1.weight`,
`net.blocks.0.mlp.layer1.weight`, block count, etc. -- there is no
`config.json` shipped alongside these `.pt` checkpoints, unlike Wan's).

The two released sizes differ *only* in `dim`/`ffn_dim`/`num_heads`/
`num_layers` -- `head_dim` (128), `context_dim`/`context_raw_dim` (the
Reason1 cross-attention width/input width), `adaln_lora_dim`, `patch_size`,
RoPE extrapolation ratios, and `timestep_scale` are all identical between
2B and 14B (confirmed both from checkpoint tensor shapes and from the
reference's `model_14b_reason_1p1_rectified_flow.py` experiment config,
which sets `shift=5`/`rope_h_extrapolation_ratio=3.0`/
`rope_w_extrapolation_ratio=3.0`/`rope_t_extrapolation_ratio=1.0`/
`timestep_scale=0.001` -- exactly `CosmosDiT`'s own 2B defaults).
"""

BASE_2B_CONFIG = dict(
    dim=2048, ffn_dim=8192, num_heads=16, head_dim=128, num_layers=28,
)

BASE_14B_CONFIG = dict(
    dim=5120, ffn_dim=20480, num_heads=40, head_dim=128, num_layers=36,
)
