"""Named `Cosmos3Transformer` hyperparameter presets for the released
Cosmos3 checkpoints, cross-checked against each checkpoint's own
`transformer/config.json`.
"""

NANO_CONFIG = dict(
    vocab_size=151936, hidden_size=4096, intermediate_size=12288,
    num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, rms_norm_eps=1e-6, hidden_act="silu",
    qk_norm_for_text=True, use_und_k_norm_for_gen=False,
    rope_theta=5_000_000.0, rope_axes_dim=(24, 20, 20),
    timestep_scale=0.001,
)

EDGE_CONFIG = dict(
    vocab_size=131072, hidden_size=2048, intermediate_size=9216,
    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, rms_norm_eps=1e-5, hidden_act="relu2",
    qk_norm_for_text=False, use_und_k_norm_for_gen=True,
    rope_theta=100_000_000.0, rope_axes_dim=(24, 20, 20),
    timestep_scale=0.001,
)
