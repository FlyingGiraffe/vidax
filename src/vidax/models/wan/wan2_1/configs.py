"""Named `WanDiT` hyperparameter presets for Wan2.1's released checkpoints
(`Wan2.1-main/wan/configs/*.py` in the reference). `WanDiT` itself is
architecture-only and config-driven; these dicts are just its constructor
kwargs for each released size/task, cross-checked against each checkpoint's
own `config.json`.
"""

T2V_1_3B_CONFIG = dict(
    dim=1536, ffn_dim=8960, num_heads=12, num_layers=30,
    freq_dim=256, text_dim=4096, text_len=512, in_dim=16, out_dim=16,
    patch_size=(1, 2, 2), qk_norm=True, cross_attn_norm=True, eps=1e-6,
    model_type="t2v",
)

T2V_14B_CONFIG = dict(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=40,
    freq_dim=256, text_dim=4096, text_len=512, in_dim=16, out_dim=16,
    patch_size=(1, 2, 2), qk_norm=True, cross_attn_norm=True, eps=1e-6,
    model_type="t2v",
)

I2V_14B_CONFIG = dict(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=40,
    freq_dim=256, text_dim=4096, text_len=512, in_dim=16, out_dim=16,
    patch_size=(1, 2, 2), qk_norm=True, cross_attn_norm=True, eps=1e-6,
    model_type="i2v", image_dim=1280,
)
