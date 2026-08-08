"""Named `WanDiT` hyperparameter presets for Wan2.2's released checkpoints
(`Wan2.2-main/wan/configs/*.py` in the reference). `WanDiT` itself is
architecture-only and config-driven; these dicts are just its constructor
kwargs for each released size/task, cross-checked against each checkpoint's
own `config.json`.

Unlike Wan2.1, Wan2.2 has no `model_type`/`image_dim` fields at all (no CLIP
branch) -- A14B's I2V variant conditions purely through `in_dim` (36 instead
of 16: the noisy latent's 16 channels plus a 20-channel mask+VAE-latent `y`,
concatenated before `patch_embedding` by the caller), see
`examples/generate_wan2_2_i2v_a14b.py`.
"""

TI2V_5B_CONFIG = dict(
    dim=3072, ffn_dim=14336, num_heads=24, num_layers=30,
    patch_size=(1, 2, 2), in_dim=48, out_dim=48, freq_dim=256,
    text_dim=4096, text_len=512, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)

T2V_A14B_CONFIG = dict(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=40,
    patch_size=(1, 2, 2), in_dim=16, out_dim=16, freq_dim=256,
    text_dim=4096, text_len=512, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)

I2V_A14B_CONFIG = dict(
    dim=5120, ffn_dim=13824, num_heads=40, num_layers=40,
    patch_size=(1, 2, 2), in_dim=36, out_dim=16, freq_dim=256,
    text_dim=4096, text_len=512, qk_norm=True, cross_attn_norm=True, eps=1e-6,
)
