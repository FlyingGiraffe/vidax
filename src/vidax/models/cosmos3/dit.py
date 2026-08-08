"""Cosmos3's diffusion transformer (`Cosmos3OmniTransformer` in the
reference, refs/diffusers-cosmos3/transformer_cosmos3.py), T2V/I2V-only:
the `embed_tokens` + N dual-pathway decoder layers + vision patchify/
unpatchify path, with `lm_head` and all `action_*`/`audio_*` submodules
dropped (this port never needs a next-token/action/sound head). Shared by
both Cosmos3-Nano and Cosmos3-Edge -- see `vidax.models.cosmos3.configs` for
their respective hyperparameters.

No AdaLN modulation anywhere in this architecture (unlike Wan/Cosmos-Predict
2.5's DiTs) -- the diffusion timestep is injected exactly once, additively,
directly into the noisy vision tokens themselves (`_inject_timestep_embed`,
matching the reference's `_apply_timestep_embeds_to_noisy_tokens`), and then
flows through ordinary pre-norm transformer blocks with no further
timestep-conditioned scale/shift.

Ported against a fixed-shape `(B, seq_len, hidden)` packed sequence (see
`vidax.models.cosmos3.dit_layers`'s module docstring) rather than the
reference's ragged flat-buffer-with-global-indices design: callers build and
pass `und_position_ids`/`gen_position_ids` (mRoPE ids, from
`vidax.models.cosmos3.mrope`) directly, keeping this module itself agnostic
to how the caller derived them (prompt length, image conditioning,
resolution, etc. -- all pipeline-level concerns, not part of the DiT).
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import RMSNorm
from vidax.core.rope3d import sinusoidal_embedding_1d
from vidax.models.cosmos3.dit_layers import Cosmos3VLTextMoTDecoderLayer
from vidax.models.cosmos3.mrope import compute_cosmos3_mrope_cos_sin


class Cosmos3Transformer(nn.Module):
    """Cosmos3 DiT (Nano or Edge), T2V/I2V. Build via
    `Cosmos3Transformer(**vidax.models.cosmos3.configs.NANO_CONFIG)` or
    `..EDGE_CONFIG`.

    Args (see module docstring for the reference class this ports):
        vocab_size, hidden_size, intermediate_size, num_hidden_layers,
            num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps:
            standard transformer config, matching `transformer/config.json`.
        hidden_act, qk_norm_for_text, use_und_k_norm_for_gen: per-checkpoint
            toggles, see `vidax.models.cosmos3.dit_layers`.
        latent_channel: VAE latent channel count (48, Wan2.2-TI2V's VAE).
        latent_patch_size: spatial patch size for vision tokens (2x2).
        rope_theta, rope_axes_dim: interleaved 3D mRoPE config
            (`vidax.models.cosmos3.mrope`).
        timestep_scale: DiT-internal sigma rescale before the sinusoidal
            embedding -- must be applied, or the network is conditioned on
            an out-of-distribution noise level at every sampling step (same
            mechanism as Cosmos-Predict2.5's `MinimalV1LVGDiT.timestep_scale`).
        mesh: TPU mesh for the attention dispatch (see `dit_layers.py`).
    """
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    qk_norm_for_text: bool = True
    use_und_k_norm_for_gen: bool = False
    latent_channel: int = 48
    latent_patch_size: int = 2
    rope_theta: float = 5_000_000.0
    rope_axes_dim: Tuple[int, int, int] = (24, 20, 20)
    timestep_scale: float = 0.001
    mesh: Optional[Mesh] = None

    def _patchify(self, latents: jnp.ndarray) -> jnp.ndarray:
        """(B, T, H, W, C) -> (B, T*Hp*Wp, p*p*C), matching the reference's
        `einsum("cthpwq->thwpqc")` channel-last equivalent: within a patch,
        the flattened order is (p_h, p_w, channel) with channel fastest."""
        b, t, h, w, c = latents.shape
        p = self.latent_patch_size
        hp, wp = h // p, w // p
        x = latents.reshape(b, t, hp, p, wp, p, c)
        x = x.transpose(0, 1, 2, 4, 3, 5, 6)  # (b, t, hp, wp, p, p, c)
        return x.reshape(b, t * hp * wp, p * p * c)

    def _unpatchify(self, tokens: jnp.ndarray, t: int, h: int, w: int) -> jnp.ndarray:
        """Inverse of `_patchify`: (B, T*Hp*Wp, p*p*C) -> (B, T, H, W, C)."""
        b = tokens.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel
        hp, wp = h // p, w // p
        x = tokens.reshape(b, t, hp, wp, p, p, c)
        x = x.transpose(0, 1, 2, 4, 3, 5, 6)  # (b, t, hp, p, wp, p, c) -- self-inverse permutation
        return x.reshape(b, t, hp * p, wp * p, c)

    def _inject_timestep_embed(
        self, tokens: jnp.ndarray, per_frame_sigma_scaled: jnp.ndarray,
        noisy_mask: jnp.ndarray, tokens_per_frame: int,
    ) -> jnp.ndarray:
        """Adds the (already `timestep_scale`-scaled) per-frame timestep
        embedding into every patch token of that frame, masked to noisy
        frames only -- the dense, `(B, ...)`-batched equivalent of the
        reference's `scatter_add` over `noisy_frame_indexes`.

        Args:
            tokens: (B, T*Hp*Wp, hidden).
            per_frame_sigma_scaled: (B, T) -- already `* timestep_scale`.
            noisy_mask: (B, T) -- 1.0 at noisy frames, 0.0 at clean/conditioned
                frames (which get no timestep embedding at all, matching the
                reference's `noisy_frame_indexes`-only scatter).
            tokens_per_frame: `Hp * Wp`.
        """
        t_sin = sinusoidal_embedding_1d(256, per_frame_sigma_scaled.reshape(-1))  # (B*T, 256)
        t_sin = t_sin.reshape(per_frame_sigma_scaled.shape + (256,)).astype(tokens.dtype)
        time_embed = nn.Dense(self.hidden_size, name="time_embedder_linear_1")(t_sin)
        time_embed = nn.silu(time_embed)
        time_embed = nn.Dense(self.hidden_size, name="time_embedder_linear_2")(time_embed)  # (B, T, hidden)
        time_embed = time_embed * noisy_mask[..., None]
        time_embed = jnp.repeat(time_embed, tokens_per_frame, axis=1)  # (B, T*Hp*Wp, hidden)
        return tokens + time_embed

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,           # (B, und_len) int32
        und_position_ids: jnp.ndarray,    # (3, B, und_len) float32
        und_valid_mask: jnp.ndarray,      # (B, und_len) bool, True = real token
        vision_latents: jnp.ndarray,      # (B, T, H, W, latent_channel)
        gen_position_ids: jnp.ndarray,    # (3, B, T*Hp*Wp) float32
        vision_sigma: jnp.ndarray,        # (B, T) raw sigma, one value per latent frame
        vision_noisy_mask: jnp.ndarray,   # (B, T) 1.0 = noisy frame, 0.0 = clean/conditioned
    ) -> jnp.ndarray:
        b, t, h, w, _ = vision_latents.shape
        p = self.latent_patch_size
        hp, wp = h // p, w // p

        und_seq = nn.Embed(self.vocab_size, self.hidden_size, name="embed_tokens")(input_ids)

        gen_tokens = self._patchify(vision_latents)
        gen_tokens = nn.Dense(self.hidden_size, name="proj_in")(gen_tokens)
        gen_tokens = self._inject_timestep_embed(
            gen_tokens, vision_sigma * self.timestep_scale, vision_noisy_mask, hp * wp)

        und_len = und_seq.shape[1]
        causal_mask = jnp.tril(jnp.ones((und_len, und_len), dtype=jnp.bool_))[None, None]

        cos_und, sin_und = compute_cosmos3_mrope_cos_sin(
            und_position_ids, self.head_dim, self.rope_theta, self.rope_axes_dim)
        cos_gen, sin_gen = compute_cosmos3_mrope_cos_sin(
            gen_position_ids, self.head_dim, self.rope_theta, self.rope_axes_dim)
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)

        gen_seq = gen_tokens
        for i in range(self.num_hidden_layers):
            und_seq, gen_seq = Cosmos3VLTextMoTDecoderLayer(
                hidden_size=self.hidden_size, head_dim=self.head_dim,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                intermediate_size=self.intermediate_size, eps=self.rms_norm_eps,
                hidden_act=self.hidden_act, qk_norm_for_text=self.qk_norm_for_text,
                use_und_k_norm_for_gen=self.use_und_k_norm_for_gen,
                mesh=self.mesh, name=f"layers_{i}",
            )(und_seq, gen_seq, rotary_emb, causal_mask, und_valid_mask)

        gen_out = RMSNorm(self.hidden_size, eps=self.rms_norm_eps, name="norm_moe_gen")(gen_seq)
        preds = nn.Dense(p * p * self.latent_channel, name="proj_out")(gen_out)
        return self._unpatchify(preds, t, h, w)
