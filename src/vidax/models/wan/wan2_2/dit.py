"""Wan2.2 Diffusion Transformer backbone (Flax/JAX).

A structural port of the reference PyTorch ``WanModel`` from
Wan2.2-main/wan/modules/model.py. Architecturally this is Wan2.1's DiT minus
the CLIP-vision cross-attention branch (Wan2.2 has no
``WanI2VCrossAttention``/``MLPProj`` at all -- every block's attribute names
(``self_attn``, ``cross_attn``, ``norm1/2/3``, ``ffn.{0,2}``, ``modulation``,
``patch_embedding``, ``text_embedding.{0,2}``, ``time_embedding.{0,2}``,
``time_projection.1``, ``head.*``) are otherwise identical to Wan2.1's, so
`vidax.translator.mappings.common.map_wan_dit_keys` maps both).

The one real architectural difference is the timestep embedding: Wan2.2
always computes it **per patch token**, not once per sample. The reference
always expands a scalar-per-sample ``t`` to ``(B, seq_len)`` before doing
anything else (``if t.dim() == 1: t = t.expand(t.size(0), seq_len)``) --
this is what lets image-conditioned generation (via TI2V-5B's inpainting-
style sampling loop, which forces the known conditioning frame's tokens back
to timestep 0 at every step while other tokens keep the true step) share the
exact same model as pure text-to-video, where every token simply gets the
same value. This implementation always computes the per-token path (the
uniform-``t`` case is just its degenerate, but numerically identical,
special case), so `WanDiT`/`Wan22Head` here are NOT interchangeable with
`vidax.models.wan.wan2_1.dit`'s per-sample versions -- every intermediate
modulation tensor carries an extra token axis.

`y`/mask-based image conditioning (substituting the known conditioning
latent back into `x` between sampling steps, driven by a per-token timestep
of 0 there) is pipeline orchestration, not model architecture -- it isn't
implemented here, only plain text-to-video generation is currently wired up
in `examples/generate_wan2_2_ti2v.py`.

``sequence_parallel``: at TI2V-5B's only supported resolution (704x1280,
121 frames), the patch-token sequence is ~27k long, and Wan2.2's per-token
modulation tensors (`e`/`e0` above, plus every block's per-token
shift/scale/gate) scale with that directly -- Megatron-style tensor
parallelism (sharding attention heads/FFN channels, `mesh` alone) keeps the
*full* sequence on every device and doesn't shrink these, so it alone
doesn't fit a 4-chip v4 slice's HBM even after cutting weight memory 4x.
Setting `sequence_parallel=True` instead shards the *token sequence itself*
between blocks (each device holds only `seq_len / sp_size` tokens for the
FFN/norm/modulation-heavy part of every block, cutting exactly the memory
that was overflowing), reshuffling to a head-sharded, full-sequence view
only for the duration of self-attention itself via
`vidax.core.attention.sequence_parallel_self_attention`'s all-to-all
(DeepSpeed-Ulysses, matching Wan2.2-main/wan/distributed/sequence_parallel.py
+ ulysses.py). This requires the whole `WanDiT.apply(...)` call to run
inside `jax.experimental.shard_map.shard_map` over `mesh` (not just the
attention op, unlike the Megatron path) -- see
`examples/generate_wan2_2_ti2v.py` for how that's wired up, and DiT
parameters should be **replicated**, not tensor-sharded (`shard_wan_params`),
when this is enabled: cutting activation memory by sharding the sequence
made the OOM go away, not sharding weights, and the two schemes shard
fundamentally different axes (heads/channels vs. tokens) in ways that don't
compose without deliberately combining them, which this doesn't attempt.
"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.rope3d import sinusoidal_embedding_1d
from vidax.models.wan.common.dit_layers import attend, chunk_by_rank as _chunk_by_rank


class Wan22DiTBlock(nn.Module):
    """One transformer block: self-attn -> cross-attn -> FFN, AdaLN-modulated
    per-token (see this module's docstring for why, vs. Wan2.1's per-sample
    modulation).
    """
    dim: int
    ffn_dim: int
    num_heads: int
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mesh: Optional[Mesh] = None
    sequence_parallel: bool = False
    sp_axis_name: str = "tp"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        t_mod: jnp.ndarray,
        rope_freqs: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Args:
            x: (B, L, dim).
            context: (B, text_len, dim).
            t_mod: (B, L, 6, dim) -- per-token time projection (`e0` in the
                reference), one of 6 modulation vectors per token.
            rope_freqs: (cos, sin) RoPE angles.
        """
        modulation = self.param(
            "modulation", nn.initializers.normal(stddev=self.dim**-0.5),
            (1, 6, self.dim))

        # PyTorch computes modulation/gating in float32 regardless of the
        # ambient activation dtype; mirror that for numerical parity.
        e = (modulation.astype(jnp.float32)[:, None, :, :] + t_mod.astype(jnp.float32))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            e[:, :, i, :] for i in range(6)
        ]

        # --- self-attention ---
        norm_x = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm1")(x.astype(jnp.float32))
        norm_x = (norm_x * (1 + scale_msa) + shift_msa).astype(x.dtype)
        attn_out = attend(
            norm_x, norm_x, self.dim, self.num_heads, self.eps,
            prefix="self_attn", rope_freqs=rope_freqs, qk_norm=self.qk_norm,
            mesh=self.mesh, sequence_parallel=self.sequence_parallel,
            sp_axis_name=self.sp_axis_name)
        x = (x.astype(jnp.float32) +
             attn_out.astype(jnp.float32) * gate_msa).astype(x.dtype)

        # --- cross-attention ---
        norm_cross = nn.LayerNorm(
            use_scale=self.cross_attn_norm, use_bias=self.cross_attn_norm,
            epsilon=self.eps, name="norm3")(x)
        x = x + attend(
            norm_cross, context, self.dim, self.num_heads, self.eps,
            prefix="cross_attn", qk_norm=self.qk_norm, mesh=self.mesh,
            sequence_parallel=self.sequence_parallel, sp_axis_name=self.sp_axis_name)

        # --- feed-forward ---
        norm_h = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm2")(x.astype(jnp.float32))
        norm_h = (norm_h * (1 + scale_mlp) + shift_mlp).astype(x.dtype)
        h = nn.Dense(self.ffn_dim, name="ffn_0")(norm_h)
        h = nn.gelu(h, approximate=True)
        h = nn.Dense(self.dim, name="ffn_2")(h)
        x = (x.astype(jnp.float32) +
             h.astype(jnp.float32) * gate_mlp).astype(x.dtype)
        return x


class Wan22Head(nn.Module):
    """Final AdaLN-modulated projection back to patch space, per-token."""
    dim: int
    out_dim: int
    patch_size: Tuple[int, int, int]
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, L, dim).
            e: (B, L, dim) -- the plain (unprojected) per-token time embedding.
        """
        modulation = self.param(
            "modulation", nn.initializers.normal(stddev=self.dim**-0.5),
            (1, 2, self.dim))
        mod = modulation.astype(jnp.float32)[:, None, :, :] + e.astype(jnp.float32)[:, :, None, :]
        shift, scale = mod[:, :, 0, :], mod[:, :, 1, :]

        x = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm")(x.astype(jnp.float32))
        x = (x * (1 + scale) + shift).astype(e.dtype)

        out_dim = math.prod(self.patch_size) * self.out_dim
        return nn.Dense(out_dim, name="head")(x)


class WanDiT(nn.Module):
    """Wan2.2 DiT. Defaults match the released TI2V 5B config.

    Only text-to-video generation is exercised end-to-end so far (see this
    module's docstring); the architecture itself has no CLIP/i2v-specific
    parameters at all, unlike `vidax.models.wan.wan2_1.dit.WanDiT`.
    """
    dim: int = 3072
    ffn_dim: int = 14336
    num_heads: int = 24
    num_layers: int = 30
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 48
    out_dim: int = 48
    freq_dim: int = 256
    text_dim: int = 4096
    text_len: int = 512
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mesh: Optional[Mesh] = None
    sequence_parallel: bool = False
    sp_axis_name: str = "tp"

    @nn.compact
    def __call__(
        self,
        latents: jnp.ndarray,
        t: jnp.ndarray,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        context: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            latents: (B, T, H, W, C_in) video latents. When
                `sequence_parallel`, this must be the *full* (unsharded)
                latents on every device -- chunking happens internally,
                after `patch_embedding`, matching the reference (see this
                module's docstring). Must be called from *within*
                `shard_map(..., mesh=self.mesh)` in that case.
            t: (B,) or (B, L) diffusion timesteps, L = T/pt * H/ph * W/pw the
                number of patch tokens -- a scalar-per-sample `t` is
                broadcast to every token (see this module's docstring for
                why the reference always does this, even for plain t2v).
            freqs: (cos, sin) RoPE angles for this (T, H, W) grid, as
                returned by ``vidax.core.rope3d.create_rope3d_freqs``.
            context: (B, L_text, text_dim) text embeddings, L_text <= text_len.

        Returns:
            (B, T, H, W, C_out) denoised/velocity prediction.
        """
        input_dtype = latents.dtype
        b, t_p, h_p, w_p = latents.shape[0], *[
            latents.shape[1 + i] // self.patch_size[i] for i in range(3)
        ]

        # --- patchify ---
        x = nn.Conv(
            self.dim, self.patch_size, strides=self.patch_size,
            padding="VALID", name="patch_embedding")(latents)
        x = x.reshape(b, -1, self.dim)
        seq_len = x.shape[1]

        if t.ndim == 1:
            t = jnp.broadcast_to(t[:, None], (b, seq_len))

        # --- timestep embedding (float32, matching reference amp.autocast) ---
        t_freq = sinusoidal_embedding_1d(self.freq_dim, t.reshape(-1))
        e = nn.Dense(self.dim, name="time_embedding_0")(t_freq)
        e = nn.silu(e)
        e = nn.Dense(self.dim, name="time_embedding_2")(e)
        e = e.reshape(b, seq_len, self.dim)
        e0 = nn.Dense(self.dim * 6, name="time_projection_1")(nn.silu(e))
        e0 = e0.reshape(b, seq_len, 6, self.dim)

        # --- text context embedding (zero-padded to text_len, like the ref) ---
        text_pad = self.text_len - context.shape[1]
        assert text_pad >= 0, "context length exceeds text_len"
        if text_pad > 0:
            context = jnp.pad(context, ((0, 0), (0, text_pad), (0, 0)))
        context = nn.Dense(self.dim, name="text_embedding_0")(context)
        context = nn.gelu(context, approximate=True)
        context = nn.Dense(self.dim, name="text_embedding_2")(context)

        # --- sequence-parallel chunk: split the token sequence itself across
        # `sp_axis_name`, so every block's per-token activations (x, e, e0,
        # and every intermediate inside Wan22DiTBlock) only ever cover this
        # device's local share -- see this module's docstring. Self-
        # attention alone reshuffles back to a full-sequence view internally
        # (`attend`'s `sequence_parallel` path), so `context`/`freqs` need no
        # special handling: `context` is already small and fully replicated,
        # and `freqs` is chunked the same way as `x`/`e`/`e0` since RoPE
        # angles are position-indexed identically to the token sequence.
        cos, sin = freqs
        if self.sequence_parallel:
            sp_size = self.mesh.shape[self.sp_axis_name]
            assert seq_len % sp_size == 0, (
                f"sequence_parallel requires the patch token count ({seq_len}) "
                f"to be evenly divisible by the sequence-parallel size ({sp_size})")
            rank = jax.lax.axis_index(self.sp_axis_name)
            x = _chunk_by_rank(x, 1, sp_size, rank)
            e = _chunk_by_rank(e, 1, sp_size, rank)
            e0 = _chunk_by_rank(e0, 1, sp_size, rank)
            cos = _chunk_by_rank(cos, 1, sp_size, rank)
            sin = _chunk_by_rank(sin, 1, sp_size, rank)
        freqs = (cos, sin)

        # --- transformer blocks ---
        for i in range(self.num_layers):
            x = Wan22DiTBlock(
                dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                qk_norm=self.qk_norm, cross_attn_norm=self.cross_attn_norm,
                eps=self.eps, mesh=self.mesh, sequence_parallel=self.sequence_parallel,
                sp_axis_name=self.sp_axis_name, name=f"blocks_{i}")(
                    x, context, e0, freqs)

        # --- head + unpatchify ---
        x = Wan22Head(
            self.dim, self.out_dim, self.patch_size, self.eps,
            name="head")(x, e)
        x = x.astype(input_dtype)

        if self.sequence_parallel:
            # Re-assemble the full token sequence (every device's local
            # output chunk, in rank order) before unpatchify -- matches the
            # reference's `gather_forward` in `sp_dit_forward`.
            x = jax.lax.all_gather(x, self.sp_axis_name, axis=1, tiled=True)

        pt, ph, pw = self.patch_size
        x = x.reshape(b, t_p, h_p, w_p, pt, ph, pw, self.out_dim)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.reshape(b, t_p * pt, h_p * ph, w_p * pw, self.out_dim)
        return x
