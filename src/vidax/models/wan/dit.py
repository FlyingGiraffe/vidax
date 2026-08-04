"""Wan2.1 Diffusion Transformer backbone (Flax/JAX), t2v and i2v.

A structural port of the reference PyTorch ``WanModel`` from
Wan2.1-main/wan/modules/model.py, with weights loaded via
``vidax.translator``. Only the t2v and i2v paths are implemented (no
first-last-frame/vace conditioning).

Differences from the PyTorch reference are limited to what static-shape JAX
requires: this implementation assumes every video in a batch shares the same
(T, H, W) grid, so it has no need for the reference's per-sample
grid_sizes/seq_lens padding machinery — RoPE frequencies are precomputed
once per call (see ``vidax.core.rope3d.create_rope3d_freqs``) for the shared
grid instead of per-sample.

i2v conditioning (``model_type="i2v"``): the reference concatenates two
extra signals onto the noisy latent before patchifying -- a per-frame mask
(1 for the conditioning frame, 0 elsewhere) and the VAE-encoded
conditioning frame itself (`WanVAEEncoder`, zero-padded to the full video
length) -- and additionally cross-attends each block onto CLIP image
features (`ClipVisionTransformer`) through a second, image-only K/V
projection (`WanI2VCrossAttention` in the reference), summed with the usual
text cross-attention before the output projection. Building `y` (the
mask+latent concatenation) is pipeline orchestration, not model
architecture, so it lives in `examples/generate_wan2_1_i2v.py`, not here.
"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import RMSNorm, dot_product_attention
from vidax.core.rope3d import apply_rope3d, sinusoidal_embedding_1d


def _attend(
    x_q: jnp.ndarray,
    x_kv: jnp.ndarray,
    dim: int,
    num_heads: int,
    eps: float,
    prefix: str,
    rope_freqs: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    qk_norm: bool = True,
    mesh: Optional[Mesh] = None,
    image_context: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Shared QKV-projection + RMSNorm + attention + output-projection path.

    Used for both self-attention (x_kv is x, rope_freqs set) and
    cross-attention (x_kv is the text context, rope_freqs None), matching
    ``WanSelfAttention``/``WanT2VCrossAttention`` in the reference, which
    share this exact structure.

    ``image_context``, when given (i2v cross-attention only), adds a second
    K/V projection over CLIP image features and sums its attention output
    with the text one before the shared output projection -- matching
    ``WanI2VCrossAttention``, which reuses the same query for both.
    """
    head_dim = dim // num_heads
    b = x_q.shape[0]

    q = nn.Dense(dim, name=f"{prefix}_q")(x_q)
    k = nn.Dense(dim, name=f"{prefix}_k")(x_kv)
    v = nn.Dense(dim, name=f"{prefix}_v")(x_kv)

    if qk_norm:
        q = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_q")(q)
        k = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_k")(k)

    q = q.reshape(b, -1, num_heads, head_dim)
    k = k.reshape(b, -1, num_heads, head_dim)
    v = v.reshape(b, -1, num_heads, head_dim)

    if rope_freqs is not None:
        q = apply_rope3d(q, rope_freqs)
        k = apply_rope3d(k, rope_freqs)

    out = dot_product_attention(q, k, v, mesh=mesh)
    out = out.reshape(b, -1, dim)

    if image_context is not None:
        k_img = nn.Dense(dim, name=f"{prefix}_k_img")(image_context)
        v_img = nn.Dense(dim, name=f"{prefix}_v_img")(image_context)
        if qk_norm:
            k_img = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_k_img")(k_img)
        k_img = k_img.reshape(b, -1, num_heads, head_dim)
        v_img = v_img.reshape(b, -1, num_heads, head_dim)
        img_out = dot_product_attention(q, k_img, v_img, mesh=mesh).reshape(b, -1, dim)
        out = out + img_out

    return nn.Dense(dim, name=f"{prefix}_o")(out)


class WanDiTBlock(nn.Module):
    """One transformer block: self-attn -> cross-attn -> FFN, AdaLN-modulated."""
    dim: int
    ffn_dim: int
    num_heads: int
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mesh: Optional[Mesh] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        t_mod: jnp.ndarray,
        rope_freqs: Tuple[jnp.ndarray, jnp.ndarray],
        image_context: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        modulation = self.param(
            "modulation", nn.initializers.normal(stddev=self.dim**-0.5),
            (1, 6, self.dim))

        # PyTorch computes modulation/gating in float32 regardless of the
        # ambient activation dtype; mirror that for numerical parity.
        e = (modulation.astype(jnp.float32) + t_mod.astype(jnp.float32))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            e[:, i:i + 1, :] for i in range(6)
        ]

        # --- self-attention ---
        norm_x = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm1")(x.astype(jnp.float32))
        norm_x = (norm_x * (1 + scale_msa) + shift_msa).astype(x.dtype)
        attn_out = _attend(
            norm_x, norm_x, self.dim, self.num_heads, self.eps,
            prefix="self_attn", rope_freqs=rope_freqs, qk_norm=self.qk_norm,
            mesh=self.mesh)
        x = (x.astype(jnp.float32) +
             attn_out.astype(jnp.float32) * gate_msa).astype(x.dtype)

        # --- cross-attention ---
        norm_cross = nn.LayerNorm(
            use_scale=self.cross_attn_norm, use_bias=self.cross_attn_norm,
            epsilon=self.eps, name="norm3")(x)
        x = x + _attend(
            norm_cross, context, self.dim, self.num_heads, self.eps,
            prefix="cross_attn", qk_norm=self.qk_norm, mesh=self.mesh,
            image_context=image_context)

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


class MLPProj(nn.Module):
    """Projects CLIP image features into the DiT's cross-attention dimension."""
    in_dim: int = 1280
    out_dim: int = 1536
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=self.eps, name="proj_0")(x.astype(jnp.float32)).astype(x.dtype)
        x = nn.Dense(self.in_dim, name="proj_1")(x)
        x = nn.gelu(x, approximate=False)  # reference uses exact (erf) GELU here.
        x = nn.Dense(self.out_dim, name="proj_3")(x)
        x = nn.LayerNorm(epsilon=self.eps, name="proj_4")(x.astype(jnp.float32)).astype(x.dtype)
        return x


class WanHead(nn.Module):
    """Final AdaLN-modulated projection back to patch space."""
    dim: int
    out_dim: int
    patch_size: Tuple[int, int, int]
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
        modulation = self.param(
            "modulation", nn.initializers.normal(stddev=self.dim**-0.5),
            (1, 2, self.dim))
        mod = modulation.astype(jnp.float32) + e.astype(jnp.float32)[:, None, :]
        shift, scale = mod[:, 0:1, :], mod[:, 1:2, :]

        x = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm")(x.astype(jnp.float32))
        x = (x * (1 + scale) + shift).astype(e.dtype)

        out_dim = math.prod(self.patch_size) * self.out_dim
        return nn.Dense(out_dim, name="head")(x)


class WanDiT(nn.Module):
    """Wan2.1 DiT. Defaults match the released t2v 1.3B config.

    For i2v (``model_type="i2v"``), pass the 14B config's `dim`/`ffn_dim`/
    `num_heads`/`num_layers` explicitly -- there is no 1.3B i2v checkpoint.
    """
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_dim: int = 16
    out_dim: int = 16
    freq_dim: int = 256
    text_dim: int = 4096
    text_len: int = 512
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mesh: Optional[Mesh] = None
    model_type: str = "t2v"  # "t2v" or "i2v"
    image_dim: int = 1280  # CLIP ViT-H/14's vision_dim; only used if model_type == "i2v".

    @nn.compact
    def __call__(
        self,
        latents: jnp.ndarray,
        t: jnp.ndarray,
        freqs: Tuple[jnp.ndarray, jnp.ndarray],
        context: jnp.ndarray,
        y: Optional[jnp.ndarray] = None,
        clip_fea: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            latents: (B, T, H, W, C_in) video latents.
            t: (B,) diffusion timesteps.
            freqs: (cos, sin) RoPE angles for this (T, H, W) grid, as
                returned by ``vidax.core.rope3d.create_rope3d_freqs``.
            context: (B, L, text_dim) text embeddings, L <= text_len.
            y: i2v only -- (B, T, H, W, C_y) conditioning signal (mask +
                VAE-encoded conditioning frame), concatenated onto `latents`
                before patchifying. See this module's docstring.
            clip_fea: i2v only -- (B, 257, image_dim) CLIP image features
                from `ClipVisionTransformer`.

        Returns:
            (B, T, H, W, C_out) denoised/velocity prediction.
        """
        if self.model_type == "i2v":
            assert y is not None and clip_fea is not None, (
                "model_type='i2v' requires both `y` and `clip_fea`")
            latents = jnp.concatenate([latents, y], axis=-1)

        input_dtype = latents.dtype
        b, t_p, h_p, w_p = latents.shape[0], *[
            latents.shape[1 + i] // self.patch_size[i] for i in range(3)
        ]

        # --- patchify ---
        x = nn.Conv(
            self.dim, self.patch_size, strides=self.patch_size,
            padding="VALID", name="patch_embedding")(latents)
        x = x.reshape(b, -1, self.dim)

        context_img = None
        if self.model_type == "i2v":
            context_img = MLPProj(self.image_dim, self.dim, name="img_emb")(clip_fea)

        # --- timestep embedding (float32, matching reference amp.autocast) ---
        t_freq = sinusoidal_embedding_1d(self.freq_dim, t)
        e = nn.Dense(self.dim, name="time_embedding_0")(t_freq)
        e = nn.silu(e)
        e = nn.Dense(self.dim, name="time_embedding_2")(e)
        e0 = nn.Dense(self.dim * 6, name="time_projection_1")(nn.silu(e))
        e0 = e0.reshape(b, 6, self.dim)

        # --- text context embedding (zero-padded to text_len, like the ref) ---
        text_pad = self.text_len - context.shape[1]
        assert text_pad >= 0, "context length exceeds text_len"
        if text_pad > 0:
            context = jnp.pad(context, ((0, 0), (0, text_pad), (0, 0)))
        context = nn.Dense(self.dim, name="text_embedding_0")(context)
        context = nn.gelu(context, approximate=True)
        context = nn.Dense(self.dim, name="text_embedding_2")(context)

        # --- transformer blocks ---
        for i in range(self.num_layers):
            x = WanDiTBlock(
                dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                qk_norm=self.qk_norm, cross_attn_norm=self.cross_attn_norm,
                eps=self.eps, mesh=self.mesh, name=f"blocks_{i}")(
                    x, context, e0, freqs, image_context=context_img)

        # --- head + unpatchify ---
        x = WanHead(
            self.dim, self.out_dim, self.patch_size, self.eps,
            name="head")(x, e)
        x = x.astype(input_dtype)

        pt, ph, pw = self.patch_size
        x = x.reshape(b, t_p, h_p, w_p, pt, ph, pw, self.out_dim)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.reshape(b, t_p * pt, h_p * ph, w_p * pw, self.out_dim)
        return x
