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
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.rope3d import sinusoidal_embedding_1d
from vidax.models.wan.common.dit_layers import WanHead, attend as _attend, chunk_by_rank


class WanDiTBlock(nn.Module):
    """One transformer block: self-attn -> cross-attn -> FFN, AdaLN-modulated.

    ``x`` is expected to already be float32 on entry, and stays float32 on
    return -- it is the persistent residual-stream accumulator, matching the
    reference's `amp.autocast(dtype=torch.float32)`-wrapped gated residual
    updates (`x = x + y * e[2]` for self-attn, and the FFN's identical
    pattern). Those two ops are the *only* place the reference explicitly
    forces float32, but since neither ever casts the result back down, and
    PyTorch's ordinary type promotion keeps a float32+bf16 add in float32
    too (the un-wrapped cross-attention residual `x = x + self.cross_attn
    (...)`), the reference's residual stream is float32 for the entire
    network from partway through block 0 onward -- only individual sub-layer
    matmuls (Q/K/V/O, FFN) run in bf16, via the ambient `amp.autocast
    (dtype=torch.bfloat16)` set once outside the whole model. `WanDiT
    .__call__` reproduces this by upcasting once before the block loop
    starts (matching "partway through block 0" closely enough -- the only
    difference is a few elementwise ops on the very first block's *input*
    running in bf16 vs fp32, negligible) rather than piecemeal per block.

    Getting this right matters beyond numerical parity: re-quantizing the
    residual stream to bf16 after every one of the 80 gated residual adds
    (2 per layer x 40 layers) -- what this module used to do -- compounds
    into visibly corrupted (flat, hazy, low-detail) output at large token
    counts (e.g. 720p x 81 frames), while looking fine at smaller scales;
    see docs/models/wan2_1.md#status for the full investigation.

    ``compute_dtype`` is the dtype sub-layer matmuls run in (the model's
    overall dtype, e.g. bfloat16) -- every normalized/modulated activation
    is explicitly downcast to it right before entering `attend`/FFN Dense
    calls, mirroring autocast's per-op downcast of an fp32 input for a
    lower-precision-registered op, and each sub-layer's output is cast back
    up to float32 before being added into the residual stream.
    """
    dim: int
    ffn_dim: int
    num_heads: int
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    mesh: Optional[Mesh] = None
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"
    compute_dtype: jnp.dtype = jnp.bfloat16

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
        # ambient activation dtype; mirror that for numerical parity. Unlike
        # Wan2.2, `t_mod` (`e0`) is per-*sample*, not per-token (shape
        # (B, 6, dim)), so it broadcasts unchanged over `x` whether `x` is
        # the full token sequence or (under sequence_parallel) just this
        # device's local chunk of it -- no chunking of `t_mod`/`e` is needed
        # anywhere in this file, unlike `vidax.models.wan.wan2_2.dit`.
        e = (modulation.astype(jnp.float32) + t_mod.astype(jnp.float32))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            e[:, i:i + 1, :] for i in range(6)
        ]

        # --- self-attention ---
        norm_x = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm1")(x)
        norm_x = (norm_x * (1 + scale_msa) + shift_msa).astype(self.compute_dtype)
        attn_out = _attend(
            norm_x, norm_x, self.dim, self.num_heads, self.eps,
            prefix="self_attn", rope_freqs=rope_freqs, qk_norm=self.qk_norm,
            mesh=self.mesh, sequence_parallel=self.sequence_parallel,
            sp_axis_name=self.sp_axis_name)
        x = x + attn_out.astype(jnp.float32) * gate_msa

        # --- cross-attention ---
        norm_cross = nn.LayerNorm(
            use_scale=self.cross_attn_norm, use_bias=self.cross_attn_norm,
            epsilon=self.eps, name="norm3")(x).astype(self.compute_dtype)
        x = x + _attend(
            norm_cross, context, self.dim, self.num_heads, self.eps,
            prefix="cross_attn", qk_norm=self.qk_norm, mesh=self.mesh,
            image_context=image_context, sequence_parallel=self.sequence_parallel,
            sp_axis_name=self.sp_axis_name).astype(jnp.float32)

        # --- feed-forward ---
        norm_h = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm2")(x)
        norm_h = (norm_h * (1 + scale_mlp) + shift_mlp).astype(self.compute_dtype)
        # `ffn_0` is column-parallel: under `sequence_parallel` (running
        # inside `shard_map`), its declared output width must already be
        # this device's local share -- see `vidax.models.wan.common
        # .dit_layers.attend`'s identical comment for the full reasoning
        # (GSPMD handles this automatically outside `shard_map`, but nothing
        # does inside it). A no-op when 'tp' has size 1.
        tp_size = self.mesh.shape["tp"] if (self.sequence_parallel and self.mesh is not None) else 1
        h = nn.Dense(self.ffn_dim // tp_size, name="ffn_0")(norm_h)
        h = nn.gelu(h, approximate=True)
        h = nn.Dense(self.dim, name="ffn_2")(h)
        # `ffn_2` is row-parallel -- see `vidax.models.wan.common.dit_layers
        # .attend`'s identical comment for why this manual reduce is only
        # needed under `sequence_parallel`, and why it's a safe no-op
        # otherwise.
        if self.sequence_parallel:
            h = jax.lax.psum(h, "tp")
        x = x + h.astype(jnp.float32) * gate_mlp
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


class WanDiT(nn.Module):
    """Wan2.1 DiT. Defaults match the released t2v 1.3B config.

    For i2v (``model_type="i2v"``), pass the 14B config's `dim`/`ffn_dim`/
    `num_heads`/`num_layers` explicitly -- there is no 1.3B i2v checkpoint.

    ``sequence_parallel``: shards the token sequence itself across
    `sp_axis_name` (the mesh's `'sp'` axis) between blocks (DeepSpeed-
    Ulysses, reshuffling to a head-sharded full-sequence view only for the
    duration of self-attention -- see `vidax.models.wan.wan2_2.dit`'s module
    docstring for the full mechanism, which this reuses unchanged via
    `attend`, including composing with ordinary Megatron weight-sharding on
    the independent `'tp'` axis). The one thing genuinely simpler here than
    in Wan2.2: `e`/`e0` (the timestep modulation) are per-*sample*, not
    per-token, so unlike Wan2.2 they never need chunking -- only `x` and
    `freqs` do. Requires the whole `WanDiT.apply(...)` call to run inside
    `shard_map(..., mesh=mesh)`, same as Wan2.2; see
    `examples/generate_wan2_1_t2v.py`/`generate_wan2_1_i2v.py` for how
    that's wired up.
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
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"
    # dtype each `WanDiTBlock`'s sub-layer matmuls (Q/K/V/O, FFN) transiently
    # downcast to before entering `attend`/Dense calls -- should match the
    # DiT's *weight* dtype (`--dit_dtype` in the example scripts), not
    # necessarily `latents`'s own dtype. Left unset (`None`), this defaults
    # to `latents.dtype` for backward compatibility, which is only correct
    # when weights and activations share one dtype; the moment they diverge
    # (e.g. `--dit_dtype float32` weights with `--dtype bfloat16`
    # latents/T5/VAE), an explicit compute_dtype here is required --
    # otherwise every block still narrows activations down to
    # `latents.dtype` before each matmul regardless of the wider weight
    # precision, silently reintroducing the same repeated-bf16-rounding
    # corruption at scale that using float32 weights was meant to fix in
    # the first place (see docs/models/wan2_1.md#status).
    compute_dtype: Optional[jnp.dtype] = None

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
            latents: (B, T, H, W, C_in) video latents. When
                `sequence_parallel`, this must be the *full* (unsharded)
                latents on every device -- chunking happens internally,
                after `patch_embedding`. Must be called from *within*
                `shard_map(..., mesh=self.mesh)` in that case.
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
        seq_len = x.shape[1]

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

        # --- sequence-parallel chunk: split the token sequence (and its
        # matching RoPE angles) across `sp_axis_name` -- `e`/`e0`/`context`/
        # `context_img` need no such treatment, see this module's docstring.
        cos, sin = freqs
        if self.sequence_parallel:
            sp_size = self.mesh.shape[self.sp_axis_name]
            assert seq_len % sp_size == 0, (
                f"sequence_parallel requires the patch token count ({seq_len}) "
                f"to be evenly divisible by the sequence-parallel size ({sp_size})")
            rank = jax.lax.axis_index(self.sp_axis_name)
            x = chunk_by_rank(x, 1, sp_size, rank)
            cos = chunk_by_rank(cos, 1, sp_size, rank)
            sin = chunk_by_rank(sin, 1, sp_size, rank)
        freqs = (cos, sin)

        # --- transformer blocks ---
        # `x` becomes the persistent float32 residual-stream accumulator
        # from here on -- see `WanDiTBlock`'s docstring for why (matches the
        # reference's `amp.autocast(dtype=torch.float32)`-wrapped gated
        # residual updates, which never cast back down to bf16, so its
        # residual stream is float32 for virtually the whole network).
        # Individual sub-layer matmuls still run in `input_dtype` (bf16);
        # only the accumulator itself stays wide.
        x = x.astype(jnp.float32)
        _effective_compute_dtype = self.compute_dtype if self.compute_dtype is not None else input_dtype
        for i in range(self.num_layers):
            x = WanDiTBlock(
                dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                qk_norm=self.qk_norm, cross_attn_norm=self.cross_attn_norm,
                eps=self.eps, mesh=self.mesh, sequence_parallel=self.sequence_parallel,
                sp_axis_name=self.sp_axis_name,
                compute_dtype=_effective_compute_dtype,
                name=f"blocks_{i}")(
                    x, context, e0, freqs, image_context=context_img)

        # --- head + unpatchify ---
        x = WanHead(
            self.dim, self.out_dim, self.patch_size, self.eps,
            name="head")(x, e)
        x = x.astype(input_dtype)

        if self.sequence_parallel:
            # Re-assemble the full token sequence (every device's local
            # output chunk, in rank order) before unpatchify.
            x = jax.lax.all_gather(x, self.sp_axis_name, axis=1, tiled=True)

        pt, ph, pw = self.patch_size
        x = x.reshape(b, t_p, h_p, w_p, pt, ph, pw, self.out_dim)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        x = x.reshape(b, t_p * pt, h_p * ph, w_p * pw, self.out_dim)
        return x
