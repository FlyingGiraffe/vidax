"""Cosmos-Predict2.5 Diffusion Transformer backbone (Flax/JAX).

A structural port of the reference PyTorch `MinimalV1LVGDiT`
(cosmos-predict2.5-main/cosmos_predict2/_src/predict2/networks/
minimal_v1_lvg_dit.py, subclassing `MiniTrainDIT` in minimal_v4_dit.py).
Architecture-only and config-driven, the same way
`vidax.models.wan.wan2_1`/`wan2_2`'s `WanDiT` are: this module's dataclass
defaults match the released 2B base checkpoint
(`checkpoints/Cosmos-Predict2.5-2B/base/pre-trained/.../model_ema_bf16.pt`);
`configs.py`'s `BASE_14B_CONFIG` overrides `dim`/`ffn_dim`/`num_heads`/
`num_layers` for the released 14B checkpoint
(`checkpoints/Cosmos-Predict2.5-14B/base/pre-trained/..._ema_bf16.pt`) --
every other hyperparameter (RoPE ratios, `adaln_lora_dim`, `context_dim`,
`timestep_scale`, ...) is identical between the two sizes. Weights load via
`vidax.translator.mappings.cosmos2_5` regardless of size (the state_dict key
structure doesn't depend on model width/depth).

Architecture, at a glance (see this repo's `docs/models/cosmos2_5.md` for the
full writeup, and `docs/lessons/cosmos2_5_debugging.md` for the
debugging history of getting this port producing correct output):

  - Patchify: `patch_size=(1, 2, 2)` (no temporal compression at the DiT
    level -- all of it happens in the VAE), via an explicit reshape+Dense
    (matching the reference's `Rearrange("b c (t r)(h m)(w n) -> b t h w
    (c r m n)") + Linear`) rather than a strided conv -- flatten order
    `(channel, t-in-patch, h-in-patch, w-in-patch)`, channel outermost.
    Unpatchify (in `CosmosFinalLayer`'s consumer below) uses a *different*,
    non-symmetric order -- see its own comment; assuming symmetry here was
    an actual, since-fixed bug (channel-scrambled output), not a
    simplification that happens to be safe.
  - Two extra concat channels beyond the 16 VAE latent channels: a padding
    mask (always 1s here -- vidax doesn't do multi-resolution batching) and
    a video-conditioning mask (1 = "this latent frame is given, not
    denoised" -- all 0s for pure text2video; see `__call__`'s docstring for
    how image2world/video2world conditioning would set this).
  - 3D axial RoPE (`vidax.models.cosmos2_5.rope`) on self-attention
    only; cross-attention has none.
  - AdaLN-LoRA modulation: unlike Wan's single 6-way-split-per-block
    modulation vector, each of a block's three sublayers (self-attn,
    cross-attn, MLP) has its *own* small modulation MLP
    (`silu -> Linear(dim, 256) -> Linear(256, 3*dim)`), and every one of
    them adds the same shared, global "AdaLN-LoRA" correction term computed
    once by `t_embedder` -- see `CosmosDiTBlock`'s docstring.
  - Per-*frame* (not per-sample) timestep conditioning: `timesteps` is
    `(B, T)`, letting video-conditioning frames carry a different
    (near-zero) noise level than the frames being generated, all within one
    forward pass -- the mechanism `examples/generate_cosmos2_5_*.py` uses
    for image2world/video2world.
  - Text conditioning: cross-attention against the Reason1 (Qwen2.5-VL-7B)
    text encoder's raw per-token hidden states (`context_raw_dim=100352`,
    `text_len=512`), down-projected once (`crossattn_proj`) to
    `context_dim=1024` before every block's cross-attention K/V.
"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.rope3d import sinusoidal_embedding_1d
from vidax.core.attention import RMSNorm, chunk_by_rank
from vidax.models.cosmos2_5.dit_layers import cosmos_attend
from vidax.models.cosmos2_5.rope import create_cosmos_rope3d_freqs


def _adaln_lora_modulation(
    emb: jnp.ndarray, adaln_lora: jnp.ndarray, out_dim: int, adaln_lora_dim: int, name: str,
) -> jnp.ndarray:
    """`silu -> Linear(dim, adaln_lora_dim, bias=False) -> Linear(adaln_lora_dim,
    out_dim, bias=False)`, plus the shared global AdaLN-LoRA correction term
    (already projected to `out_dim` width by the caller). Shared by every
    block sublayer and the final layer -- only `out_dim` (3*dim for blocks,
    2*dim for the final layer) and the global-correction slice differ.
    """
    h = nn.silu(emb)
    h = nn.Dense(adaln_lora_dim, use_bias=False, name=f"{name}_1")(h)
    h = nn.Dense(out_dim, use_bias=False, name=f"{name}_2")(h)
    return (h.astype(jnp.float32) + adaln_lora.astype(jnp.float32)[..., :out_dim])


def _expand_temporal(mod: jnp.ndarray, h_p: int, w_p: int) -> jnp.ndarray:
    """Broadcasts a per-frame `(B, T, D)` modulation tensor to per-token
    `(B, T*H*W, D)`, matching the patchified sequence's (T, H, W) flatten
    order (T outermost) -- each frame's modulation is repeated across all of
    that frame's `H*W` spatial tokens.
    """
    return jnp.repeat(mod, h_p * w_p, axis=1)


class CosmosDiTBlock(nn.Module):
    """One transformer block: self-attn -> cross-attn -> MLP, each with its
    own AdaLN-LoRA modulation MLP (see module docstring)."""
    dim: int
    ffn_dim: int
    num_heads: int
    head_dim: int
    context_dim: int
    adaln_lora_dim: int = 256
    eps: float = 1e-6
    mesh: Optional[Mesh] = None
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        emb: jnp.ndarray,
        adaln_lora: jnp.ndarray,
        context: jnp.ndarray,
        rope_freqs: Tuple[jnp.ndarray, jnp.ndarray],
        h_p: int,
        w_p: int,
    ) -> jnp.ndarray:
        # `emb`/`adaln_lora` are already this device's local T-axis chunk
        # under `sequence_parallel` (chunked once, by `CosmosDiT.__call__`,
        # before the block loop -- see its docstring) -- `_expand_temporal`
        # broadcasting them to per-token here, using the *full* (unsharded)
        # `h_p`/`w_p`, therefore naturally produces exactly this device's
        # local token-sequence chunk too, with no further chunking needed
        # inside the block itself.
        def modulate(name):
            mod = _adaln_lora_modulation(
                emb, adaln_lora, 3 * self.dim, self.adaln_lora_dim, name)
            mod = _expand_temporal(mod, h_p, w_p)
            return jnp.split(mod, 3, axis=-1)

        # --- self-attention ---
        shift_sa, scale_sa, gate_sa = modulate("adaln_modulation_self_attn")
        x_norm = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="layer_norm_self_attn")(x.astype(jnp.float32))
        x_norm = (x_norm * (1 + scale_sa) + shift_sa).astype(x.dtype)
        attn_out = cosmos_attend(
            x_norm, x_norm, self.dim, self.num_heads, self.head_dim, self.eps,
            "self_attn", rope_freqs=rope_freqs, mesh=self.mesh,
            sequence_parallel=self.sequence_parallel, sp_axis_name=self.sp_axis_name)
        x = (x.astype(jnp.float32) + gate_sa * attn_out.astype(jnp.float32)).astype(x.dtype)

        # --- cross-attention ---
        shift_ca, scale_ca, gate_ca = modulate("adaln_modulation_cross_attn")
        x_norm = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="layer_norm_cross_attn")(x.astype(jnp.float32))
        x_norm = (x_norm * (1 + scale_ca) + shift_ca).astype(x.dtype)
        cross_out = cosmos_attend(
            x_norm, context, self.dim, self.num_heads, self.head_dim, self.eps,
            "cross_attn", rope_freqs=None, mesh=self.mesh,
            sequence_parallel=self.sequence_parallel, sp_axis_name=self.sp_axis_name)
        x = (x.astype(jnp.float32) + gate_ca * cross_out.astype(jnp.float32)).astype(x.dtype)

        # --- MLP (GPT2FeedForward: Linear -> GELU -> Linear, no bias) ---
        shift_mlp, scale_mlp, gate_mlp = modulate("adaln_modulation_mlp")
        x_norm = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="layer_norm_mlp")(x.astype(jnp.float32))
        x_norm = (x_norm * (1 + scale_mlp) + shift_mlp).astype(x.dtype)
        # `mlp_layer1` is column-parallel -- see `vidax.models.wan.common
        # .dit_layers.attend`'s identical comment for why its declared
        # output width must be halved under `sequence_parallel` (a no-op
        # when 'tp' has size 1).
        tp_size = self.mesh.shape["tp"] if (self.sequence_parallel and self.mesh is not None) else 1
        h = nn.Dense(self.ffn_dim // tp_size, use_bias=False, name="mlp_layer1")(x_norm)
        h = nn.gelu(h, approximate=False)
        h = nn.Dense(self.dim, use_bias=False, name="mlp_layer2")(h)
        # `mlp_layer2` is row-parallel -- see `vidax.models.wan.common
        # .dit_layers.attend`'s identical comment for why this manual reduce
        # is only needed under `sequence_parallel`, and why it's a safe
        # no-op otherwise.
        if self.sequence_parallel:
            h = jax.lax.psum(h, "tp")
        x = (x.astype(jnp.float32) + gate_mlp * h.astype(jnp.float32)).astype(x.dtype)
        return x


class CosmosFinalLayer(nn.Module):
    """AdaLN-modulated projection back to patch-pixel space."""
    dim: int
    out_dim: int
    adaln_lora_dim: int = 256
    eps: float = 1e-6

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, emb: jnp.ndarray, adaln_lora: jnp.ndarray,
        h_p: int, w_p: int,
    ) -> jnp.ndarray:
        mod = _adaln_lora_modulation(
            emb, adaln_lora, 2 * self.dim, self.adaln_lora_dim, "adaln_modulation")
        mod = _expand_temporal(mod, h_p, w_p)
        shift, scale = jnp.split(mod, 2, axis=-1)

        x_norm = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=self.eps,
            name="norm")(x.astype(jnp.float32))
        x_norm = (x_norm * (1 + scale) + shift).astype(x.dtype)
        return nn.Dense(self.out_dim, use_bias=False, name="linear")(x_norm)


class CosmosDiT(nn.Module):
    """Cosmos-Predict2.5 DiT. Dataclass defaults match the released 2B base
    checkpoint; pass `**configs.BASE_14B_CONFIG` for the 14B checkpoint (see
    `configs.py` for exactly which fields differ).

    Two parallelism strategies, both ported directly from Wan's DiTs (see
    `vidax.models.wan.wan2_1.dit`/`wan2_2.dit`'s module docstrings for the
    fuller mechanism writeups -- the reasoning is identical here, just
    applied to Cosmos's own attention/modulation shapes):

    - `mesh` alone (Megatron-style tensor parallelism): shards attention
      heads and FFN channels across `mesh`'s `'tp'` axis via
      `vidax.core.sharding.shard_wan_params` (its column/row-parallel name
      table covers `CosmosDiT`'s own Dense-layer names too) -- every device
      holds the *full* token sequence but only its own slice of heads/
      channels. `cosmos_attend` already threads `mesh` through to
      `dot_product_attention`, so this needs no other change here. At 2B
      params this mainly matters for very high resolution/frame-count runs
      (the 2B model fits comfortably on a single chip otherwise); at 14B it
      matters for fitting the weights themselves, the same tradeoff Wan's
      14B-class DiTs document.
    - `sequence_parallel=True` (DeepSpeed-Ulysses): shards the *token
      sequence itself* between blocks instead, reshuffling to a
      head-sharded full-sequence view only for the duration of
      self-attention (`cosmos_attend`'s `sequence_parallel_self_attention`
      path). Requires the whole `CosmosDiT.apply(...)` call to run inside
      `jax.experimental.shard_map.shard_map(..., mesh=self.mesh)` -- see
      `examples/generate_cosmos2_5.py` for how that's wired up -- and DiT
      parameters should be left **replicated**, not tensor-sharded, when
      this is enabled (cutting activation memory by sharding the sequence,
      not weights). Requires `t_p` (the *latent frame count*, not the full
      patch-token count) to divide evenly by the sequence-parallel size --
      see `__call__`'s chunking comment for why the chunk boundary has to
      land on a frame boundary, not an arbitrary token offset.
    """
    dim: int = 2048
    ffn_dim: int = 8192
    num_heads: int = 16
    head_dim: int = 128
    num_layers: int = 28
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16  # VAE latent channels.
    out_channels: int = 16
    context_raw_dim: int = 100352  # Reason1: 28 Qwen2.5-VL-7B layers x 3584.
    context_dim: int = 1024  # Post-`crossattn_proj` cross-attention width.
    adaln_lora_dim: int = 256
    eps: float = 1e-6
    theta: float = 10000.0
    rope_h_extrapolation_ratio: float = 3.0
    rope_w_extrapolation_ratio: float = 3.0
    rope_t_extrapolation_ratio: float = 1.0
    timestep_scale: float = 0.001  # reference `MinimalV1LVGDiT`'s trained value for this checkpoint.
    mesh: Optional[Mesh] = None
    sequence_parallel: bool = False
    sp_axis_name: str = "sp"

    @nn.compact
    def __call__(
        self,
        latents: jnp.ndarray,
        timesteps: jnp.ndarray,
        context: jnp.ndarray,
        padding_mask: Optional[jnp.ndarray] = None,
        condition_video_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            latents: (B, T, H, W, in_channels) noisy VAE-latent video.
            timesteps: (B,) or (B, T) diffusion timesteps -- per-*frame* when
                (B, T): video2world/image2world conditioning frames carry a
                separate (near-zero) timestep from the frames being
                generated, all in a single forward pass. A (B,) input is
                broadcast to every frame (the plain text2video case).
            context: (B, L, context_raw_dim) raw Reason1 text embeddings,
                L <= 512 (zero-padded to 512 if shorter, matching the
                reference's fixed `NUM_EMBEDDING_PADDING_TOKENS`).
            padding_mask: (B, T, H, W, 1) optional; defaults to all-ones
                (vidax assumes one shared (T, H, W) grid per batch, so this
                is only useful if a caller has a real reason to pad).
            condition_video_mask: (B, T, H, W, 1) optional; defaults to
                all-zeros (pure text2video -- no conditioning frames). For
                image2world/video2world, set to 1 for the latent frames
                whose content is given (and blended into `latents` in place
                of noise by the caller -- see this module's package
                docstring / the reference's `denoise()` frame-replacement
                logic, not reproduced here since it's sampling-loop
                orchestration, not DiT architecture).

        Returns:
            (B, T, H, W, out_channels) denoised x0 prediction.
        """
        b, t, h, w, _ = latents.shape
        pt, ph, pw = self.patch_size
        assert t % pt == 0 and h % ph == 0 and w % pw == 0, (
            f"latent grid {(t, h, w)} must be divisible by patch_size {self.patch_size}")
        t_p, h_p, w_p = t // pt, h // ph, w // pw

        if padding_mask is None:
            padding_mask = jnp.ones((b, t, h, w, 1), dtype=latents.dtype)
        if condition_video_mask is None:
            condition_video_mask = jnp.zeros((b, t, h, w, 1), dtype=latents.dtype)
        # Callers commonly build these masks as plain float32 (e.g. for
        # arithmetic with a float32 conditional-timestep constant elsewhere
        # in the sampling loop) -- `jnp.concatenate` silently *promotes the
        # whole result* to the widest input dtype, which would otherwise
        # upcast every downstream activation in this forward pass to
        # float32 even when running the DiT itself in bf16, surfacing only
        # much later as a hard-to-place dtype-mismatch error in attention
        # (query, from this now-float32 `x`, vs. key, from `context`, which
        # was never touched by this bug). Cast explicitly rather than rely
        # on every caller getting this right.
        padding_mask = padding_mask.astype(latents.dtype)
        condition_video_mask = condition_video_mask.astype(latents.dtype)
        # Channel order matches the reference exactly: `MinimalV1LVGDiT.forward`
        # appends `condition_video_input_mask` to the latents first, then
        # `MiniTrainDIT.prepare_embedded_sequence` appends `padding_mask`
        # last -- i.e. [latents, condition_mask, padding_mask], NOT
        # [latents, padding_mask, condition_mask]. The two mask channels
        # carry opposite-meaning constants for plain text2video (condition
        # mask all-zero, padding mask all-one), so swapping their order
        # silently fed the trained x_embedder weights the wrong per-channel
        # semantics on every call.
        x = jnp.concatenate([latents, condition_video_mask, padding_mask], axis=-1)

        # --- patchify: matches the reference's Rearrange
        # "b c (t r)(h m)(w n) -> b t h w (c r m n)" (channel outermost,
        # then t-in-patch/h-in-patch/w-in-patch), applied to our
        # channel-last layout.
        in_ch = x.shape[-1]
        x = x.reshape(b, t_p, pt, h_p, ph, w_p, pw, in_ch)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)  # (b, t_p, h_p, w_p, C, pt, ph, pw)
        x = x.reshape(b, t_p, h_p, w_p, in_ch * pt * ph * pw)
        x = nn.Dense(self.dim, use_bias=False, name="x_embedder_proj_1")(x)
        x = x.reshape(b, t_p * h_p * w_p, self.dim)

        # --- timestep embedding: `emb` is the *raw* sinusoidal embedding
        # (RMSNorm'd), fed to every block's own small modulation MLP;
        # `adaln_lora` is a separate, shared global correction term added
        # into every one of those MLPs' outputs -- see module docstring.
        if timesteps.ndim == 1:
            timesteps = jnp.broadcast_to(timesteps[:, None], (b, t_p))
        # The reference's `MinimalV1LVGDiT.forward` rescales the incoming
        # (already-EDM-preconditioned) timestep by a second, DiT-internal
        # `timestep_scale` before the sinusoidal embedding -- distinct from
        # `RectifiedFlowScaling`'s own `t_scaling_factor` applied upstream
        # in the sampling loop. Missing this leaves the network conditioned
        # on a ~1000x out-of-distribution timestep at every sampling step.
        timesteps = timesteps * self.timestep_scale
        t_sin = sinusoidal_embedding_1d(self.dim, timesteps.reshape(-1)).reshape(b, t_p, self.dim)
        adaln_lora = nn.Dense(self.dim, use_bias=False, name="t_embedder_1_linear_1")(t_sin)
        adaln_lora = nn.silu(adaln_lora)
        adaln_lora = nn.Dense(3 * self.dim, use_bias=False, name="t_embedder_1_linear_2")(adaln_lora)
        emb = RMSNorm(self.dim, eps=self.eps, name="t_embedding_norm")(t_sin)

        # --- text context: down-project once, shared by every block.
        text_pad = 512 - context.shape[1]
        assert text_pad >= 0, "context length exceeds the fixed 512-token Reason1 embedding length"
        if text_pad > 0:
            context = jnp.pad(context, ((0, 0), (0, text_pad), (0, 0)))
        context = nn.Dense(self.context_dim, name="crossattn_proj_0")(context)
        context = nn.gelu(context, approximate=False)

        rope_freqs = create_cosmos_rope3d_freqs(
            t_p, h_p, w_p, self.head_dim, theta=self.theta,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio)

        # --- sequence-parallel chunk: split the *frame* axis of `emb`/
        # `adaln_lora` (T, not the full T*H*W token sequence) across
        # `sp_axis_name`, and the full patchified token sequence `x` (and
        # its matching `cos`/`sin` RoPE angles) the same way. Both chunkings
        # agree exactly: `x`'s flatten order is (t, h, w) with t outermost
        # (see the patchify comment above), so a contiguous `1/sp_size`
        # slice of the frame axis and a contiguous `1/sp_size` slice of the
        # full token sequence cover the *same* frames/tokens, as long as
        # `t_p` (not just `t_p*h_p*w_p`) is itself evenly divisible by
        # `sp_size` -- required below. `context` needs no such treatment
        # (small, already fully replicated -- `cosmos_attend`'s
        # cross-attention path runs as a plain local call under
        # `sequence_parallel`, not the distributed one).
        cos, sin = rope_freqs
        if self.sequence_parallel:
            sp_size = self.mesh.shape[self.sp_axis_name]
            assert t_p % sp_size == 0, (
                f"sequence_parallel requires the latent frame count ({t_p}) "
                f"to be evenly divisible by the sequence-parallel size ({sp_size})")
            rank = jax.lax.axis_index(self.sp_axis_name)
            x = chunk_by_rank(x, 1, sp_size, rank)
            cos = chunk_by_rank(cos, 1, sp_size, rank)
            sin = chunk_by_rank(sin, 1, sp_size, rank)
            emb = chunk_by_rank(emb, 1, sp_size, rank)
            adaln_lora = chunk_by_rank(adaln_lora, 1, sp_size, rank)
        rope_freqs = (cos, sin)

        for i in range(self.num_layers):
            x = CosmosDiTBlock(
                dim=self.dim, ffn_dim=self.ffn_dim, num_heads=self.num_heads,
                head_dim=self.head_dim, context_dim=self.context_dim,
                adaln_lora_dim=self.adaln_lora_dim, eps=self.eps, mesh=self.mesh,
                sequence_parallel=self.sequence_parallel, sp_axis_name=self.sp_axis_name,
                name=f"blocks_{i}")(x, emb, adaln_lora, context, rope_freqs, h_p, w_p)

        out_patch_dim = math.prod(self.patch_size) * self.out_channels
        x = CosmosFinalLayer(
            self.dim, out_patch_dim, self.adaln_lora_dim, self.eps,
            name="final_layer")(x, emb, adaln_lora, h_p, w_p)

        if self.sequence_parallel:
            # Re-assemble the full token sequence (every device's local
            # output chunk, in rank order) before unpatchify.
            x = jax.lax.all_gather(x, self.sp_axis_name, axis=1, tiled=True)

        # --- unpatchify: NOT the inverse of the patchify reshape's own
        # channel order -- the reference's `unpatchify` uses a *different*
        # per-patch flatten order than `PatchEmbed` does: "B T H W (p1 p2 t
        # C) -> B C (T t) (H p1) (W p2)", i.e. (height-patch, width-patch,
        # temporal-patch, channel) with channel *innermost*, versus
        # patchify's (channel, temporal-patch, height-patch, width-patch)
        # with channel outermost (see the patchify comment above). These
        # are asymmetric in the actual reference, not just two equivalent
        # ways of writing the same thing -- assuming symmetry here (an
        # earlier version of this code did) silently scrambles which of the
        # 64 final_layer output values lands at which (channel, row-in-
        # patch, col-in-patch) position, which decodes through the VAE as a
        # blocky grid of locally-scrambled colors rather than a coherent
        # image.
        x = x.reshape(b, t_p, h_p, w_p, ph, pw, pt, self.out_channels)
        x = x.transpose(0, 1, 6, 2, 4, 3, 5, 7)  # (b, t_p, pt, h_p, ph, w_p, pw, C)
        x = x.reshape(b, t_p * pt, h_p * ph, w_p * pw, self.out_channels)
        return x
