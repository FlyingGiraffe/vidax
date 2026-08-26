"""LTX-2.5 diffusion (NATTEN) VAE decoder (Flax/JAX), single-full-volume-tile.

A structural port of the reference PyTorch `DiffusionVideoDecoder`/`NABlock`/
`DiffusionNABlock`/`NeighborhoodAttention3D`/`LinearPixelShuffleUpsample`/
`SwiGLU`/`AdaLNZero` from `refs/LTX-2-main/packages/ltx-core/src/ltx_core/
model/video_vae/{diffusion_video_decoder,transformer/{blocks,attention,
layers,swiglu,qkv,rope_math,det_attn_rope,combined/*}}.py`, for the real
`ltx-2.5-video-vae-bf16.safetensors` checkpoint (`config.vae._class_name ==
"CausalDiffusionVAE"`, decoder `NADiffusionDecoder`) -- see
`docs/lessons/ltx2_5_debugging.md` for the full port/verification writeup
and for why this decoder exists at all (the conv-decoder VAE in
`vidax.models.ltx2_5.vae` has a real, checkpoint-inherent periodic
artifact). **This decoder reduces that
artifact by roughly half (proportionally), it does not eliminate it** --
confirmed real, checkpoint-inherent behavior of the actual reference
`NADiffusionDecoder` too (not a porting bug in either VAE), verified by
directly comparing this port against the real PyTorch reference on the
same constant-latent diagnostic that found the conv decoder's own artifact
-- see `docs/lessons/ltx2_5_debugging.md`'s "The period-8 artifact is
checkpoint-inherent here too" entry.

**Scope: single full-volume tile only** -- no NATTEN tile-schedule/blend
machinery (`diffusion_tiling.py`'s `TilingConfig`/multi-tile pixel blend).
The reference's own `prepare_tile_schedule(..., tiling_config=None)` already
degenerates to exactly one untiled full-volume tile, so this is a real,
supported reference code path, not an approximation -- just not the one
that lets large resolutions fit in memory (see the plan doc's own
recommendation to validate single-tile correctness before investing in
tiling). Because there is no tiling, this module also skips two pieces of
tiling-only machinery entirely: the `chunked`/`dsl_kernels`/Blackwell
pathways (PyTorch-compile/GPU-specific, no JAX equivalent needed), and the
det/nested RoPE positional split the reference keeps for `torch.compile`
tracing reasons (`det_attn_rope.py` vs `combined/attn.py` apply the exact
same formula, just parameterized by different position arrays for tiled
vs. full-volume calls -- with only one full-volume tile ever processed
here, both call sites always use plain `arange(length)` positions, so one
shared RoPE helper below serves both).

**Real checkpoint deltas from the scoping doc's assumptions** (confirmed by
reading `ltx-2.5-video-vae-bf16.safetensors`'s own embedded
`config.vae`/tensor keys directly, not assumed): `stage5_kernel=(11, 11,
11)` (not `(3, 7, 7)`), `stage_channels=(2048, 1024, 512, 512, 256)`,
`stage_depths=(4, 6, 4, 2, 8)`, `default_num_inference_steps=1`,
`model_output_type="x0"` (top-level `config.vae`, confirming the scoping
doc's single-step-no-Euler-step simplification is the real recipe for this
checkpoint). The encoder's `latent_log_var` is `"constant"` here vs. the
conv checkpoint's `"uniform"` -- but the real reference `VideoEncoder.
forward` handles both identically (`conv_out` emits `latent_channels + 1`
channels either way, the extra channel is always discarded, only the mean
half is ever used -- confirmed from `video_vae.py`'s own `forward`), and
the checkpoint's own `encoder.conv_out.conv.weight` shape (`[129, ...]`,
matching `latent_channels + 1`) confirms this directly -- so
`vidax.models.ltx2_5.vae.Encoder` is reused here unchanged, no new encoder
code. The checkpoint also carries one top-level `decoder.type_emb`
parameter (shape `(128,)`) not referenced anywhere in
`refs/LTX-2-main`'s own loader/model code (`grep`-confirmed) -- treated as
an intentionally-unused/vestigial weight here, same as this port's existing
precedent of loading-but-ignoring unused audio-branch weights elsewhere;
flagged for the bit-exact check to confirm.

**Gate folding, already done upstream**: the reference's own checkpoint
loader (`model_configurator._build_diffusion_vae_decoder_sd_ops`) folds
`DiffusionNABlock`'s static `gate_msa`/`gate_mlp`/`gate_ctx` weights
directly into `attn.proj`/`mlp.w_down`/`context_proj`'s weights at load
time and drops the separate gate keys -- the real checkpoint's own raw
`decoder.diff_blocks.*` keys (confirmed via `safetensors.safe_open(...)
.keys()`) already have no such keys, i.e. **this checkpoint ships
post-fold**. `CombinedDiffusionNABlock.forward_combined`'s modulation
already only ever *reads* `(scale_msa, shift_msa, scale_mlp, shift_mlp)`
from `AdaLNZero`'s 7-chunk output (the 3 gate slots are computed but never
applied) -- so no gate-handling code is needed here at all, this module's
`AdaLNZero` output can be 7 chunks with 3 simply unused, matching the
reference class shape-for-shape without needing to reproduce the folding
step itself.

Q/K/V here ship **fused** in the checkpoint (`attn.qkv.{weight,bias}`,
shape `(3*dim, dim)`/`(3*dim,)`) -- split into thirds (q, k, v order,
confirmed from the reference's own `_split_fused_qkv_param`) by the
translator (`vidax.translator.mappings.ltx2_5.
map_ltx2_5_diffusion_decoder_keys`), not by this module -- this module's
`NeighborhoodAttention3D` exposes separate `to_q`/`to_k`/`to_v` Dense
layers, matching this port's existing convention (`vidax.models.ltx2_5.dit
.LTXAttention`) of one Flax submodule per logical projection.
"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from vidax.core.attention import RMSNorm
from vidax.models.ltx2_5.vae import _patchify, _split_subpixel, _unpatchify


def _default_rope_dim_split(head_dim: int) -> Tuple[int, int, int]:
    """See `rope_math.default_rope_dim_split` -- identical formula."""
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _rope_inv_freqs(dim: int, base: float) -> jnp.ndarray:
    """See `rope_math.rope_inv_freqs` -- identical formula, float32."""
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return jnp.asarray(1.0 / np.power(float(base), exponents), dtype=jnp.float32)


def _apply_abs_rope_axis(xc: jnp.ndarray, pos: jnp.ndarray, inv: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Absolute interleaved-pair RoPE on one axis chunk `xc[..., D]` (D even)
    -- see `rope_math.rot_abs_axis_impl`. **Not** the rotate-half convention
    `vidax.models.ltx2_5.rope` uses -- a genuinely different RoPE family,
    see this module's docstring.
    """
    pairs = xc.reshape(xc.shape[:-1] + (xc.shape[-1] // 2, 2))
    xe, xo = pairs[..., 0], pairs[..., 1]
    bshape = [1] * xc.ndim
    bshape[axis] = pos.shape[0]
    bshape[-1] = inv.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(bshape)
    c, s = jnp.cos(ang), jnp.sin(ang)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    return jnp.stack([re, ro], axis=-1).reshape(xc.shape)


def _apply_abs_rope(
    x: jnp.ndarray, rope_split: Tuple[int, int, int],
    inv_freqs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    positions: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Full per-head abs RoPE: `x` is `(B, T, H, W, NH, HD)`, `HD` split into
    `rope_split` (T, H, W) chunks, each rotated at absolute axis positions
    -- shared by every `NeighborhoodAttention3D` call site (det-stage and
    diffusion-stage alike, see module docstring for why one helper suffices
    here unlike the reference's det/nested split).
    """
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    pos_t, pos_h, pos_w = positions
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    xt = _apply_abs_rope_axis(x[..., :d_t], pos_t, inv_t, axis=1)
    xh = _apply_abs_rope_axis(x[..., d_t:d_t + d_h], pos_h, inv_h, axis=2)
    xw = _apply_abs_rope_axis(x[..., d_t + d_h:], pos_w, inv_w, axis=3)
    return jnp.concatenate([xt, xh, xw], axis=-1).astype(orig_dtype)


def _window_start(length: int, kernel: int) -> np.ndarray:
    """Per-index window start along one axis -- see `fallback_na.eager.
    _window_bounds` (non-causal branch). `kernel` is clamped to `length`
    first, so every window has exactly `kernel` valid positions -- no
    masking is ever needed downstream (a real simplification vs. the
    reference's own masked/batched eager fallback, which only needs masks
    because of its tile-batching mechanics, not because of the windowing
    semantics themselves).
    """
    kernel = min(kernel, length)
    idx = np.arange(length)
    lo = length - kernel
    return np.clip(idx - kernel // 2, 0, lo)


def _gather_window(x: jnp.ndarray, axis: int, length: int, kernel: int) -> jnp.ndarray:
    """Replace axis `axis` (static length `length`) with `(length, kernel)`
    -- the "im2col" gather that gives every query position its local NA
    window without materializing a full `(N, N)` attention matrix. Window
    bounds are static (numpy), computed once at trace time.
    """
    kernel = min(kernel, length)
    start = _window_start(length, kernel)
    gather_idx = jnp.asarray(start[:, None] + np.arange(kernel)[None, :])
    return jnp.take(x, gather_idx, axis=axis)


def neighborhood_attention_3d(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, kernel_size: Tuple[int, int, int],
) -> jnp.ndarray:
    """3D neighborhood attention over `(B, T, H, W, NH, HD)` tensors --
    `na3d` semantics (see `fallback_na/eager.py`'s docstring), `q` already
    pre-scaled by the caller. Returns `(B, T, H, W, NH, HD)`.

    Processes one query `(T, H)` row (all of `W` at once) per step, via a
    real `jax.lax.scan` over the flattened `T * H` index -- not a Python
    loop (which would unroll `T * H` copies of the step at trace time) and
    not a full materialization of the `(T, H, W) x (Kt, Kh, Kw)` window
    Cartesian product (which multiplies memory by the *product* of every
    window axis at once). The `Kt * Kh` window gather within each step is
    `jax.vmap`'d, not scanned -- already bounded to one row's own small
    footprint, so batching it avoids both unrolling and `scan`'s
    serialization cost where nothing forced it. See
    `docs/lessons/ltx2_5_debugging.md`'s "Diffusion (NATTEN) VAE decoder:
    the full compile-time and memory story" for the real OOM/hang
    measurements this design resolves, at each granularity tried before
    landing here.
    """
    b, t, h, w, nh, hd = q.shape
    kt, kh, kw = kernel_size
    kt_eff, kh_eff = min(kt, t), min(kh, h)
    start_t, start_h = _window_start(t, kt), _window_start(h, kh)
    softmax_dtype = jnp.promote_types(q.dtype, jnp.float32)

    def windows_w(x):
        # x: (B, W, NH, HD) -> (B, W, Kw, NH, HD)
        return _gather_window(x, 1, w, kw)

    windows_w_over_th = jax.vmap(windows_w, in_axes=1, out_axes=1)  # (B,KtKh,W,NH,HD) -> (B,KtKh,W,Kw,NH,HD)

    def scan_body(carry, xs):
        t_idx, h_idx, q_i = xs  # t_idx: (Kt,), h_idx: (Kh,), q_i: (B, W, NH, HD)
        k_th = jnp.take(jnp.take(k, t_idx, axis=1), h_idx, axis=2)  # (B, Kt, Kh, W, NH, HD)
        v_th = jnp.take(jnp.take(v, t_idx, axis=1), h_idx, axis=2)
        k_th = k_th.reshape(b, kt_eff * kh_eff, w, nh, hd)
        v_th = v_th.reshape(b, kt_eff * kh_eff, w, nh, hd)
        k_win = windows_w_over_th(k_th)  # (B, Kt*Kh, W, Kw, NH, HD)
        v_win = windows_w_over_th(v_th)
        k_win = jnp.moveaxis(k_win, 1, 2).reshape(b, w, -1, nh, hd)  # (B, W, Kt*Kh*Kw, NH, HD)
        v_win = jnp.moveaxis(v_win, 1, 2).reshape(b, w, -1, nh, hd)
        logits = jnp.einsum("bwnd,bwknd->bwnk", q_i, k_win).astype(softmax_dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(v_win.dtype)
        out_i = jnp.einsum("bwnk,bwknd->bwnd", weights, v_win)  # (B, W, NH, HD)
        return carry, out_i

    t_windows = np.stack([start_t[i] + np.arange(kt_eff) for i in range(t)])  # (T, Kt)
    h_windows = np.stack([start_h[i] + np.arange(kh_eff) for i in range(h)])  # (H, Kh)
    t_idx_flat = jnp.asarray(np.repeat(t_windows, h, axis=0))  # (T*H, Kt)
    h_idx_flat = jnp.asarray(np.tile(h_windows, (t, 1)))  # (T*H, Kh)
    q_flat = jnp.moveaxis(q.reshape(b, t * h, w, nh, hd), 1, 0)  # (T*H, B, W, NH, HD)

    _, outs = jax.lax.scan(scan_body, None, (t_idx_flat, h_idx_flat, q_flat))  # (T*H, B, W, NH, HD)
    return jnp.moveaxis(outs, 0, 1).reshape(b, t, h, w, nh, hd)


class NeighborhoodAttention3D(nn.Module):
    """One `NeighborhoodAttention3D` module: `to_q`/`to_k`/`to_v`/`to_out` +
    `q_norm`/`k_norm` (weighted `RMSNorm`, per-head) + absolute RoPE +
    the windowed attention primitive above. Shared by both `NABlock` (det
    stages) and `CombinedDiffusionNABlock` (stage 5) -- architecturally
    identical, matching the reference's own shared `NeighborhoodAttention3D`
    class.
    """
    dim: int
    kernel_size: Tuple[int, int, int]
    head_dim: int = 64
    rope_dim_split: Optional[Tuple[int, int, int]] = None
    rope_base: float = 10000.0
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, t, h, w, _ = x.shape
        num_heads = self.dim // self.head_dim
        rope_split = self.rope_dim_split or _default_rope_dim_split(self.head_dim)

        shape = (b, t, h, w, num_heads, self.head_dim)
        q = nn.Dense(self.dim, name="to_q")(x).reshape(shape)
        k = nn.Dense(self.dim, name="to_k")(x).reshape(shape)
        v = nn.Dense(self.dim, name="to_v")(x).reshape(shape)

        q = RMSNorm(self.head_dim, eps=self.eps, name="q_norm")(q)
        k = RMSNorm(self.head_dim, eps=self.eps, name="k_norm")(k)
        q = q * (self.head_dim ** -0.5)

        inv_freqs = (
            _rope_inv_freqs(rope_split[0], self.rope_base),
            _rope_inv_freqs(rope_split[1], self.rope_base),
            _rope_inv_freqs(rope_split[2], self.rope_base),
        )
        positions = (
            jnp.arange(t, dtype=jnp.float32), jnp.arange(h, dtype=jnp.float32), jnp.arange(w, dtype=jnp.float32))
        q = _apply_abs_rope(q, rope_split, inv_freqs, positions)
        k = _apply_abs_rope(k, rope_split, inv_freqs, positions)

        out = neighborhood_attention_3d(q, k, v, self.kernel_size)
        out = out.reshape(b, t, h, w, self.dim)
        # Named `to_out`, not `proj` -- reuses `vidax.core.sharding
        # .ROW_PARALLEL_NAMES`'s existing entry (Cosmos3's own attention
        # output) for Megatron-TP sharding, and avoids colliding with this
        # module's *other* `proj`-named Denses (`LinearPixelShuffleUpsample`/
        # `AdaLNZero`), which must stay replicated, not row-sharded.
        return nn.Dense(self.dim, name="to_out")(out)


class SwiGLU(nn.Module):
    """`w_down(silu(w_gate(x)) * w_up(x))` -- three unbiased `Dense`s.

    `num_tiles`: splits the flattened leading (token) dimension into this
    many contiguous chunks, processed one at a time through the same three
    `Dense`s (tied weights, called repeatedly -- standard Flax pattern) --
    the reference's own `SwiGLUTileSpec`/`DEFAULT_SWIGLU_TILES` machinery.
    `num_tiles=1` (the default) is a no-op, identical output. Ported as a
    plausible fix for this decoder's real stage-5 memory pressure, but
    measured to have **zero effect** (identical peak memory with tiling on
    vs. off) -- the actual fix was unrelated (Megatron tensor parallelism,
    see `DiffusionVideoDecoder`'s class docstring and
    `docs/lessons/ltx2_5_debugging.md`). Left wired in since it's a
    harmless, real optimization the reference itself uses, not because it
    was the fix that mattered here.
    """
    hidden_dim: int
    num_tiles: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dim = x.shape[-1]
        w_gate = nn.Dense(self.hidden_dim, use_bias=False, name="w_gate")
        w_up = nn.Dense(self.hidden_dim, use_bias=False, name="w_up")
        w_down = nn.Dense(dim, use_bias=False, name="w_down")

        def mlp(chunk):
            return w_down(nn.silu(w_gate(chunk)) * w_up(chunk))

        if self.num_tiles <= 1:
            return mlp(x)

        leading = x.shape[:-1]
        n = math.prod(leading)
        x_flat = x.reshape(n, dim)
        tile_size = -(-n // self.num_tiles)  # ceil division
        outs = [
            mlp(x_flat[i * tile_size:min((i + 1) * tile_size, n)])
            for i in range(self.num_tiles) if i * tile_size < n
        ]
        return jnp.concatenate(outs, axis=0).reshape(*leading, dim)


def _swiglu_hidden(dim: int, mlp_ratio: float = 4.0) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class NABlock(nn.Module):
    """Deterministic (det-stage) pre-norm block: NA -> SwiGLU, both residual,
    no AdaLN modulation."""
    dim: int
    kernel_size: Tuple[int, int, int]
    head_dim: int = 64
    eps: float = 1e-6
    mlp_num_tiles: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = RMSNorm(self.dim, eps=self.eps, name="norm1")(x)
        x = x + NeighborhoodAttention3D(
            self.dim, self.kernel_size, head_dim=self.head_dim, eps=self.eps, name="attn")(y)
        y2 = RMSNorm(self.dim, eps=self.eps, name="norm2")(x)
        x = x + SwiGLU(_swiglu_hidden(self.dim), num_tiles=self.mlp_num_tiles, name="mlp")(y2)
        return x


class LinearPixelShuffleUpsample(nn.Module):
    """Decoder-side resampler: `Linear` channel-expand, then channels-last
    pixel-shuffle (reuses `vidax.models.ltx2_5.vae._split_subpixel`, whose
    `(c, pt, ph, pw)` channel-split order matches the reference's own
    `"b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"` einops pattern
    exactly -- see module docstring). `drop_leading_frame` mirrors
    `vae.DepthToSpaceUpsample`'s existing `stride[0] == 2` convention.
    """
    stride: Tuple[int, int, int]
    out_channels_reduction_factor: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, drop_leading_frame: bool = True) -> jnp.ndarray:
        in_channels = x.shape[-1]
        pt, ph, pw = self.stride
        proj_out = math.prod(self.stride) * in_channels // self.out_channels_reduction_factor
        x = nn.Dense(proj_out, name="proj")(x)
        x = _split_subpixel(x, pt, ph, pw)
        if pt == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x


class AdaLNZero(nn.Module):
    """`t_emb -> SiLU -> Linear -> 7 chunks` (scale/shift/gate x
    msa/mlp, + gate_ctx). Only 4 of the 7 chunks (`scale_msa`, `shift_msa`,
    `scale_mlp`, `shift_mlp`) are ever consumed downstream -- see module
    docstring for why the 3 gate chunks are legitimately unused here (their
    real effect is already folded into this checkpoint's Linear weights).
    """
    dim: int
    NUM_CHUNKS = 7

    @nn.compact
    def __call__(self, t_emb: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        h = nn.Dense(self.NUM_CHUNKS * self.dim, name="proj")(nn.silu(t_emb))
        chunks = jnp.split(h, self.NUM_CHUNKS, axis=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


class CombinedDiffusionNABlock(nn.Module):
    """Stage-5 diffusion block: inject context (`context_proj`), AdaLN-
    modulated NA self-attention residual, AdaLN-modulated SwiGLU residual.
    `modulation` is `shared_adaln`'s 7-tuple; this block adds its own
    `scale_shift_table[i]` on top (`_modulation` in the reference) before
    using the 4 relevant chunks. Takes `context` and the running `x` half
    separately (not a concatenated `context_and_x` buffer, unlike the
    reference's in-place-`copy_` buffer reuse -- that's a PyTorch memory
    optimization with no JAX equivalent needed; functionally identical,
    see module docstring) and returns only the updated `x`.
    """
    dim: int
    kernel_size: Tuple[int, int, int]
    context_channels: int
    head_dim: int = 64
    eps: float = 1e-6
    mlp_num_tiles: int = 1

    @nn.compact
    def __call__(self, context: jnp.ndarray, x: jnp.ndarray, modulation: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        scale_shift_table = self.param("scale_shift_table", nn.initializers.zeros, (7, self.dim))
        mod = [modulation[i] + scale_shift_table[i].reshape(1, 1, 1, 1, -1) for i in range(7)]
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = mod

        x = x + nn.Dense(self.dim, name="context_proj")(context)

        y = RMSNorm(self.dim, eps=self.eps, name="norm1")(x)
        y = y * (1.0 + scale_msa) + shift_msa
        x = x + NeighborhoodAttention3D(
            self.dim, self.kernel_size, head_dim=self.head_dim, eps=self.eps, name="attn")(y)

        y2 = RMSNorm(self.dim, eps=self.eps, name="norm2")(x)
        y2 = y2 * (1.0 + scale_mlp) + shift_mlp
        x = x + SwiGLU(_swiglu_hidden(self.dim), num_tiles=self.mlp_num_tiles, name="mlp")(y2)
        return x


def _get_timestep_sinusoidal_embedding(timesteps: jnp.ndarray, num_channels: int = 256) -> jnp.ndarray:
    """Same as `vidax.models.ltx2_5.vae._get_timestep_sinusoidal_embedding`
    -- duplicated (not imported) since that symbol is private to `vae.py`
    and this is the same small, dependency-free formula already duplicated
    between `vidax.models.ltx_video.dit`/`vidax.models.ltx2_5.vae`."""
    half_dim = num_channels // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=jnp.float32) / half_dim
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)


def _all_stages_min_tile_size(
    stage_kernels: Tuple[Tuple[int, int, int], ...],
    upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """See `diffusion_tiling.all_stages_min_tile_size` -- per-axis
    latent-grid floor so every stage's NA sees dims >= its kernel_size."""
    cumulative = [(1, 1, 1)]
    ct, ch, cw = 1, 1, 1
    for stride, _ in upsamples:
        ct, ch, cw = ct * stride[0], ch * stride[1], cw * stride[2]
        cumulative.append((ct, ch, cw))
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return tuple(mins)


def _resize_axis_repeat_last(x: jnp.ndarray, axis: int, size: int) -> Tuple[jnp.ndarray, int]:
    """`resize_axis(..., mode="repeat_last")` restricted to this module's
    only two uses (pad or crop the *end* of an axis by repeating/dropping
    the last slice) -- see `diffusion_tiling.resize_axis`."""
    length = x.shape[axis]
    if length == size:
        return x, 0
    if length < size:
        pad = size - length
        last = jax.lax.slice_in_dim(x, length - 1, length, axis=axis)
        reps = jnp.repeat(last, pad, axis=axis)
        return jnp.concatenate([x, reps], axis=axis), pad
    return jax.lax.slice_in_dim(x, 0, size, axis=axis), 0


def _resize_axis_symmetric(x: jnp.ndarray, axis: int, size: int) -> Tuple[jnp.ndarray, Tuple[int, int]]:
    """`resize_axis(..., mode="symmetric")` -- edge-pad/crop both ends."""
    length = x.shape[axis]
    if length == size:
        return x, (0, 0)
    if length < size:
        need = size - length
        before, after = need // 2, need - need // 2
        first = jax.lax.slice_in_dim(x, 0, 1, axis=axis)
        last = jax.lax.slice_in_dim(x, length - 1, length, axis=axis)
        x = jnp.concatenate([jnp.repeat(first, before, axis=axis), x, jnp.repeat(last, after, axis=axis)], axis=axis)
        return x, (before, after)
    need = length - size
    before = need // 2
    return jax.lax.slice_in_dim(x, before, before + size, axis=axis), (before, need - before)


def crop_temporal_pad(
    out: jnp.ndarray, t_pad: int, upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...],
) -> jnp.ndarray:
    """Crops `self.context`'s own kernel-floor temporal pad (`t_pad`, a
    plain Python `int`) back off the final decoded pixel output -- shared
    by `DiffusionVideoDecoder.diffuse` and any caller driving
    `diffuse_prepare`/`diffuse_step`/`diffuse_finalize` directly (see that
    class's docstring), since both need the exact same crop.
    """
    if not t_pad:
        return out
    time_scale = math.prod(s[0][0] for s in upsamples)
    cropped, _ = _resize_axis_repeat_last(out, 1, out.shape[1] - t_pad * time_scale)
    return cropped


class DiffusionVideoDecoder(nn.Module):
    """Top-level `NADiffusionDecoder` port, config-driven (pass
    `vidax.models.ltx2_5.configs.DIFFUSION_VAE_CONFIG`-derived kwargs, read
    from the real checkpoint's own embedded metadata) -- same pattern as
    `vidax.models.ltx2_5.vae.LTXVAE`. **Single full-volume tile only** (see
    module docstring). No `encode` -- callers reuse
    `vidax.models.ltx2_5.vae.Encoder`/`LTXVAE.encode` directly, the encoder
    is unchanged between the conv and diffusion VAE checkpoints.

    `setup()`-based (not `@nn.compact`), exposing `context`/`diffuse` (and,
    within `diffuse`, `diffuse_prepare`/`diffuse_step`/`diffuse_finalize`)
    as independently-`.apply()`-able methods, split at real seams the
    reference itself has -- same fix shape as `WanVAEDecoder`'s
    `pre_process`/`decode_chunk` split (`docs/hardware_and_sharding.md`'s
    "The VAE decode 'hang' that wasn't a hang") and the DiT's own
    `--offload_dit_weights` per-chunk compilation: a single fused `jax.jit`
    over this decoder's 24 blocks (16 det + 8 diffusion) does not free
    per-block temporaries across the trace, and neither does a coarser
    context/diffuse-only split -- only splitting stage 5 down to one block
    per compiled program actually fits a v4 chip's HBM budget at this
    decoder's reference resolution. See `docs/lessons/ltx2_5_debugging.md`'s
    "Diffusion (NATTEN) VAE decoder: the full compile-time and memory
    story" for the real measurements at each granularity. Production
    callers should `jax.jit` `context`, then `diffuse_prepare`/
    `diffuse_step`(x8)/`diffuse_finalize`, as separate compiled programs
    (see `examples/generate_ltx2_5.py`); `decode()` below is a single-call
    convenience for small-scale testing/verification only -- wrapping *it*
    in one `jax.jit` reintroduces the same fused-trace memory problem this
    split exists to avoid.
    """
    in_channels: int = 128
    out_channels: int = 3
    patch_size: int = 4
    head_dim: int = 64
    stage_channels: Tuple[int, ...] = (2048, 1024, 512, 512, 256)
    stage_depths: Tuple[int, ...] = (4, 6, 4, 2, 8)
    stage_kernels: Tuple[Tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11))
    upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...] = (
        ((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2))
    stage5_kernel: Tuple[int, int, int] = (11, 11, 11)
    stage5_channels: Optional[int] = None
    t_emb_dim: int = 384
    timestep_scale_multiplier: float = 1000.0
    default_num_inference_steps: int = 1
    model_output_type: str = "x0"
    eps: float = 1e-6
    mlp_num_tiles: int = 4  # See `SwiGLU`'s docstring -- 4 matches the reference's own `DEFAULT_SWIGLU_TILES`.

    def setup(self):
        self.per_channel_statistics_mean = self.param(
            "per_channel_statistics_mean", nn.initializers.zeros, (self.in_channels,))
        self.per_channel_statistics_std = self.param(
            "per_channel_statistics_std", nn.initializers.ones, (self.in_channels,))

        self.conv_in = nn.Dense(self.stage_channels[0], name="conv_in")
        self.det_stages = [
            [
                NABlock(
                    self.stage_channels[stage_i], self.stage_kernels[stage_i], head_dim=self.head_dim,
                    eps=self.eps, mlp_num_tiles=self.mlp_num_tiles, name=f"det_stages_{stage_i}_{block_i}")
                for block_i in range(self.stage_depths[stage_i])
            ]
            for stage_i in range(4)
        ]
        self.det_upsamples = [
            LinearPixelShuffleUpsample(self.upsamples[stage_i][0], self.upsamples[stage_i][1],
                                        name=f"upsamples_{stage_i}")
            for stage_i in range(4)
        ]

        c5 = self.stage5_channels or self.stage_channels[-1]
        self.conv_in_x_t = nn.Dense(c5, name="conv_in_x_t")
        self.t_embedder_linear_1 = nn.Dense(self.t_emb_dim, name="t_embedder_linear_1")
        self.t_embedder_linear_2 = nn.Dense(self.t_emb_dim, name="t_embedder_linear_2")
        self.shared_adaln = AdaLNZero(c5, name="shared_adaln")
        self.diff_blocks = [
            CombinedDiffusionNABlock(
                c5, self.stage5_kernel, context_channels=self.stage_channels[-1], head_dim=self.head_dim,
                eps=self.eps, mlp_num_tiles=self.mlp_num_tiles, name=f"diff_blocks_{i}")
            for i in range(self.stage_depths[-1])
        ]
        self.norm_out = RMSNorm(c5, eps=self.eps, name="norm_out")
        self.conv_out = nn.Dense(self.out_channels * self.patch_size ** 2, name="conv_out")

    def context(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Stages 1-4: un-normalize -> `conv_in` -> 4x (`NABlock` x depth +
        upsample) -> crop the NATTEN last-frame ghost pad. Meant to be
        `.apply()`d (and `jax.jit`'d) on its own -- see class docstring.

        Returns:
            context: (B, F'', H'', W'', stage_channels[-1]) the stage-5
                conditioning volume.
            t_pad: a plain Python `int` (static, not an array -- depends
                only on `z`'s shape), the number of latent frames padded on
                to clear this decoder's own NA kernel floor. Threaded
                through to `diffuse` to crop the final pixel output back to
                the input's real (un-padded) temporal extent.
        """
        z = z * self.per_channel_statistics_std.astype(z.dtype) + self.per_channel_statistics_mean.astype(z.dtype)

        min_sizes = _all_stages_min_tile_size(self.stage_kernels, self.upsamples, self.stage5_kernel)
        z, t_pad = _resize_axis_repeat_last(z, 1, max(z.shape[1], min_sizes[0]))
        z, h_pad = _resize_axis_symmetric(z, 2, max(z.shape[2], min_sizes[1]))
        z, w_pad = _resize_axis_symmetric(z, 3, max(z.shape[3], min_sizes[2]))
        if h_pad != (0, 0) or w_pad != (0, 0):
            # Only the kernel-floor spatial pad path is unimplemented here --
            # a real code path in the reference (`ensure_min_latent_shape`'s
            # symmetric H/W branch) but one that only triggers for inputs
            # smaller than this decoder's own NA kernel footprint (spatially
            # tiny test clips), which this port's own verification/production
            # inputs are never expected to hit. Cropping it back out requires
            # the same latent->pixel spatial scale factor
            # `vidax.models.ltx2_5.configs.vae_scale_factors` already
            # computes elsewhere -- wire it through here if/when a real input
            # actually needs this path, rather than risk a silently-wrong
            # crop now.
            raise NotImplementedError(
                "DiffusionVideoDecoder.context: input latent H/W is smaller than this "
                "decoder's NA kernel floor -- spatial size-floor crop-back is not "
                "implemented yet (see comment above); pass a larger input.")

        natten_pad_frames = (self.stage_kernels[0][0] // 2) * 2
        z_padded, _ = _resize_axis_repeat_last(z, 1, z.shape[1] + natten_pad_frames)

        x = self.conv_in(z_padded)
        context = None
        for stage_i in range(4):
            for block in self.det_stages[stage_i]:
                x = block(x)
            x = self.det_upsamples[stage_i](x, drop_leading_frame=True)
            if stage_i == 3:
                context = x

        time_scale = math.prod(s[0][0] for s in self.upsamples)
        ghost = natten_pad_frames * time_scale
        content_t = max(context.shape[1] - ghost, 1)
        keep = min(context.shape[1], max(content_t, self.stage5_kernel[0]))
        context, _ = _resize_axis_repeat_last(context, 1, keep)
        return context, t_pad

    def diffuse_prepare(self, context: jnp.ndarray, x_t: jnp.ndarray, t_now: float) -> Tuple[jnp.ndarray, ...]:
        """Stage-5 pre-block work: timestep embedding -> `shared_adaln`
        modulation, and `conv_in_x_t(patchify(x_t))` -> the running `x_half`
        the block loop updates. Split out (with `diffuse_step`/
        `diffuse_finalize` below) so a production caller can `jax.jit` each
        of this decoder's 8 stage-5 blocks **separately** -- see the class
        docstring and `docs/lessons/ltx2_5_debugging.md` for why this
        granularity is the one that actually fits.
        """
        b = context.shape[0]
        t_arr = jnp.full((b,), t_now, dtype=jnp.float32)
        t_sin = _get_timestep_sinusoidal_embedding(
            self.timestep_scale_multiplier * t_arr, 256).astype(context.dtype)
        t_emb = self.t_embedder_linear_1(t_sin)
        t_emb = nn.silu(t_emb)
        t_emb = self.t_embedder_linear_2(t_emb)
        modulation = self.shared_adaln(t_emb)
        patched = _patchify(x_t, self.patch_size)
        x_half = self.conv_in_x_t(patched)
        return x_half, modulation

    def diffuse_step(
        self, context: jnp.ndarray, x_half: jnp.ndarray, modulation: Tuple[jnp.ndarray, ...], block_idx: int,
    ) -> jnp.ndarray:
        """One stage-5 `CombinedDiffusionNABlock` (see `diffuse_prepare`'s
        docstring for why this is its own method) -- `block_idx` is a
        static Python `int` (indexes `self.diff_blocks`, a plain Python
        list built in `setup()`), so `jax.jit`-ing a call to this method
        compiles one program per distinct `block_idx` -- 8 small compiles
        (cheap) instead of one huge fused one.
        """
        return self.diff_blocks[block_idx](context, x_half, modulation)

    def diffuse_finalize(self, x_half: jnp.ndarray) -> jnp.ndarray:
        """`norm_out` -> `conv_out` -> unpatchify -- the model's pixel
        prediction (see `diffuse_prepare`'s docstring for the split)."""
        x_half = self.norm_out(x_half)
        x_half = self.conv_out(x_half)
        return _unpatchify(x_half, self.patch_size)

    def diffuse(self, context: jnp.ndarray, x_t: jnp.ndarray, t_pad: int = 0) -> jnp.ndarray:
        """Stage 5: single (or, as a fallback, multi-step Euler) diffusion
        decode of noised patchified pixels `x_t`, conditioned on `context`
        (from `self.context`). Meant to be `.apply()`d (and `jax.jit`'d) on
        its own for small-scale/testing use -- production callers should
        instead `jax.jit` `diffuse_prepare`/`diffuse_step`/`diffuse_finalize`
        separately (see `diffuse_prepare`'s docstring for why).

        Args:
            context: the stage-5 conditioning volume from `self.context`.
            x_t: initial diffusion noise, pixel-space,
                `(B, F, ch*patch, cw*patch, out_channels)` matching
                `context`'s own T/H/W.
            t_pad: `self.context`'s own return value -- crops the final
                pixel output back to the real (un-padded) temporal extent.

        Returns:
            (B, F, H, W, out_channels) RGB video in [-1, 1].
        """
        timesteps = np.linspace(1.0, 1.0 / self.default_num_inference_steps, self.default_num_inference_steps)

        def diff_step(x_t, t_now):
            x_half, modulation = self.diffuse_prepare(context, x_t, t_now)
            for block_idx in range(len(self.diff_blocks)):
                x_half = self.diffuse_step(context, x_half, modulation, block_idx)
            return self.diffuse_finalize(x_half)

        if self.default_num_inference_steps == 1 and self.model_output_type == "x0":
            out = diff_step(x_t, float(timesteps[0]))
        else:
            for i in range(self.default_num_inference_steps):
                t_now = float(timesteps[i])
                model_out = diff_step(x_t, t_now)
                v_pred = model_out if self.model_output_type == "v" else (
                    (x_t.astype(jnp.float32) - model_out.astype(jnp.float32)) / max(t_now, 1e-8))
                t_next = float(timesteps[i + 1]) if i + 1 < len(timesteps) else 0.0
                dt = t_now - t_next
                x_t = (x_t.astype(jnp.float32) - dt * v_pred).astype(x_t.dtype)
            out = x_t if self.model_output_type != "x0" else model_out

        return crop_temporal_pad(out, t_pad, self.upsamples)

    def decode(self, z: jnp.ndarray, rng: jax.Array, x_t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Single-call convenience wrapper (`context` then `diffuse`) --
        small-scale/testing use only, see class docstring for why
        production callers should `jax.jit` `context`/`diffuse` separately
        instead of jitting a call to this method.

        Args:
            z: (B, F', H', W', in_channels) **normalized** latent -- same
                convention as `vidax.models.ltx2_5.vae.LTXVAE.decode`.
            rng: PRNG key for the initial diffusion noise `x_t`. Unused if
                `x_t` is given directly.
            x_t: optional explicit initial diffusion noise (pixel-space,
                patchified-resolution `(B, F, ch*patch, cw*patch,
                out_channels)`, matching the context volume's own T/H/W) --
                a verification-only hook (bit-exact checks need identical
                noise fed to both this port and the real PyTorch reference,
                which two independent RNGs can't guarantee) bypassing the
                internal `jax.random.normal` draw. Production callers should
                leave this `None`.

        Returns:
            (B, F, H, W, out_channels) RGB video in [-1, 1].
        """
        context, t_pad = self.context(z)
        if x_t is None:
            b, cf, ch, cw, _ = context.shape
            noise_shape = (b, cf, ch * self.patch_size, cw * self.patch_size, self.out_channels)
            x_t = jax.random.normal(rng, noise_shape, dtype=context.dtype)
        return self.diffuse(context, x_t, t_pad)
