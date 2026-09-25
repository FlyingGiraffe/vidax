"""Dot-product attention and QK-normalization primitives shared by DiT/T5 blocks."""
import os
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import Mesh, PartitionSpec as P

_FLASH_BLOCK = 128  # Fixed tile size of jax's TPU Pallas flash-attention kernel.

# Tuned Pallas flash-attention tile sizes per TPU generation. The kernel's
# upstream default (128x128 tiles) is severely under-utilized on TPU v7
# (Ironwood) at video-DiT sequence lengths: a (B=2, S=32768, H=12, D=128)
# bf16 self-attention measured 1026ms with the default vs 69ms with
# 2048/1024/1024 tiles (~15x; 12.8 -> 191 TFLOP/s). Older generations (v4)
# were validated at the default and keep it. Override via the
# VIDAX_FLASH_BLOCK_SIZES env var ("block_q,block_k_major,block_k", e.g.
# "2048,1024,1024") -- mostly useful for tuning on future hardware.
_FLASH_BLOCK_SIZES_BY_DEVICE_KIND = {
    "TPU7x": (2048, 1024, 1024),  # (block_q, block_k_major, block_k)
}


def _flash_block_sizes():
    """Returns the configured (block_q, block_k_major, block_k) for the
    current device kind, or None to use the kernel's own default."""
    env = os.environ.get("VIDAX_FLASH_BLOCK_SIZES")
    if env:
        if env.strip().lower() == "default":
            return None  # force the kernel's built-in 128-tile default
        parts = tuple(int(p) for p in env.split(","))
        assert len(parts) == 3, "VIDAX_FLASH_BLOCK_SIZES must be 'block_q,block_k_major,block_k'"
        return parts
    return _FLASH_BLOCK_SIZES_BY_DEVICE_KIND.get(jax.devices()[0].device_kind)


# --- Splash attention (jax.experimental.pallas.ops.tpu.splash_attention) ---
#
# On TPU v7 the splash kernel beats the (tile-tuned) legacy flash kernel by a
# further ~1.6x at video-DiT shapes: (B=2, S=32768, H=12, D=128) bf16 measures
# 43ms (307 TFLOP/s) vs 69ms for the best legacy BlockSizes, and matches a
# chunked fp32 einsum reference to 3.3e-4 max abs diff (better than the legacy
# kernel's own bf16 error). On short-KV cross-attention (S_q=32760, S_kv=512)
# the gap is much bigger: 1.0ms vs 14.7ms (~15x) with (2048,512,512) tiles --
# the legacy kernel's 128-tile default is latency-bound on the long Q side.
#
# Two splash implementations are available on v7, picked by VIDAX_V7_KERNEL
# (default "auto" = maxdiff):
# - "maxdiff": the MaxDiffusion-vendored kernel (`_maxdiffusion_splash`,
#   self-contained, Apache-2.0) -- 35ms at the self-attention shape above
#   (375 TFLOP/s, ~1.2x over jax-splash) thanks to its head-dim-major output
#   layout and inner-KV sub-blocking (block_kv_compute_in), which also let it
#   run block_q=4096+ inside the 64MiB VMEM budget where jax-splash OOMs.
#   Differences vs jax-splash: base-2 exp by default (we fold log2(e) into the
#   softmax scale), actual-vs-padded sequence lengths instead of segment ids
#   for padding, and an (H, D, S) output layout (we transpose back).
# - "splash": the jax.experimental kernel (kept as fallback; cross-attention
#   with KV<512 tokens, or if the vendored file ever fails to import).
#
# Shared integration notes: neither applies a softmax scale internally (the
# caller must pre-scale Q), and both take per-batch-item (H, S, D) tensors
# (we vmap over the batch). Their kernel closures must not be cached across
# jit traces (jax-splash's MaskInfo closes over traced arrays), so we rebuild
# per trace -- a few ms of host time at compile time only.
# Anything ineligible (fp32-DiT pipelines, sub-512-token contexts, bias
# carriers like Cosmos3's padded-text attention) keeps the tuned legacy
# kernel. Set VIDAX_V7_KERNEL=legacy or =splash to override.
_MD_SELF_ATTN_BLOCKS = (4096, 1024, 1024)   # (block_q, block_kv, block_kv_compute)
_MD_CROSS_ATTN_BLOCKS = (2048, 512, 512)    # short-KV cross-attention
_SPLASH_SELF_ATTN_BLOCKS = (2048, 2048, 1024)
_SPLASH_CROSS_ATTN_BLOCKS = (2048, 512, 512)


def _splash_block_sizes_for(sq: int, sk: int):
    """Returns splash (block_q, block_kv, block_kv_compute) for a q x kv shape
    on this device, or None if splash shouldn't handle it. Caller picks the
    kernel preference; this only encodes shape eligibility."""
    if jax.devices()[0].device_kind != "TPU7x":
        return None
    if sq < 2048:
        return None
    if sk >= 2048:
        return _SPLASH_SELF_ATTN_BLOCKS
    if sk >= 512:
        return _SPLASH_CROSS_ATTN_BLOCKS
    return None


_md_available: Optional[bool] = None


def _md_importable() -> bool:
    """Whether the vendored MaxDiffusion kernel imports cleanly."""
    global _md_available
    if _md_available is None:
        try:
            from vidax.core import _maxdiffusion_splash  # noqa: F401
            _md_available = True
        except ImportError:
            _md_available = False
    return _md_available


def _md_splash_attention_tpu(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, scale: float,
    block_q: int, block_kv: int, block_kv_compute: int,
) -> jnp.ndarray:
    """MaxDiffusion-vendored splash kernel (see module notes above)."""
    from vidax.core import _maxdiffusion_splash as mds
    import math

    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))
    qt, sq0 = _pad_seq(qt, axis=2, multiple=math.lcm(block_q, _FLASH_BLOCK))
    kt, sk0 = _pad_seq(kt, axis=2, multiple=math.lcm(block_kv, _FLASH_BLOCK))
    vt, _ = _pad_seq(vt, axis=2, multiple=math.lcm(block_kv, _FLASH_BLOCK))

    # The kernel uses base-2 exp and applies no softmax scale itself: fold
    # both log2(e) and our scale into Q. Padding exclusion uses its
    # actual-vs-padded length truncation, not segment ids.
    qt = qt * jnp.asarray(scale * 1.4426950408889634, dtype=qt.dtype)
    kernel = mds.make_splash_mha(
        mds._BlockSizes(block_q=block_q, block_kv=block_kv,
                        block_kv_compute=block_kv_compute),
        orig_q_seq_len=sq0, orig_kv_seq_len=sk0)
    out = jax.vmap(lambda qq, kk, vv: kernel(qq, kk, vv))(qt, kt, vt)
    # Kernel output is (H, D, S) per batch item (head-dim-major).
    out = jnp.transpose(out, (0, 3, 1, 2))
    return out  # already sliced to sq0 by the kernel's actual-length indexing

_splash_available: Optional[bool] = None


def _splash_importable() -> bool:
    """Whether the splash_attention package is importable in this jax build
    (it moved between `jax.experimental.pallas.ops.tpu` layouts across
    releases; absence must degrade to the legacy kernel, not crash)."""
    global _splash_available
    if _splash_available is None:
        try:
            from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: F401
                splash_attention_kernel, splash_attention_mask)
            _splash_available = True
        except ImportError:
            _splash_available = False
    return _splash_available


def _splash_attention_tpu(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, scale: float,
    block_q: int, block_kv: int, block_kv_compute: int,
) -> jnp.ndarray:
    """Splash flash-attention (see module notes above). No bias/mask support
    here -- callers with an additive bias stay on the legacy kernel."""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash, splash_attention_mask as mask_lib)
    import math

    b, sq, h, d = q.shape

    pad_q = math.lcm(block_q, _FLASH_BLOCK)
    pad_kv = math.lcm(block_kv, _FLASH_BLOCK)
    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))
    qt, sq0 = _pad_seq(qt, axis=2, multiple=pad_q)
    kt, sk0 = _pad_seq(kt, axis=2, multiple=pad_kv)
    vt, _ = _pad_seq(vt, axis=2, multiple=pad_kv)

    segment_ids = None
    if qt.shape[2] != sq0 or kt.shape[2] != sk0:
        q_ids = jnp.where(jnp.arange(qt.shape[2]) < sq0, 1, 0)[None, :]
        kv_ids = jnp.where(jnp.arange(kt.shape[2]) < sk0, 1, 0)[None, :]
        segment_ids = splash.SegmentIds(
            q=jnp.broadcast_to(q_ids, (b, qt.shape[2])),
            kv=jnp.broadcast_to(kv_ids, (b, kt.shape[2])))

    # NB: no kernel-object caching here. The kernel closes over MaskInfo
    # arrays that `make_splash_mha_single_device` builds with jnp ops; if this
    # runs under a jit/shard_map trace (it always does in practice), those
    # arrays are tracers of that trace. Caching the kernel leaks them into
    # later traces (UnexpectedTracerError), and host-converting them at build
    # time is itself an np.asarray(tracer) (TracerArrayConversionError).
    # Rebuilding per call costs a few ms of host time *at trace time only*
    # (the compiled program is then reused for every layer/step) and the
    # MaskInfo is baked into the program as constants -- correct by
    # construction.
    mask = mask_lib.MultiHeadMask(
        tuple(mask_lib.FullMask((qt.shape[2], kt.shape[2])) for _ in range(h)))
    kernel = splash.make_splash_mha_single_device(
        mask,
        block_sizes=splash.BlockSizes(
            block_q=block_q, block_kv=block_kv, block_kv_compute=block_kv_compute))

    # Splash applies no softmax scale internally -- fold ours into Q.
    qt = qt * jnp.asarray(scale, dtype=qt.dtype)
    out = jax.vmap(
        lambda qq, kk, vv, seg: kernel(qq, kk, vv, segment_ids=seg)
    )(qt, kt, vt, segment_ids)
    out = out[:, :, :sq0, :]
    return jnp.transpose(out, (0, 2, 1, 3))


def _v7_kernel_pref():
    """Which flash-attention kernel to prefer on v7: 'auto' (MaxDiffusion
    vendored kernel, falling back to jax-splash), 'maxdiff', 'splash', or
    'legacy'. Honors VIDAX_V7_KERNEL."""
    return os.environ.get("VIDAX_V7_KERNEL", "auto")


def chunk_by_rank(x: jnp.ndarray, axis: int, sp_size: int, rank: jnp.ndarray) -> jnp.ndarray:
    """Slices out this device's contiguous `1/sp_size` share of `x` along
    `axis`, indexed by a *traced* `rank` (e.g. `jax.lax.axis_index(...)`
    inside `shard_map`) -- `sp_size` must be a static Python int (chunk size
    has to be known at trace time), so this only supports even splits.

    Model-family-agnostic: shared by every sequence-parallel DiT in this
    repo (Wan2.1/2.2 via `vidax.models.wan.common.dit_layers`, which
    re-exports this rather than defining its own copy; Cosmos-Predict2.5 via
    `vidax.models.cosmos2_5.dit`) for chunking the token sequence
    (and, where it also varies per token/frame, the timestep-modulation
    state) before the block loop.
    """
    size = x.shape[axis] // sp_size
    return jax.lax.dynamic_slice_in_dim(x, rank * size, size, axis=axis)


class RMSNorm(nn.Module):
    """RMSNorm matching Wan2.1's ``WanRMSNorm``: normalized in float32, cast back."""
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.eps)
        return (normed.astype(orig_dtype)) * scale.astype(orig_dtype)


class TPShardedRMSNorm(nn.Module):
    """RMSNorm whose reduction spans a Megatron-sharded feature axis.

    Ordinary `RMSNorm` normalizes over its own local last axis -- correct
    when the whole feature axis is present locally, but wrong when it's
    column-sharded across devices (Wan's Q/K-RMSNorm, which -- unlike
    Cosmos's per-head norm -- reduces over the *entire* projected `dim`,
    before that gets split into heads; see
    `vidax.models.wan.common.dit_layers.attend`). A per-device-local
    mean-square would only see this device's slice of channels, not the
    true global one the reference model computes, so this sums each
    device's local sum-of-squares via `jax.lax.psum('tp')` first. Must be
    called from inside `shard_map` over a mesh with a `'tp'` axis (a
    correctness requirement, not just a perf detail: `psum` needs a bound
    axis environment, which only exists inside `shard_map`/`pmap`).

    `scale` is itself Megatron-sharded (see `vidax.core.sharding
    .shard_wan_params`'s column-parallel handling of this module's params),
    so no extra slicing is needed for it here -- the `params` pytree
    `shard_map` hands this call is already this device's local share.
    """
    dim_local: int
    global_dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim_local,))
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        local_sq_sum = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
        global_sq_sum = jax.lax.psum(local_sq_sum, "tp")
        var = global_sq_sum / self.global_dim
        normed = x * jax.lax.rsqrt(var + self.eps)
        return (normed.astype(orig_dtype)) * scale.astype(orig_dtype)


def _pad_seq(x: jnp.ndarray, axis: int, multiple: int = _FLASH_BLOCK):
    """Zero-pads `x` along `axis` up to the next multiple of `multiple`."""
    size = x.shape[axis]
    pad_len = (-size) % multiple
    if pad_len == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), size


def _legacy_block_sizes_for(sq: int, sk: int):
    """Per-device-kind tile selection for the legacy Pallas flash kernel.
    Returns (BlockSizes | None, pad_multiple) -- None means "use the kernel's
    built-in 128-tile default". Shared by `_flash_attention_tpu` and
    HunyuanVideo's segment-masked path."""
    from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes
    import math

    bs_cfg = _flash_block_sizes()
    if bs_cfg is not None and sq >= bs_cfg[0] and sk >= bs_cfg[1]:
        block_q, block_k_major, block_k = bs_cfg
        return (BlockSizes(block_q=block_q, block_k_major=block_k_major,
                           block_k=block_k, block_b=1),
                math.lcm(_FLASH_BLOCK, block_k_major))
    if (jax.devices()[0].device_kind == "TPU7x" and sq >= 2048
            and _FLASH_BLOCK <= sk < 1024):
        # Short-KV cross-attention on v7: the legacy kernel requires
        # block_k_major == block_k == padded KV length (it loads all of KV per
        # q-block), so give it a big Q tile + a whole-KV K block -- the
        # 128-tile default is latency-bound on the long Q side (measured
        # 14.7ms -> ~3ms at Sq=32760/Skv=512).
        skv_pad = ((sk + _FLASH_BLOCK - 1) // _FLASH_BLOCK) * _FLASH_BLOCK
        return (BlockSizes(block_q=2048, block_k_major=skv_pad, block_k=skv_pad,
                           block_b=1),
                _FLASH_BLOCK)
    return None, _FLASH_BLOCK


def _flash_attention_tpu(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    bias: Optional[jnp.ndarray], scale: float,
) -> jnp.ndarray:
    """TPU Pallas flash attention: O(S) memory instead of materializing the
    full (B, num_heads, S_q, S_k) attention matrix. This is the real fix for
    DiT self-attention over tens of thousands of video patches, where the
    naive materialized matrix alone can exceed a chip's HBM.

    The kernel requires sequence lengths to be multiples of 128; since video
    patch counts (and text lengths, in general) aren't, sequences are
    zero-padded and the padding is excluded from attention via segment ids
    (not just an additive bias) so the kernel can skip whole padded blocks.
    """
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention, SegmentIds)

    b, sq, h, d = q.shape
    sk = k.shape[1]

    # Kernel dispatch on v7 (bf16, no-bias): the vendored MaxDiffusion splash
    # kernel is fastest (see module notes); jax-splash next; tuned legacy
    # flash otherwise / as fallback. head_dim must be a 128-multiple for the
    # splash kernels -- head_dim 64 models (CogVideoX, LTX-Video) are *padded*
    # to 128 with zeros (exact: zero K dims add 0 to every logit, zero V dims
    # are sliced off the output): even paying 2x padding FLOPs this is ~2x
    # faster than the best hd64-native tile config (60ms vs 121ms at
    # S=25916/H=32), because no hd64-native tile shape fills the MXU well.
    if bias is None and q.dtype == jnp.bfloat16:
        pref = _v7_kernel_pref()
        is_v7 = jax.devices()[0].device_kind == "TPU7x"
        md_blocks = None
        if is_v7 and pref in ("auto", "maxdiff") and _md_importable():
            if sq >= 2048 and sk >= 2048:
                md_blocks = _MD_SELF_ATTN_BLOCKS
            elif sq >= 2048 and sk >= 512:
                md_blocks = _MD_CROSS_ATTN_BLOCKS
        splash_cfg = None
        if md_blocks is None and pref in ("auto", "splash") and _splash_importable():
            splash_cfg = _splash_block_sizes_for(sq, sk)
        if md_blocks is not None or splash_cfg is not None:
            pad_hd = 0
            if is_v7 and d % 128 != 0 and d < 128:
                pad_hd = 128 - d
                pad = ((0, 0), (0, 0), (0, 0), (0, pad_hd))
                q, k, v = jnp.pad(q, pad), jnp.pad(k, pad), jnp.pad(v, pad)
            if md_blocks is not None:
                out = _md_splash_attention_tpu(q, k, v, scale, *md_blocks)
            else:
                out = _splash_attention_tpu(q, k, v, scale, *splash_cfg)
            return out[..., :out.shape[-1] - pad_hd] if pad_hd else out

    # Pick tile sizes for this device kind (see `_flash_block_sizes`). The
    # kernel requires block_k_major/block_k to divide the (padded) KV length
    # and every block to fit its dim, so pad to a block-compatible multiple
    # and fall back to the kernel default when the sequence is too short
    # (e.g. single-token refiners).
    block_sizes, pad_multiple = _legacy_block_sizes_for(sq, sk)

    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))

    qt, sq0 = _pad_seq(qt, axis=2, multiple=pad_multiple)
    kt, sk0 = _pad_seq(kt, axis=2, multiple=pad_multiple)
    vt, _ = _pad_seq(vt, axis=2, multiple=pad_multiple)
    # If padding wasn't enough to make the KV length block-divisible, fall
    # back to the kernel default tiles rather than erroring.
    if block_sizes is not None and kt.shape[2] % block_sizes.block_k_major != 0:
        block_sizes = None

    segment_ids = None
    if qt.shape[2] != sq0 or kt.shape[2] != sk0:
        q_ids = jnp.where(jnp.arange(qt.shape[2]) < sq0, 1, 0)[None, :]
        kv_ids = jnp.where(jnp.arange(kt.shape[2]) < sk0, 1, 0)[None, :]
        segment_ids = SegmentIds(
            q=jnp.broadcast_to(q_ids, (b, qt.shape[2])),
            kv=jnp.broadcast_to(kv_ids, (b, kt.shape[2])))

    ab = None
    if bias is not None:
        ab = jnp.broadcast_to(bias, (b, h, sq, sk)).astype(jnp.float32)
        ab = jnp.pad(ab, ((0, 0), (0, 0), (0, qt.shape[2] - sq), (0, kt.shape[2] - sk)))

    out = flash_attention(qt, kt, vt, ab=ab, segment_ids=segment_ids, sm_scale=scale,
                          block_sizes=block_sizes)
    out = out[:, :, :sq0, :]
    return jnp.transpose(out, (0, 2, 1, 3))


# (B, S, num_heads, head_dim) sharding matching vidax.core.sharding's TP
# scheme: batch on 'dp', heads on 'tp', everything else replicated.
_QKV_SPEC = P('dp', None, 'tp', None)


def _flash_attention_tpu_sharded(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, scale: float, mesh: Mesh,
) -> jnp.ndarray:
    """Runs `_flash_attention_tpu` under `shard_map`.

    Pallas/Mosaic TPU kernels are opaque custom calls that GSPMD cannot
    auto-partition ("Mosaic kernels cannot be automatically partitioned" is
    a hard error, for *any* sharded axis -- batch included, not just tensor-
    parallel ones) -- so whenever q/k/v are sharded across more than one
    device, the flash-attention call must be explicitly wrapped in
    `shard_map`, giving each device the kernel call over its own local
    (batch, heads) slice with no cross-device communication needed (exactly
    matching the column/row-parallel attention scheme: each device already
    owns a disjoint, complete subset of attention heads).
    """
    from jax import shard_map

    def _local(q, k, v):
        return _flash_attention_tpu(q, k, v, None, scale)

    return shard_map(
        _local, mesh=mesh, in_specs=(_QKV_SPEC, _QKV_SPEC, _QKV_SPEC),
        out_specs=_QKV_SPEC, check_vma=False)(q, k, v)


def local_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, scale: Optional[float] = None,
) -> jnp.ndarray:
    """Plain dot-product attention that always runs as a single, local
    (non-cross-device) call -- for use from *within* an already per-device
    context (e.g. inside `shard_map`, alongside
    `sequence_parallel_self_attention`'s calls for cross-attention against a
    small, fully-replicated context, where no cross-device communication is
    needed for that op specifically). `dot_product_attention`'s own
    heuristics can't tell they're already inside a sharded body -- they'd
    see `jax.device_count() > 1` with no `mesh` given and fall back to the
    slow XLA-materializing path, so this bypasses that dispatch entirely.
    """
    head_dim = q.shape[-1]
    sm_scale = head_dim ** -0.5 if scale is None else scale
    if jax.devices()[0].platform == "tpu":
        return _flash_attention_tpu(q, k, v, None, sm_scale)
    return jax.nn.dot_product_attention(q, k, v, scale=sm_scale)


def sequence_parallel_self_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    sp_axis_name: str, scale: Optional[float] = None,
) -> jnp.ndarray:
    """DeepSpeed-Ulysses sequence-parallel self-attention (arxiv.org/abs/2309.14509),
    matching Wan2.2-main/wan/distributed/ulysses.py's `distributed_attention`.

    Must be called from *within* an active `shard_map` over a mesh with an
    axis named `sp_axis_name` (bound by the caller -- typically the entire
    DiT forward pass runs inside one `shard_map`, not just this call; see
    `vidax.models.wan.wan2_2.dit`'s module docstring for why and how the
    surrounding chunk-before/gather-after logic fits together).

    Where Megatron-style tensor parallelism (`_flash_attention_tpu_sharded`)
    shards attention *heads* and keeps the full token sequence on every
    device, this shards the *sequence* between blocks (cutting the large
    per-token activations -- Wan2.2's per-token AdaLN modulation tensors in
    particular -- by `sp_axis_name`'s size) and only reshuffles to a
    head-sharded view of the *full* sequence for the duration of self-
    attention itself, via two `all_to_all`s: each device already holds every
    head for its local sequence chunk (having just computed q/k/v locally);
    the first all_to_all redistributes that into every device holding every
    sequence position for its local head chunk (a pure data reshuffle, no
    device recomputes another's tokens), local (non-distributed) flash
    attention runs on that, and the second all_to_all reshuffles back.

    Args:
        q, k, v: Shape (B, L_local, num_heads, head_dim) -- this device's
            local sequence chunk, full heads.
        sp_axis_name: Name of the mesh axis to reshuffle across.
        scale: Optional override for the softmax scale (default 1/sqrt(head_dim)).

    Returns:
        (B, L_local, num_heads, head_dim), same shape as the inputs.
    """
    head_dim = q.shape[-1]
    sm_scale = head_dim ** -0.5 if scale is None else scale

    q = jax.lax.all_to_all(q, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)
    k = jax.lax.all_to_all(k, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)
    v = jax.lax.all_to_all(v, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)

    if jax.devices()[0].platform == "tpu":
        out = _flash_attention_tpu(q, k, v, None, sm_scale)
    else:
        out = jax.nn.dot_product_attention(q, k, v, scale=sm_scale)

    return jax.lax.all_to_all(out, sp_axis_name, split_axis=1, concat_axis=2, tiled=True)


def sequence_parallel_joint_self_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    text_len: int, sp_axis_name: str, scale: Optional[float] = None,
) -> jnp.ndarray:
    """DeepSpeed-Ulysses sequence-parallel self-attention over a *joint*
    ``[text(text_len); visual]`` token sequence in which only the visual
    tokens are sequence-parallel-chunked across ``sp_axis_name`` and the
    ``text_len`` text-prefix tokens are fully replicated on every device.

    This is CogVideoX's attention layout (`vidax.models.cogvideo.dit
    .CogVideoXAttention` -- one joint self-attention per block, no separate
    cross-attention). `sequence_parallel_self_attention` above can't be used
    directly: naively all-to-all-ing the concatenated ``[text; visual_chunk]``
    would replicate the text tokens ``sp_size`` times in the reshuffled KV.
    Instead the text and visual q/k/v are split apart, only the visual part
    goes through the Ulysses head<->sequence all-to-all, the (small,
    replicated) text q/k/v is sliced down to this device's local head range
    to match, joint local flash attention runs over
    ``[text(full); visual(full)]`` for that head range, and the two outputs
    are reshuffled back independently (visual via the reverse all-to-all,
    text via an ``all_gather`` over the head axis).

    Must be called from *within* an active ``shard_map`` over a mesh with an
    axis named ``sp_axis_name`` (same requirement as
    ``sequence_parallel_self_attention``).

    Args:
        q, k, v: Shape (B, text_len + L_visual_local, num_heads, head_dim) --
            the text prefix at full length, the visual tokens this device's
            local sequence chunk; full heads.
        text_len: Number of text-prefix tokens (static). ``num_heads`` must be
            divisible by ``sp_axis_name``'s size.
        sp_axis_name: Name of the mesh axis to reshuffle the visual tokens across.
        scale: Optional softmax-scale override (default 1/sqrt(head_dim)).

    Returns:
        (B, text_len + L_visual_local, num_heads, head_dim), same layout as
        the inputs (text prefix replicated, visual tokens local chunk).
    """
    head_dim = q.shape[-1]
    sm_scale = head_dim ** -0.5 if scale is None else scale

    q_txt, q_vis = q[:, :text_len], q[:, text_len:]
    k_txt, k_vis = k[:, :text_len], k[:, text_len:]
    v_txt, v_vis = v[:, :text_len], v[:, text_len:]

    # visual: full heads / local sequence chunk -> local head chunk / full sequence.
    q_vis = jax.lax.all_to_all(q_vis, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)
    k_vis = jax.lax.all_to_all(k_vis, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)
    v_vis = jax.lax.all_to_all(v_vis, sp_axis_name, split_axis=2, concat_axis=1, tiled=True)

    # text: keep exactly the head range the visual all-to-all just handed this
    # device (heads [rank*hl : (rank+1)*hl]).
    heads_local = q_vis.shape[2]
    rank = jax.lax.axis_index(sp_axis_name)
    q_txt = jax.lax.dynamic_slice_in_dim(q_txt, rank * heads_local, heads_local, axis=2)
    k_txt = jax.lax.dynamic_slice_in_dim(k_txt, rank * heads_local, heads_local, axis=2)
    v_txt = jax.lax.dynamic_slice_in_dim(v_txt, rank * heads_local, heads_local, axis=2)

    q_j = jnp.concatenate([q_txt, q_vis], axis=1)
    k_j = jnp.concatenate([k_txt, k_vis], axis=1)
    v_j = jnp.concatenate([v_txt, v_vis], axis=1)

    if jax.devices()[0].platform == "tpu":
        out_j = _flash_attention_tpu(q_j, k_j, v_j, None, sm_scale)
    else:
        out_j = jax.nn.dot_product_attention(q_j, k_j, v_j, scale=sm_scale)

    out_txt, out_vis = out_j[:, :text_len], out_j[:, text_len:]
    # visual: local head chunk / full sequence -> full heads / local sequence chunk.
    out_vis = jax.lax.all_to_all(out_vis, sp_axis_name, split_axis=1, concat_axis=2, tiled=True)
    # text: reassemble full heads (identical on every device afterwards).
    out_txt = jax.lax.all_gather(out_txt, sp_axis_name, axis=2, tiled=True)
    return jnp.concatenate([out_txt, out_vis], axis=1)


def dot_product_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
    mesh: Optional[Mesh] = None,
) -> jnp.ndarray:
    """Full (non-causal) dot-product attention.

    On TPU, dispatches to a real (O(S) memory) Pallas flash-attention kernel
    rather than `jax.nn.dot_product_attention`'s default "xla" path, which
    fully materializes the (B, num_heads, S_q, S_k) attention matrix -- for
    DiT self-attention over tens of thousands of video patches, that
    materialized matrix alone can exceed a chip's HBM. Falls back to
    `jax.nn.dot_product_attention` elsewhere (CPU/GPU), whenever a boolean
    `mask` is given (the flash kernel has no boolean-mask input), or when
    both `bias` and multi-device sharding are given at once (the sharded
    flash path doesn't thread `bias` through yet). An additive `bias` alone
    on a single device *does* still take the flash path -- needed for
    Cosmos3's dual-pathway generation attention, which cross-attends over a
    padded text segment and must mask out the padding via `bias`, at a
    sequence length large enough (tens of thousands of video patches) that
    the materializing fallback isn't viable.

    This is the Megatron-style (head-sharded, full-sequence-per-device)
    attention path; see `sequence_parallel_self_attention` for the
    alternative (sequence-sharded) scheme Wan2.2's DiT uses instead, which
    this function has no part in -- that path calls flash attention directly.

    Args:
        q, k, v: Shape (B, S, num_heads, head_dim). k/v may have a different
            sequence length than q (cross-attention).
        bias: Optional additive attention bias (e.g. T5's relative position
            bias), broadcastable to (B, num_heads, S_q, S_k).
        mask: Optional boolean mask, broadcastable to (B, num_heads, S_q, S_k).
        scale: Optional override for the softmax scale (default 1/sqrt(head_dim)).
            T5 attention uses scale=1.0 (no scaling).
        mesh: The device mesh q/k/v are sharded over, if any (see
            `vidax.core.sharding`). Required on TPU whenever running across
            more than one device -- Mosaic kernels can't infer this on their
            own the way ordinary XLA ops can.

    Returns:
        Attention output, shape (B, S_q, num_heads, head_dim).
    """
    multi_device = jax.device_count() > 1
    # The flash kernel itself takes an additive `bias` (see `_flash_attention_tpu`'s
    # `ab` argument) -- only a boolean `mask` forces the slower path, since the
    # kernel has no boolean-mask input of its own. Only the single-device flash
    # path is extended to carry `bias` through for now (`_flash_attention_tpu_sharded`
    # still forces `bias=None`); multi-device + bias falls back to the correct,
    # if slower, XLA path below rather than silently dropping the bias.
    can_use_flash = (
        mask is None and jax.devices()[0].platform == "tpu"
        and (not multi_device or (mesh is not None and bias is None)))
    if can_use_flash:
        head_dim = q.shape[-1]
        sm_scale = head_dim ** -0.5 if scale is None else scale
        if multi_device:
            return _flash_attention_tpu_sharded(q, k, v, sm_scale, mesh)
        return _flash_attention_tpu(q, k, v, bias, sm_scale)
    # Multi-device with no mesh given (e.g. WanVAEDecoder, which isn't
    # tensor-parallel sharded): Mosaic kernels can't run un-sharded across
    # multiple devices at all ("cannot be automatically partitioned" is a
    # hard error even for a trivially-replicated array), so without a mesh
    # to shard_map over, fall back to the slower materializing path.
    return jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask, scale=scale)
