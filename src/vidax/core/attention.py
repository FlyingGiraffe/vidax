"""Dot-product attention and QK-normalization primitives shared by DiT/T5 blocks."""
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import Mesh, PartitionSpec as P

_FLASH_BLOCK = 128  # Fixed tile size of jax's TPU Pallas flash-attention kernel.


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


def _pad_seq(x: jnp.ndarray, axis: int, multiple: int = _FLASH_BLOCK):
    """Zero-pads `x` along `axis` up to the next multiple of `multiple`."""
    size = x.shape[axis]
    pad_len = (-size) % multiple
    if pad_len == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), size


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
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention, SegmentIds

    b, sq, h, d = q.shape
    sk = k.shape[1]
    qt = jnp.transpose(q, (0, 2, 1, 3))
    kt = jnp.transpose(k, (0, 2, 1, 3))
    vt = jnp.transpose(v, (0, 2, 1, 3))

    qt, sq0 = _pad_seq(qt, axis=2)
    kt, sk0 = _pad_seq(kt, axis=2)
    vt, _ = _pad_seq(vt, axis=2)

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

    out = flash_attention(qt, kt, vt, ab=ab, segment_ids=segment_ids, sm_scale=scale)
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
    from jax.experimental.shard_map import shard_map

    def _local(q, k, v):
        return _flash_attention_tpu(q, k, v, None, scale)

    return shard_map(
        _local, mesh=mesh, in_specs=(_QKV_SPEC, _QKV_SPEC, _QKV_SPEC),
        out_specs=_QKV_SPEC, check_rep=False)(q, k, v)


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
    `jax.nn.dot_product_attention` elsewhere (CPU/GPU), or whenever `bias`
    or `mask` is given (the flash kernel only takes an additive bias, and in
    practice only T5's small, fixed-length self-attention uses one, which
    doesn't need flash attention's memory savings anyway).

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
    if bias is None and mask is None and jax.devices()[0].platform == "tpu" and (
            not multi_device or mesh is not None):
        head_dim = q.shape[-1]
        sm_scale = head_dim ** -0.5 if scale is None else scale
        if multi_device:
            return _flash_attention_tpu_sharded(q, k, v, sm_scale, mesh)
        return _flash_attention_tpu(q, k, v, None, sm_scale)
    # Multi-device with no mesh given (e.g. WanVAEDecoder, which isn't
    # tensor-parallel sharded): Mosaic kernels can't run un-sharded across
    # multiple devices at all ("cannot be automatically partitioned" is a
    # hard error even for a trivially-replicated array), so without a mesh
    # to shard_map over, fall back to the slower materializing path.
    return jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask, scale=scale)
