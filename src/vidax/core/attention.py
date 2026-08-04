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
