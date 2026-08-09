"""Building blocks shared by every Wan DiT version (Wan2.1, Wan2.2, ...).

``attend`` and ``WanHead`` port the reference's ``WanSelfAttention`` /
``WanT2VCrossAttention`` / ``Head`` classes, which are structurally identical
between Wan2.1-main/wan/modules/model.py and Wan2.2-main/wan/modules/model.py
(Wan2.2 simply never passes ``image_context``, since it has no CLIP-based
image cross-attention branch -- see ``vidax.models.wan.wan2_1.dit`` for that
i2v-only extension).
"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import (
    RMSNorm, TPShardedRMSNorm, chunk_by_rank, dot_product_attention, local_attention,
    sequence_parallel_self_attention,
)
from vidax.core.rope3d import apply_rope3d

# Re-exported for backward compatibility -- every caller in this repo
# imports `chunk_by_rank` from here (or from `vidax.models.cosmos2_5
# .dit`, which imports the same underlying function directly from
# `vidax.core.attention`, its natural model-family-agnostic home).
__all__ = ["chunk_by_rank", "attend", "WanHead"]


def attend(
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
    sequence_parallel: bool = False,
    sp_axis_name: str = "sp",
) -> jnp.ndarray:
    """Shared QKV-projection + RMSNorm + attention + output-projection path.

    Used for both self-attention (x_kv is x, rope_freqs set) and
    cross-attention (x_kv is the text context, rope_freqs None), matching
    ``WanSelfAttention``/``WanT2VCrossAttention`` in the reference, which
    share this exact structure.

    ``image_context``, when given (Wan2.1 i2v cross-attention only), adds a
    second K/V projection over CLIP image features and sums its attention
    output with the text one before the shared output projection -- matching
    ``WanI2VCrossAttention``, which reuses the same query for both.

    ``sequence_parallel``, when set, must be called from *within* an active
    ``shard_map`` over a mesh with an axis named ``sp_axis_name``: self-
    attention (``rope_freqs`` given) dispatches to
    ``sequence_parallel_self_attention``'s all-to-all reshuffle; cross-
    attention runs as an ordinary *local* call instead (``x_kv`` is the small,
    already fully-replicated text context, so no cross-device reshuffle is
    needed there -- just a plain per-device flash-attention call, since
    ``dot_product_attention``'s own dispatch heuristics can't tell they're
    already running inside a sharded body).

    Composes with Megatron weight-sharding on the mesh's independent ``'tp'``
    axis (see ``vidax.core.sharding.build_tpu_mesh``): whenever
    ``sequence_parallel`` is set, `q`/`k`/`v`'s Dense output width and head
    count are divided by ``mesh.shape['tp']`` (a no-op when that's 1) to
    match what's actually local to this device under `shard_map` -- unlike
    plain Megatron TP (no `shard_map`), where GSPMD's auto-partitioner
    reconciles a Dense layer's *declared* (global) output width against its
    physically-sharded weight transparently, `shard_map` hands every op the
    true local array, so the declared width must already say so. Q/K-RMSNorm
    needs an extra fix beyond the shape change: it reduces over the *entire*
    projected `dim`, which is now split across devices, so a per-device
    local mean-square would be a different (wrong) number from the
    reference's -- ``TPShardedRMSNorm`` sums each device's local
    sum-of-squares via ``psum`` first (see its own docstring). ``image_context``
    (Wan2.1 i2v's CLIP cross-attention) isn't threaded through this yet --
    combining it with TP weight-sharding under `sequence_parallel` raises
    below rather than silently producing wrong output.
    """
    head_dim = dim // num_heads
    b = x_q.shape[0]

    tp_size = mesh.shape["tp"] if (sequence_parallel and mesh is not None) else 1
    if image_context is not None and tp_size > 1:
        raise NotImplementedError(
            "Combining --tensor_parallel_size > 1 with --sequence_parallel_size > 1 "
            "isn't supported for Wan2.1's i2v CLIP cross-attention (image_context) yet "
            "-- k_img/v_img aren't Megatron-sharded, so they'd end up with a different "
            "head count than the (now TP-sharded) q. Use --tensor_parallel_size 1 with "
            "--sequence_parallel_size, or --sequence_parallel_size 1 with "
            "--tensor_parallel_size, for i2v.")
    dim_local = dim // tp_size
    num_heads_local = num_heads // tp_size

    q = nn.Dense(dim_local, name=f"{prefix}_q")(x_q)
    k = nn.Dense(dim_local, name=f"{prefix}_k")(x_kv)
    v = nn.Dense(dim_local, name=f"{prefix}_v")(x_kv)

    if qk_norm:
        if tp_size > 1:
            q = TPShardedRMSNorm(dim_local, dim, eps=eps, name=f"{prefix}_norm_q")(q)
            k = TPShardedRMSNorm(dim_local, dim, eps=eps, name=f"{prefix}_norm_k")(k)
        else:
            q = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_q")(q)
            k = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_k")(k)

    q = q.reshape(b, -1, num_heads_local, head_dim)
    k = k.reshape(b, -1, num_heads_local, head_dim)
    v = v.reshape(b, -1, num_heads_local, head_dim)

    if rope_freqs is not None:
        q = apply_rope3d(q, rope_freqs)
        k = apply_rope3d(k, rope_freqs)

    if sequence_parallel:
        if rope_freqs is not None:
            out = sequence_parallel_self_attention(q, k, v, sp_axis_name)
        else:
            out = local_attention(q, k, v)
    else:
        out = dot_product_attention(q, k, v, mesh=mesh)
    out = out.reshape(b, -1, dim_local)

    if image_context is not None:
        # Only reached when tp_size == 1 (guarded above), so dim_local ==
        # dim / num_heads_local == num_heads here.
        k_img = nn.Dense(dim, name=f"{prefix}_k_img")(image_context)
        v_img = nn.Dense(dim, name=f"{prefix}_v_img")(image_context)
        if qk_norm:
            k_img = RMSNorm(dim, eps=eps, name=f"{prefix}_norm_k_img")(k_img)
        k_img = k_img.reshape(b, -1, num_heads, head_dim)
        v_img = v_img.reshape(b, -1, num_heads, head_dim)
        # Same reasoning as the text cross-attention branch above: CLIP
        # image features are small and already fully replicated, so under
        # sequence_parallel this is a local call too, not the mesh-based
        # dispatch (which can't tell it's already running inside a sharded
        # body and would pick the wrong path).
        if sequence_parallel:
            img_out = local_attention(q, k_img, v_img).reshape(b, -1, dim)
        else:
            img_out = dot_product_attention(q, k_img, v_img, mesh=mesh).reshape(b, -1, dim)
        out = out + img_out

    out = nn.Dense(dim, name=f"{prefix}_o")(out)
    # `{prefix}_o` is row-parallel (its input, `out`, is already TP-split
    # across attention heads): outside `sequence_parallel`, GSPMD's ordinary
    # auto-partitioner inserts the cross-device sum this needs automatically;
    # inside it (this whole call running under `shard_map`), nothing does
    # that for us, so it must be requested by hand. A no-op when the 'tp'
    # axis has size 1 (weights not actually split), so this is safe to
    # always include whenever `sequence_parallel` is on, regardless of
    # whether TP weight-sharding is also in use.
    if sequence_parallel:
        out = jax.lax.psum(out, "tp")
    return out


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
