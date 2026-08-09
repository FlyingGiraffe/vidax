"""Attention building block shared by Cosmos DiT versions (2.5 now, 3 later).

Differs from `vidax.models.wan.common.dit_layers.attend` in exactly the ways
Cosmos's `Attention` class (cosmos-predict2.5-main/cosmos_predict2/_src/
predict2/networks/minimal_v4_dit.py) differs from Wan's `WanSelfAttention`:
QK-RMSNorm is applied *per attention head* (over `head_dim`), not over the
full projected `dim`, and RoPE (`create_cosmos_rope3d_freqs` /
`apply_cosmos_rope3d`) uses the rotate-half convention, not Wan's interleaved
pairs -- see `vidax.models.cosmos2_5.rope`'s module docstring. All
projections are bias-free (`nn.Linear(..., bias=False)` throughout the
reference), unlike Wan's biased Dense layers.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from vidax.core.attention import (
    RMSNorm, dot_product_attention, local_attention, sequence_parallel_self_attention,
)
from vidax.models.cosmos2_5.rope import apply_cosmos_rope3d


def cosmos_attend(
    x_q: jnp.ndarray,
    x_kv: jnp.ndarray,
    dim: int,
    num_heads: int,
    head_dim: int,
    eps: float,
    prefix: str,
    rope_freqs: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    mesh: Optional[Mesh] = None,
    sequence_parallel: bool = False,
    sp_axis_name: str = "sp",
) -> jnp.ndarray:
    """Shared QKV-projection + per-head RMSNorm + (optional RoPE) + attention
    + output-projection path, matching Cosmos's `Attention` class. Used for
    both self-attention (`x_kv is x_q`, `rope_freqs` given) and cross-
    attention (`x_kv` is the projected text context, `rope_freqs=None` --
    the reference never applies RoPE to cross-attention, since the text
    tokens' positional signal is already encoded by the text model itself).

    ``sequence_parallel`` (DeepSpeed-Ulysses, see
    ``vidax.models.cosmos2_5.dit``'s module docstring): must be
    called from *within* an active ``shard_map`` over a mesh with an axis
    named ``sp_axis_name``. Self-attention (``rope_freqs`` given) dispatches
    to ``sequence_parallel_self_attention``'s all-to-all reshuffle;
    cross-attention runs as a plain *local* call instead (``x_kv`` is the
    small, already fully-replicated text context, so no cross-device
    reshuffle is needed there) -- exactly mirroring
    ``vidax.models.wan.common.dit_layers.attend``'s own split between the
    two, since the reasoning (and the underlying primitives) are identical.

    Composes with Megatron weight-sharding on the mesh's independent ``'tp'``
    axis (see ``vidax.core.sharding.build_tpu_mesh``): whenever
    ``sequence_parallel`` is set, `q`/`k`/`v`'s Dense output width and head
    count are divided by ``mesh.shape['tp']`` (a no-op when that's 1) --
    see ``attend``'s identical comment for why (GSPMD reconciles this
    automatically outside `shard_map`, but nothing does inside it). Unlike
    Wan's Q/K-RMSNorm, Cosmos's is per-*head* (reduces over `head_dim`
    only, which Megatron sharding never splits -- whole heads always stay
    on one device), so it needs no distributed-reduction equivalent of
    ``vidax.core.attention.TPShardedRMSNorm``.
    """
    b = x_q.shape[0]
    inner_dim = num_heads * head_dim
    tp_size = mesh.shape["tp"] if (sequence_parallel and mesh is not None) else 1
    inner_dim_local = inner_dim // tp_size
    num_heads_local = num_heads // tp_size

    q = nn.Dense(inner_dim_local, use_bias=False, name=f"{prefix}_q_proj")(x_q)
    k = nn.Dense(inner_dim_local, use_bias=False, name=f"{prefix}_k_proj")(x_kv)
    v = nn.Dense(inner_dim_local, use_bias=False, name=f"{prefix}_v_proj")(x_kv)

    q = q.reshape(b, -1, num_heads_local, head_dim)
    k = k.reshape(b, -1, num_heads_local, head_dim)
    v = v.reshape(b, -1, num_heads_local, head_dim)

    # Per-head RMSNorm (over the last axis, `head_dim`) -- unlike Wan, which
    # normalizes over the full `dim` before splitting into heads.
    q = RMSNorm(head_dim, eps=eps, name=f"{prefix}_q_norm")(q)
    k = RMSNorm(head_dim, eps=eps, name=f"{prefix}_k_norm")(k)

    if rope_freqs is not None:
        q = apply_cosmos_rope3d(q, rope_freqs)
        k = apply_cosmos_rope3d(k, rope_freqs)

    if sequence_parallel:
        if rope_freqs is not None:
            out = sequence_parallel_self_attention(q, k, v, sp_axis_name)
        else:
            out = local_attention(q, k, v)
    else:
        out = dot_product_attention(q, k, v, mesh=mesh)
    out = out.reshape(b, -1, inner_dim_local)
    out = nn.Dense(dim, use_bias=False, name=f"{prefix}_output_proj")(out)
    # Row-parallel: see `vidax.models.wan.common.dit_layers.attend`'s
    # identical comment for why this manual reduce is needed only inside
    # `sequence_parallel` (running under `shard_map`), and why it's a safe
    # no-op when 'tp' has size 1.
    if sequence_parallel:
        out = jax.lax.psum(out, "tp")
    return out
