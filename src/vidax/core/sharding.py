"""TPU device mesh construction and Megatron-style 1D tensor-parallel sharding.

Sharding strategy for Wan2.1's DiT and T5 encoder: the residual/hidden
stream (`dim`) is always kept replicated across the 'tp' axis (needed for
correct RMSNorm/LayerNorm, which reduce over the full feature dimension).
Only the attention QKV projections and FFN up-projection are *column*-
sharded (their output splits cleanly along whole attention heads / FFN
channels), and the attention output projection and FFN down-projection are
*row*-sharded (their input is already split, and JAX's GSPMD auto-inserts
the all-reduce needed to produce a replicated `dim`-wide output). This is
the standard Megatron-LM 1D tensor-parallel layout, and it directly
addresses the real memory bottleneck for video DiTs: self-attention over
tens of thousands of patches needs an O(S^2 * num_heads) attention matrix,
which this scheme divides by `tensor_parallel_size` since each device only
ever holds & computes its local subset of heads.

Everything else (norms, biases of row-parallel layers, embeddings,
modulation, patch/time/text embedding layers) is left replicated: it's
small, and GSPMD correctly and cheaply broadcasts replicated operands
against tensor-parallel-sharded ones in elementwise ops (no communication
needed), so there's no correctness or meaningful memory cost to leaving it be.
"""
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# Dense-layer names (the parent key one level above "kernel"/"bias" in a
# flax param tree) that are column-parallel (shard the output/last axis) vs.
# row-parallel (shard the input/first axis). Covers both WanDiT's and
# T5Encoder's naming conventions.
COLUMN_PARALLEL_NAMES = frozenset([
    "self_attn_q", "self_attn_k", "self_attn_v",
    "cross_attn_q", "cross_attn_k", "cross_attn_v",
    "ffn_0",                        # WanDiT FFN up-projection
    "attn_q", "attn_k", "attn_v",   # T5 self-attention
    "ffn_gate_0", "ffn_fc1",        # T5 FFN up-projection
])
ROW_PARALLEL_NAMES = frozenset([
    "self_attn_o", "cross_attn_o",
    "ffn_2",        # WanDiT FFN down-projection
    "attn_o",       # T5 attention output
    "ffn_fc2",      # T5 FFN down-projection
])


def build_tpu_mesh(data_parallel_size: int, tensor_parallel_size: int) -> Mesh:
    """
    Creates a JAX Mesh for TPU v4/v5e/v6e.

    Args:
        data_parallel_size: Number of devices for data parallelism.
        tensor_parallel_size: Number of devices for tensor parallelism.
    """
    devices = jax.devices()
    total_devices = len(devices)
    assert total_devices == data_parallel_size * tensor_parallel_size, \
        f"Mesh mismatch: {total_devices} devices vs {data_parallel_size}x{tensor_parallel_size}"

    device_mesh = np.array(devices).reshape(data_parallel_size, tensor_parallel_size)
    return Mesh(device_mesh, axis_names=('dp', 'tp'))


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Fully-replicated sharding (every device holds the whole array)."""
    return NamedSharding(mesh, P())


def get_batch_sharding(mesh: Mesh, ndim: int) -> NamedSharding:
    """Shards only the leading (batch) axis across 'dp'; all other axes and
    the 'tp' axis are replicated. Used for latents, token ids, and text
    embeddings, whose batch dimension maps to data-parallel replicas.
    """
    return NamedSharding(mesh, P('dp', *([None] * (ndim - 1))))


def shard_wan_params(params: dict, mesh: Mesh) -> dict:
    """Assigns a tensor-parallel `NamedSharding` to every leaf of a WanDiT or
    T5Encoder parameter pytree (see module docstring for the layout).

    Returns a pytree of `NamedSharding` with the same structure as `params`,
    suitable for `jax.device_put(params, shardings)`.
    """
    def spec_for_leaf(path, leaf):
        parent = path[-2].key if len(path) >= 2 and hasattr(path[-2], "key") else None
        leaf_name = path[-1].key if hasattr(path[-1], "key") else None

        if parent in COLUMN_PARALLEL_NAMES and leaf.ndim >= 1:
            if leaf_name == "kernel" and leaf.ndim == 2:
                return NamedSharding(mesh, P(None, 'tp'))
            elif leaf_name == "bias" and leaf.ndim == 1:
                return NamedSharding(mesh, P('tp'))
        elif parent in ROW_PARALLEL_NAMES and leaf.ndim >= 1:
            if leaf_name == "kernel" and leaf.ndim == 2:
                return NamedSharding(mesh, P('tp', None))
            # Row-parallel biases are added after GSPMD's implicit
            # all-reduce, so they stay replicated (fall through below).

        return NamedSharding(mesh, P(*([None] * leaf.ndim)))

    return jax.tree_util.tree_map_with_path(spec_for_leaf, params)
