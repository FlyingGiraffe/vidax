"""TPU device mesh construction and Megatron-style 1D tensor-parallel sharding.

Sharding strategy for Wan's DiT/T5 encoder and Cosmos-Predict2.5's DiT: the residual/hidden
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
import os

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

DEFAULT_JAX_CACHE_DIR = os.path.expanduser("~/.cache/vidax/jax")

# Dense-layer names (the parent key one level above "kernel"/"bias" in a
# flax param tree) that are column-parallel (shard the output/last axis) vs.
# row-parallel (shard the input/first axis). Covers WanDiT's, T5Encoder's,
# CosmosDiT's, and Reason1's (`vidax.models.cosmos2_5.reason1`)
# naming conventions. Cosmos's per-head QK-RMSNorm (`self_attn_q_norm`/
# `_k_norm`, shape `(head_dim,)`, shared identically across every head)
# needs no entry here at all: unlike Wan's QK-RMSNorm (over the *full*,
# TP-split `dim`), it's already local to whichever heads a device owns, so
# it falls through to the default (replicated) case correctly with no
# special-casing.
#
# Reason1's bare `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/
# `down_proj` names (standard HF Qwen2/Llama-family submodule names, not
# namespaced with a `self_attn_`/`mlp_` prefix the way Cosmos's own DiT
# names are) only need weight-sharding here -- unlike Cosmos's attention,
# Reason1's always passes an explicit causal `mask` to
# `vidax.core.attention.dot_product_attention`, which for that reason
# *never* takes the Pallas-flash-attention/mesh-sharded path regardless
# (see that function's docstring) and always falls back to plain
# `jax.nn.dot_product_attention`, which GSPMD auto-partitions correctly
# given TP-sharded weights with no `shard_map`/mesh-threading needed in
# `reason1.py` itself -- exactly the same situation T5's own
# (bias-carrying, so also always-fallback-path) attention already relies
# on. These generic names are a real (if currently harmless) collision
# risk for any future model reusing them with different sharding needs --
# noted, not fixed, since nothing in this repo does yet.
COLUMN_PARALLEL_NAMES = frozenset([
    "self_attn_q", "self_attn_k", "self_attn_v",
    "cross_attn_q", "cross_attn_k", "cross_attn_v",
    "self_attn_norm_q", "self_attn_norm_k",     # WanDiT Q/K-RMSNorm's `scale`,
    "cross_attn_norm_q", "cross_attn_norm_k",   # column-split to match Q/K's own
                                                 # split -- only matters combined
                                                 # with `sequence_parallel` (see
                                                 # `vidax.core.attention
                                                 # .TPShardedRMSNorm`); harmless
                                                 # under plain Megatron TP, where
                                                 # GSPMD reconciles it as usual.
    "ffn_0",                        # WanDiT FFN up-projection
    "attn_q", "attn_k", "attn_v",   # T5 self-attention
    "ffn_gate_0", "ffn_fc1",        # T5 FFN up-projection
    "self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj",      # CosmosDiT
    "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj",   # CosmosDiT
    "mlp_layer1",                   # CosmosDiT FFN up-projection
    "q_proj", "k_proj", "v_proj",   # Reason1 self-attention
    "gate_proj", "up_proj",         # Reason1 FFN up-projection (SwiGLU); also Cosmos3's MLP (same names)
    "to_q", "to_k", "to_v",                    # Cosmos3 "und" (text/causal) self-attention
    "add_q_proj", "add_k_proj", "add_v_proj",  # Cosmos3 "gen" (vision/diffusion) attention
])
ROW_PARALLEL_NAMES = frozenset([
    "self_attn_o", "cross_attn_o",
    "ffn_2",        # WanDiT FFN down-projection
    "attn_o",       # T5 attention output
    "ffn_fc2",      # T5 FFN down-projection
    "self_attn_output_proj", "cross_attn_output_proj",  # CosmosDiT
    "mlp_layer2",   # CosmosDiT FFN down-projection
    "o_proj",       # Reason1 attention output
    "down_proj",    # Reason1 FFN down-projection; also Cosmos3's MLP (same name)
    "to_out",       # Cosmos3 "und" attention output
    "to_add_out",   # Cosmos3 "gen" attention output
])


def build_tpu_mesh(
    data_parallel_size: int, tensor_parallel_size: int, sequence_parallel_size: int = 1,
) -> Mesh:
    """
    Creates a JAX Mesh for TPU v4/v5e/v6e.

    Args:
        data_parallel_size: Number of devices for data parallelism.
        tensor_parallel_size: Number of devices for Megatron-style weight
            sharding ('tp' axis).
        sequence_parallel_size: Number of devices for DeepSpeed-Ulysses
            token-sequence sharding ('sp' axis, independent of 'tp' -- see
            `docs/hardware_and_sharding.md`'s "Combining both" section for
            how the two compose). Defaults to 1 (today's 2-axis behavior --
            a size-1 axis is invisible to every `PartitionSpec` that doesn't
            name it, so existing callers that never reference 'sp' are
            unaffected).
    """
    devices = jax.devices()
    total_devices = len(devices)
    expected = data_parallel_size * tensor_parallel_size * sequence_parallel_size
    assert total_devices == expected, (
        f"Mesh mismatch: {total_devices} devices vs "
        f"{data_parallel_size}x{tensor_parallel_size}x{sequence_parallel_size}")

    device_mesh = np.array(devices).reshape(
        data_parallel_size, tensor_parallel_size, sequence_parallel_size)
    return Mesh(device_mesh, axis_names=('dp', 'tp', 'sp'))


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
    """Assigns a tensor-parallel `NamedSharding` to every leaf of a WanDiT,
    T5Encoder, or CosmosDiT parameter pytree (see module docstring for the
    layout) -- name kept for backward compatibility even though it now
    covers more than Wan; the dispatch is entirely name-pattern-driven
    (`COLUMN_PARALLEL_NAMES`/`ROW_PARALLEL_NAMES` above), not tied to any
    one architecture.

    Returns a pytree of `NamedSharding` with the same structure as `params`,
    suitable for `jax.device_put(params, shardings)`.
    """
    def spec_for_leaf(path, leaf):
        parent = path[-2].key if len(path) >= 2 and hasattr(path[-2], "key") else None
        leaf_name = path[-1].key if hasattr(path[-1], "key") else None

        if parent in COLUMN_PARALLEL_NAMES and leaf.ndim >= 1:
            if leaf_name == "kernel" and leaf.ndim == 2:
                return NamedSharding(mesh, P(None, 'tp'))
            elif leaf_name in ("bias", "scale") and leaf.ndim == 1:
                return NamedSharding(mesh, P('tp'))
        elif parent in ROW_PARALLEL_NAMES and leaf.ndim >= 1:
            if leaf_name == "kernel" and leaf.ndim == 2:
                return NamedSharding(mesh, P('tp', None))
            # Row-parallel biases are added after GSPMD's implicit
            # all-reduce, so they stay replicated (fall through below).

        return NamedSharding(mesh, P(*([None] * leaf.ndim)))

    return jax.tree_util.tree_map_with_path(spec_for_leaf, params)


def to_partition_specs(shardings):
    """Strips each leaf `NamedSharding` down to its bare `PartitionSpec`.

    `jax.device_put(params, shard_wan_params(params, mesh))` wants the
    `NamedSharding` pytree `shard_wan_params` already returns; `shard_map`'s
    `in_specs` wants the same per-leaf partitioning as plain `PartitionSpec`s
    instead. This converts one to the other, so a `sequence_parallel`-capable
    script's weight sharding can be identical -- both real (weight-sharded)
    and passed correctly into `shard_map` -- instead of falling back to a
    blanket "fully replicated" `P()` for the whole params pytree just
    because the call happens to run inside `shard_map`.
    """
    return jax.tree_util.tree_map(lambda s: s.spec, shardings)


def configure_jax_cache(cache_dir: str = DEFAULT_JAX_CACHE_DIR) -> None:
    """Enables JAX's persistent compilation cache, keyed on disk by program
    hash so a second run at the same (shape, dtype, sharding, mesh, step
    count) signature skips XLA compilation entirely instead of re-paying it.

    Compile time for these model sizes is tens of seconds to minutes (see
    `docs/benchmarking.md`'s Compile-time column) -- without this, every one
    of this repo's example-script invocations pays that cost fresh, even
    running the exact same command twice in a row. Call once, before
    building any mesh/model, from every example script's `main()`.
    """
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # JAX's own recommended pairing for this cache: cache every compiled
    # program regardless of how small (-1 disables the size floor) and only
    # bother once a compile actually took a little while (skips caching
    # trivial/instant compiles, which would otherwise just add disk-read
    # overhead to their next run for no real savings).
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
