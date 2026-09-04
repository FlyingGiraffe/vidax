"""Model-family-agnostic XLA / TPU primitives: attention kernels, 3D RoPE, and
device-mesh / sharding helpers. These are reusable on their own -- see
``docs/library_usage.md``.
"""

from .attention import (
    RMSNorm,
    TPShardedRMSNorm,
    chunk_by_rank,
    dot_product_attention,
    local_attention,
    sequence_parallel_joint_self_attention,
    sequence_parallel_self_attention,
)
from .rope3d import apply_rope3d, create_rope3d_freqs, sinusoidal_embedding_1d
from .sharding import (
    build_tpu_mesh,
    configure_jax_cache,
    get_batch_sharding,
    get_replicated_sharding,
    shard_wan_params,
    to_partition_specs,
)

__all__ = [
    "RMSNorm",
    "TPShardedRMSNorm",
    "chunk_by_rank",
    "dot_product_attention",
    "local_attention",
    "sequence_parallel_joint_self_attention",
    "sequence_parallel_self_attention",
    "apply_rope3d",
    "create_rope3d_freqs",
    "sinusoidal_embedding_1d",
    "build_tpu_mesh",
    "configure_jax_cache",
    "get_batch_sharding",
    "get_replicated_sharding",
    "shard_wan_params",
    "to_partition_specs",
]
