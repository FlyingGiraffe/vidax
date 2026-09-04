# API Reference

Reference documentation for the reusable building blocks in `vidax` — every
public function and class, its signature, and how to call it. For a
task-oriented walkthrough ("I want to run flash attention / a scheduler / the
translator") start with [`docs/library_usage.md`](../library_usage.md); this
section is the exhaustive per-symbol reference it links into.

Not covered here: the per-model DiT/VAE/text-encoder modules under
`vidax.models.<family>` — those are documented per family in
[`docs/models/`](../models/), with the full source tree in
[`docs/directory_layout.md`](../directory_layout.md).

| Page | Module | Contents |
| --- | --- | --- |
| [attention.md](attention.md) | `vidax.core` | `dot_product_attention` (Pallas TPU flash attention), the sequence-parallel attention variants, `local_attention`, `RMSNorm` / `TPShardedRMSNorm`, `chunk_by_rank` |
| [rope.md](rope.md) | `vidax.core` | `create_rope3d_freqs`, `apply_rope3d`, `sinusoidal_embedding_1d` |
| [sharding.md](sharding.md) | `vidax.core` | `build_tpu_mesh`, `get_replicated_sharding`, `get_batch_sharding`, `shard_wan_params`, `to_partition_specs`, `configure_jax_cache` |
| [schedulers.md](schedulers.md) | `vidax.schedulers` | `RectifiedFlowScheduler`, `FlowUniPCMultistepScheduler` + `UniPCState`, and the family-specific samplers (CogVideoX, LTX-Video, LTX-2.5) |
| [translator.md](translator.md) | `vidax.translator` | `load_torch_checkpoint_to_jax` (+ the full `model_type` table), `convert_pt_tensor_to_jax`, `pt_tensor_to_numpy`, and the per-model `map_*_keys` functions |

## Import conventions

The most common entry points are re-exported from the top-level package and
from `vidax.core`:

```python
from vidax import (
    load_torch_checkpoint_to_jax,      # vidax.translator
    RectifiedFlowScheduler,            # vidax.schedulers (Wan's)
    FlowUniPCMultistepScheduler,       # vidax.schedulers
    build_tpu_mesh,                    # vidax.core
    dot_product_attention,            # vidax.core
)
from vidax.core import (
    apply_rope3d, create_rope3d_freqs, RMSNorm,
    get_replicated_sharding, get_batch_sharding, configure_jax_cache,
)
```

Everything else is imported from its module directly
(`from vidax.schedulers.cogvideox import CogVideoXDDIMScheduler`,
`from vidax.translator.mappings import map_ltx2_5_dit_keys`, …).

## Conventions used on these pages

- **Array shapes** are written `(B, S, num_heads, head_dim)` etc.; `B` is
  batch, `S`/`L`/`N` a token-sequence length.
- Functions that must run **inside `jax.experimental.shard_map.shard_map`**
  (because they issue collectives or call an un-partitionable Mosaic kernel)
  say so explicitly — calling them at top level on sharded inputs raises.
- These primitives are plain JAX: they take and return `jnp.ndarray`s (or
  Flax `nn.Module`s), hold no global state, and are `jax.jit`-friendly
  unless noted.
