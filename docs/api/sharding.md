# `vidax.core` — device mesh & tensor-parallel sharding

Source: [`src/vidax/core/sharding.py`](../../src/vidax/core/sharding.py).
Import from `vidax.core` (or `vidax.core.sharding`).

```python
from vidax.core import (
    build_tpu_mesh,
    get_replicated_sharding,
    get_batch_sharding,
    shard_wan_params,
    to_partition_specs,
    configure_jax_cache,
)
```

The sharding scheme is standard Megatron-LM 1D tensor parallelism: the
hidden/residual stream stays replicated across the `'tp'` axis (needed for
correct norm reductions); attention QKV and FFN up-projections are
**column**-sharded (split along whole heads / FFN channels); attention
output and FFN down-projections are **row**-sharded (GSPMD inserts the
all-reduce back to a replicated output). See
[`docs/hardware_and_sharding.md`](../hardware_and_sharding.md) for the
reasoning and how sequence parallelism composes on top.

---

## `build_tpu_mesh`

```python
build_tpu_mesh(
    data_parallel_size: int,
    tensor_parallel_size: int,
    sequence_parallel_size: int = 1,
) -> jax.sharding.Mesh
```

Creates a 3-axis `jax.sharding.Mesh` with axis names **`('dp', 'tp', 'sp')`**
for TPU v4/v5e/v6e. Asserts `dp * tp * sp == jax.device_count()`. A size-1
axis is invisible to any `PartitionSpec` that doesn't name it, so
2-axis callers can leave `sequence_parallel_size=1`.

```python
import jax
mesh = build_tpu_mesh(data_parallel_size=1,
                      tensor_parallel_size=jax.device_count())
```

---

## `get_replicated_sharding`

```python
get_replicated_sharding(mesh) -> jax.sharding.NamedSharding
```

`NamedSharding(mesh, P())` — every device holds the whole array. Use for
weights/constants that aren't tensor-parallel-sharded.

## `get_batch_sharding`

```python
get_batch_sharding(mesh, ndim: int) -> jax.sharding.NamedSharding
```

`NamedSharding(mesh, P('dp', None, …))` for an array of rank `ndim` — shards
only the leading (batch) axis across `'dp'`, replicates everything else.
Use for latents, token ids, and text embeddings.

```python
latents = jax.device_put(latents, get_batch_sharding(mesh, latents.ndim))
```

---

## `shard_wan_params`

```python
shard_wan_params(params: dict, mesh) -> dict   # pytree of NamedSharding
```

Walks a **WanDiT / T5Encoder / CosmosDiT** parameter pytree and returns a
matching pytree of `NamedSharding`, assigning the Megatron column/row-parallel
spec to each Dense `kernel`/`bias` by matching the parent module name against
built-in `COLUMN_PARALLEL_NAMES` / `ROW_PARALLEL_NAMES` sets; everything else
is replicated. Feed the result straight to `jax.device_put`:

```python
params = jax.device_put(params, shard_wan_params(params, mesh))
```

The name is historical — the dispatch is entirely name-pattern-driven and
also covers T5 and Cosmos-Predict2.5. It does **not** cover every model
family; families with a different block naming convention shard in their own
example script.

## `to_partition_specs`

```python
to_partition_specs(shardings) -> pytree of PartitionSpec
```

Strips a `NamedSharding` pytree (e.g. from `shard_wan_params`) down to bare
`PartitionSpec`s, for passing as `shard_map`'s `in_specs` when the whole DiT
forward runs inside one `shard_map` — so the weights stay genuinely
weight-sharded instead of falling back to a blanket replicated `P()`.

---

## `configure_jax_cache`

```python
configure_jax_cache(cache_dir: str = "~/.cache/vidax/jax") -> None
```

Enables JAX's persistent on-disk compilation cache (keyed by program hash),
so a second run at the same `(shape, dtype, sharding, mesh, step count)`
signature skips XLA compilation entirely. Also sets
`jax_persistent_cache_min_entry_size_bytes = -1` and
`jax_persistent_cache_min_compile_time_secs = 1`. Call once, **before**
building any mesh or model. Every `examples/generate_*.py` calls this first
thing in `main()`.
