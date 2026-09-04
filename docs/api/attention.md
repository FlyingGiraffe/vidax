# `vidax.core` — attention & normalization

Source: [`src/vidax/core/attention.py`](../../src/vidax/core/attention.py).
Import from `vidax.core` (or `vidax.core.attention`).

```python
from vidax.core import (
    dot_product_attention,
    local_attention,
    sequence_parallel_self_attention,
    sequence_parallel_joint_self_attention,
    RMSNorm,
    TPShardedRMSNorm,
    chunk_by_rank,
)
```

All attention functions take `q, k, v` shaped **`(B, S, num_heads, head_dim)`**
and return **`(B, S_q, num_heads, head_dim)`**. On TPU they route to the JAX
Pallas flash-attention kernel (`jax.experimental.pallas.ops.tpu.flash_attention`),
which is `O(S)` in memory instead of materializing the `(B, num_heads, S_q, S_k)`
score matrix; on CPU/GPU they fall back to `jax.nn.dot_product_attention`. The
kernel requires sequence lengths to be multiples of 128 — inputs are
zero-padded and the padding excluded via segment ids, transparently.

---

## `dot_product_attention`

```python
dot_product_attention(
    q, k, v,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    scale: float | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array
```

Full (non-causal) dot-product attention — the Megatron-style path, where
attention **heads** are sharded and every device holds the full token
sequence.

| Arg | Meaning |
| --- | --- |
| `q, k, v` | `(B, S, num_heads, head_dim)`. `k`/`v` may have a different sequence length than `q` (cross-attention). |
| `bias` | Optional **additive** bias, broadcastable to `(B, num_heads, S_q, S_k)` — e.g. T5 relative-position bias, or a large-negative padding mask. |
| `mask` | Optional **boolean** mask, broadcastable to `(B, num_heads, S_q, S_k)`. Supplying it forces the non-flash fallback (the kernel has no boolean-mask input). |
| `scale` | Softmax scale; default `1/sqrt(head_dim)`. T5 attention passes `scale=1.0`. |
| `mesh` | The device mesh `q/k/v` are sharded over. **Required on TPU whenever running on more than one device** — Mosaic kernels can't infer sharding, so the call is wrapped in `shard_map` internally using this mesh. |

**Dispatch:** the flash kernel is used when `mask is None`, on TPU, and
either single-device *or* (`mesh` given **and** `bias is None`). An additive
`bias` on a single device still takes the flash path; `bias` + multi-device
falls back to the correct-but-slower XLA path (the sharded flash path
doesn't thread `bias` yet). Multi-device with `mesh=None` also falls back.

```python
out = dot_product_attention(q, k, v)                       # single device
out = dot_product_attention(q, k, v, mesh=mesh)            # multi-device TP
out = dot_product_attention(q, k, v, bias=t5_pos_bias, scale=1.0)
```

---

## `local_attention`

```python
local_attention(q, k, v, scale: float | None = None) -> jax.Array
```

Plain dot-product attention that **always** runs as a single local
(non-cross-device) call. Use it from *inside* an already per-device context
(e.g. within your own `shard_map` body) where `dot_product_attention`'s
`jax.device_count() > 1` heuristic would wrongly pick the materializing
fallback. On TPU it calls the flash kernel directly; elsewhere
`jax.nn.dot_product_attention`. No `bias`/`mask`/`mesh` parameters.

---

## `sequence_parallel_self_attention`

```python
sequence_parallel_self_attention(
    q, k, v, sp_axis_name: str, scale: float | None = None,
) -> jax.Array
```

DeepSpeed-Ulysses sequence-parallel self-attention
([arXiv:2309.14509](https://arxiv.org/abs/2309.14509)). The **sequence** is
sharded between blocks; this call reshuffles to a head-sharded view of the
full sequence for the duration of attention via two `all_to_all`s, then
back.

- `q, k, v`: `(B, L_local, num_heads, head_dim)` — this device's local
  sequence chunk, full heads.
- `sp_axis_name`: name of the mesh axis to reshuffle across (the `'sp'` axis
  of a [`build_tpu_mesh`](sharding.md#build_tpu_mesh) mesh).
- **Must be called from inside an active `shard_map`** over a mesh with that
  axis. Returns `(B, L_local, num_heads, head_dim)`.

---

## `sequence_parallel_joint_self_attention`

```python
sequence_parallel_joint_self_attention(
    q, k, v, text_len: int, sp_axis_name: str, scale: float | None = None,
) -> jax.Array
```

Ulysses sequence-parallel attention over a **joint `[text; visual]`** token
sequence where only the visual tokens are sequence-chunked and the
`text_len` text-prefix tokens are replicated on every device (CogVideoX's
layout — one joint self-attention per block, no separate cross-attention).

- `q, k, v`: `(B, text_len + L_visual_local, num_heads, head_dim)`.
- `text_len`: number of replicated text-prefix tokens (static). `num_heads`
  must be divisible by the `sp` axis size.
- **Must be called from inside an active `shard_map`.** Returns the same
  layout as the input.

---

## `RMSNorm`

```python
class RMSNorm(flax.linen.Module):
    dim: int
    eps: float = 1e-6

    def __call__(self, x: jax.Array) -> jax.Array
```

RMSNorm matching Wan's `WanRMSNorm`: reduces over the last axis in float32,
casts back to the input dtype, multiplies by a learned `scale` param of
shape `(dim,)`. Standard Flax module — `RMSNorm(dim=1024).init(rng, x)` /
`.apply(params, x)`.

## `TPShardedRMSNorm`

```python
class TPShardedRMSNorm(flax.linen.Module):
    dim_local: int
    global_dim: int
    eps: float = 1e-6

    def __call__(self, x: jax.Array) -> jax.Array
```

RMSNorm whose mean-square reduction spans a **Megatron-column-sharded**
feature axis: it sums each device's local sum-of-squares with
`jax.lax.psum(..., 'tp')` before normalizing, so the statistic matches the
unsharded reference. `dim_local` is this device's channel count,
`global_dim` the full feature width. **Must be called from inside
`shard_map` over a mesh with a `'tp'` axis** (`psum` needs the bound axis).
The `scale` param is itself `'tp'`-sharded (see
[`shard_wan_params`](sharding.md#shard_wan_params)).

## `chunk_by_rank`

```python
chunk_by_rank(x: jax.Array, axis: int, sp_size: int, rank: jax.Array) -> jax.Array
```

Slices out this device's contiguous `1/sp_size` share of `x` along `axis`,
indexed by a **traced** `rank` (e.g. `jax.lax.axis_index('sp')` from inside
`shard_map`). `sp_size` must be a static Python int, so only even splits are
supported. Used to chunk the token sequence (and per-token modulation
tensors) before a sequence-parallel block loop.
