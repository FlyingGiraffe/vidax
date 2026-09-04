# `vidax.core` — rotary & timestep embeddings

Source: [`src/vidax/core/rope3d.py`](../../src/vidax/core/rope3d.py). Import
from `vidax.core` (or `vidax.core.rope3d`).

```python
from vidax.core import create_rope3d_freqs, apply_rope3d, sinusoidal_embedding_1d
```

This is the Wan-family 3D RoPE: it rotates adjacent `(even, odd)` coordinate
pairs (the `torch.view_as_complex` convention, **not** GPT-NeoX
"rotate-half"), and splits the `head_dim // 2` frequency pairs unevenly
across the time/height/width axes. Other model families have their own RoPE
under `vidax.models.<family>.rope` / `.mrope` — see
[`docs/directory_layout.md`](../directory_layout.md).

---

## `create_rope3d_freqs`

```python
create_rope3d_freqs(
    t: int, h: int, w: int,
    head_dim: int,
    theta: float = 10000.0,
    max_seq_len: int = 1024,
) -> tuple[jax.Array, jax.Array]   # (cos, sin)
```

Builds per-position `(cos, sin)` rotation angles for a patchified `(T, H, W)`
video grid.

| Arg | Meaning |
| --- | --- |
| `t, h, w` | Patchified grid sizes along time, height, width. |
| `head_dim` | Per-head channel dimension (must be even). Of the `c = head_dim // 2` frequency pairs, `c - 2*(c//3)` go to T and `c//3` each to H and W. |
| `theta` | RoPE base frequency. |
| `max_seq_len` | Size of the precomputed per-axis frequency bank (matches the reference's fixed 1024). Must be `>= max(t, h, w)`. |

**Returns** `(cos, sin)`, each shaped `(1, t*h*w, 1, head_dim // 2)`, ready
to broadcast against a `(B, S, num_heads, head_dim)` tensor.

---

## `apply_rope3d`

```python
apply_rope3d(x: jax.Array, freqs: tuple[jax.Array, jax.Array]) -> jax.Array
```

Applies the rotary embedding to `x` shaped `(B, S, num_heads, head_dim)` by
pairwise complex rotation (`x1' = x1·cos − x2·sin`, `x2' = x2·cos + x1·sin`
over the even/odd channel split). Computed in float32, cast back. `freqs` is
the `(cos, sin)` pair from `create_rope3d_freqs`. Returns the rotated tensor,
same shape as `x`.

```python
freqs = create_rope3d_freqs(t=21, h=30, w=52, head_dim=128)
q = apply_rope3d(q, freqs)
k = apply_rope3d(k, freqs)
```

---

## `sinusoidal_embedding_1d`

```python
sinusoidal_embedding_1d(dim: int, position: jax.Array) -> jax.Array
```

1D sinusoidal embedding for diffusion timesteps. `dim` must be even;
`position` is shape `(B,)` (the timesteps, typically `scheduler.timesteps[i]`
broadcast to the batch). **Returns** `(B, dim)`, laid out as
`[cos(freqs), sin(freqs)]`.
