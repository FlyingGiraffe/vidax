# Using `vidax` as a library

The standalone scripts under [`examples/`](../examples/) are the primary way to
run a model — each is a complete, TPU-ready pipeline for one family/task (see
[`docs/models/`](models/)). This page is for the other case: `import vidax` and
reuse the pieces — the Pallas flash-attention kernel, the diffusion schedulers,
the PyTorch→JAX translator, or a model's DiT/VAE modules — inside your own code.

This is the task-oriented guide. For the exhaustive per-symbol reference —
every function/class, its full signature, and how to call it — see
[`docs/api/`](api/index.md).

## Install

```bash
pip install vidax           # + "vidax[tpu]" on a Cloud TPU VM
```

`torch`, `transformers`, and `sentencepiece` come as regular dependencies
(checkpoint deserialization + tokenizers only — `vidax.models` code never
imports them).

## What's importable

| You want… | Import from | Key names |
| --- | --- | --- |
| Version | `vidax` | `__version__` |
| Flash attention, RoPE, device mesh | `vidax.core` | `dot_product_attention`, `sequence_parallel_self_attention`, `local_attention`, `RMSNorm`, `create_rope3d_freqs`, `apply_rope3d`, `sinusoidal_embedding_1d`, `build_tpu_mesh`, `get_replicated_sharding`, `get_batch_sharding`, `configure_jax_cache` |
| Diffusion samplers | `vidax.schedulers` | `RectifiedFlowScheduler`, `FlowUniPCMultistepScheduler`, `UniPCState` (family-specific: `vidax.schedulers.cogvideox`, `.ltx_rectified_flow`, `.ltx2_5_ancestral_euler`) |
| PyTorch → Flax weights | `vidax.translator` | `load_torch_checkpoint_to_jax`, `convert_pt_tensor_to_jax` |
| A model's modules | `vidax.models.<family>…` | e.g. `vidax.models.wan.wan2_1` → `WanDiT`, `WanVAEDecoder`, `WanVAEEncoder`; presets in each family's `configs.py` |

The top-level package lazily re-exports the five most common entry points, so
`from vidax import load_torch_checkpoint_to_jax, RectifiedFlowScheduler,
FlowUniPCMultistepScheduler, build_tpu_mesh, dot_product_attention` works
without importing any model subpackage. Full signatures for everything in
this table are in [`docs/api/`](api/index.md)
([core](api/attention.md) · [rope](api/rope.md) · [sharding](api/sharding.md)
· [schedulers](api/schedulers.md) · [translator](api/translator.md)); for the
full module tree see [`directory_layout.md`](directory_layout.md).

## Standalone: Pallas flash attention

`vidax.core.dot_product_attention` takes `q, k, v` shaped
`(batch, seq, num_heads, head_dim)` and, on TPU, runs a real O(seq)-memory
flash-attention kernel instead of materializing the `(B, H, Sq, Sk)` score
matrix. On CPU/GPU it falls back to `jax.nn.dot_product_attention`, so this
snippet runs anywhere:

```python
import jax, jax.numpy as jnp
from vidax.core import dot_product_attention

B, S, H, D = 2, 4096, 8, 64
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
q = jax.random.normal(k1, (B, S, H, D), jnp.bfloat16)
k = jax.random.normal(k2, (B, S, H, D), jnp.bfloat16)
v = jax.random.normal(k3, (B, S, H, D), jnp.bfloat16)

out = dot_product_attention(q, k, v)          # (B, S, H, D)
print(out.shape, out.dtype)
```

Multi-device on TPU: build a mesh, shard the heads across the `tp` axis, and
pass `mesh=` (Mosaic kernels can't infer sharding on their own):

```python
from vidax.core import build_tpu_mesh
mesh = build_tpu_mesh(data_parallel_size=1, tensor_parallel_size=jax.device_count())
# ... place q/k/v with NamedSharding(mesh, P('dp', None, 'tp', None)) ...
out = dot_product_attention(q, k, v, mesh=mesh)
```

`sequence_parallel_self_attention` is the alternative DeepSpeed-Ulysses
(sequence-sharded) path — see [`hardware_and_sharding.md`](hardware_and_sharding.md).

## Standalone: a diffusion scheduler

`RectifiedFlowScheduler` is a flow-matching Euler sampler. Precompute `sigmas`/
`timesteps`, then call `.step(velocity, i, x)` in a loop over your own denoiser:

```python
import jax, jax.numpy as jnp
from vidax.schedulers import RectifiedFlowScheduler

sched = RectifiedFlowScheduler(num_steps=50, shift=5.0)   # .sigmas, .timesteps: (51,)
x = jax.random.normal(jax.random.PRNGKey(0), (1, 16, 32, 32))

def my_denoiser(x, t):        # replace with a real DiT: returns predicted velocity
    return jnp.zeros_like(x)

for i in range(sched.num_steps):
    v = my_denoiser(x, sched.timesteps[i])   # feed timesteps (~[0,1000]), not sigmas
    x = sched.step(v, i, x)
```

`FlowUniPCMultistepScheduler` is the from-scratch UniPC multistep
predictor-corrector (used by Cosmos); it threads an explicit `UniPCState`
through the loop so it stays `jax.jit`/`lax.scan`-friendly. `vidax.schedulers`
also has the CogVideoX DDIM/DPM and both LTX samplers.

## Standalone: translate a PyTorch checkpoint

`load_torch_checkpoint_to_jax(path, model_type=...)` reads a `.safetensors` /
`.pth` / sharded `*.index.json` checkpoint and returns a Flax parameter pytree
with keys and tensor layouts already remapped:

```python
from vidax.translator import load_torch_checkpoint_to_jax

params = load_torch_checkpoint_to_jax(
    "checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    model_type="wan_dit",
)
```

`model_type` selects the key mapping. Available values (see
`src/vidax/translator/mappings/__init__.py`): `wan_dit`, `wan2.1_vae`,
`wan2.1_clip`, `wan2.2_vae`, `wan2.2_vae_diffusers`, `wan_t5`, `cosmos2.5_dit`,
`cosmos3_dit`, `reason1_text_encoder`, `cogvideox_dit`, `cogvideox_vae`,
`ltx_video_dit`, `ltx_video_vae`, `ltx_video_t5`, `ltx2_5_dit`,
`ltx2_5_connector`, `ltx2_5_vae`, `ltx2_5_diffusion_decoder`, `gemma4_text`,
`hunyuan_video1_5_{dit,vae,byt5,siglip}`,
`hunyuan_video_{dit,vae,llama_text,clip_text,clip_vision,llava_llama_text,llava_projector}`.

## End-to-end: a full model

The pattern every `examples/generate_*.py` script follows:

```python
import jax
from vidax.core import build_tpu_mesh
from vidax.schedulers import RectifiedFlowScheduler
from vidax.translator import load_torch_checkpoint_to_jax
from vidax.models.wan.wan2_1 import WanDiT, WanVAEDecoder
from vidax.models.wan.wan2_1.configs import T2V_1_3B_CONFIG

mesh = build_tpu_mesh(data_parallel_size=1, tensor_parallel_size=jax.device_count())

dit = WanDiT(**T2V_1_3B_CONFIG, mesh=mesh)
dit_params = load_torch_checkpoint_to_jax(".../diffusion_pytorch_model.safetensors", model_type="wan_dit")
vae_params = load_torch_checkpoint_to_jax(".../Wan2.1_VAE.pth", model_type="wan2.1_vae")

# 1. encode the prompt with the T5 tower (vidax.models.wan.common.t5)
# 2. sample: for i in range(steps): v = dit.apply(...); x = scheduler.step(v, i, x)
# 3. decode: frames = WanVAEDecoder(...).apply(vae_params, x)
```

The wiring the comment glosses — CFG batching, RoPE frequency tables, sequence
lengths, per-chunk VAE `jit`, `donate_argnums` on the sample step, dtype
control, tensor/sequence-parallel `PartitionSpec`s, optional weight offloading —
is exactly what the corresponding `examples/generate_<family>*.py` script
already implements end to end. Start from that script rather than rebuilding it;
this page is only the map of the parts.
