# `vidax.translator` — PyTorch → Flax checkpoint loading

Loads released PyTorch `.safetensors` / `.pth` checkpoints straight into a
Flax parameter pytree — key names remapped and tensor layouts transposed to
what `vidax.models.*` modules expect, verified per model by an exact 1:1
parameter-tree match against the reference.

```python
from vidax.translator import load_torch_checkpoint_to_jax, convert_pt_tensor_to_jax
from vidax.translator.mappings import map_ltx2_5_dit_keys   # etc. — full list below
```

`vidax.translator` re-exports `load_torch_checkpoint_to_jax`,
`convert_pt_tensor_to_jax`, and the Wan `map_*` functions; every other
`map_*_keys` function is imported from `vidax.translator.mappings`.
`torch` is only needed for `.pth`/`.pt`/`.bin` inputs (`.safetensors` is
read with `safetensors.numpy`); it is a core dependency either way.

---

## `load_torch_checkpoint_to_jax`

```python
load_torch_checkpoint_to_jax(
    checkpoint_path: str,
    model_type: str = "wan2.1_dit",
) -> dict
```

Reads a checkpoint from disk and returns a Flax parameter dict.

- **`checkpoint_path`** — a `.safetensors` file, a `.safetensors.index.json`
  shard manifest (every referenced shard is loaded and merged), or a
  `.pth` / `.pt` / `.bin` (`torch.load(..., weights_only=True)`, with
  `io.BytesIO` allow-listed for Cosmos's `_extra_state` blobs).
- **`model_type`** — selects the key mapping (table below). Unknown values
  raise `NotImplementedError`.
- **Returns** a nested dict of **numpy** arrays (not `jax.Array` — kept in
  host RAM so the first device-resident copy is made, already sharded, by
  your `jax.device_put(params, sharding_tree)`; converting to `jax.Array`
  here would pile a multi-GB tree onto one device and OOM). bf16 tensors are
  preserved as bf16.

```python
params = load_torch_checkpoint_to_jax(
    "checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    model_type="wan_dit",
)
params = jax.device_put(params, shard_wan_params(params, mesh))
```

### `model_type` values

| Family | `model_type` | Target |
| --- | --- | --- |
| Wan 2.1 / 2.2 | `wan_dit` (aliases: `wan2.1_dit`, `wan2.2_dit`) | `WanDiT` (2.1 or 2.2 — identical DiT key names) |
| | `wan_t5` | `T5Encoder` (UMT5-XXL; shared, byte-identical across 2.1/2.2) |
| | `wan2.1_vae` | Wan2.1 `WanVAEEncoder` / `WanVAEDecoder` |
| | `wan2.1_clip` | Wan2.1 `ClipVisionTransformer` (I2V) |
| | `wan2.2_vae` | Wan2.2 VAE (original-repo layout) |
| | `wan2.2_vae_diffusers` | Wan2.2 VAE in diffusers' `AutoencoderKLWan` layout (also Cosmos3's VAE) |
| Cosmos-Predict2.5 | `cosmos2.5_dit` | `CosmosDiT` (2B / 14B) |
| | `reason1_text_encoder` | `Qwen2TextModel` (Reason1 / Qwen2.5-VL-7B text tower — pass the `.index.json`) |
| Cosmos3 | `cosmos3_dit` | Cosmos3 DiT (Nano / Edge) |
| CogVideoX | `cogvideox_dit` / `cogvideox_vae` | CogVideoX DiT / VAE |
| LTX-Video 0.9.8 | `ltx_video_dit` / `ltx_video_vae` / `ltx_video_t5` | LTX-Video DiT / VAE / T5-XXL |
| LTX-2.5 | `ltx2_5_dit` | LTX-2.5 DiT (+ embedded video-embeddings connector) |
| | `ltx2_5_connector` | the embeddings connector alone |
| | `ltx2_5_vae` | conv-decoder VAE |
| | `ltx2_5_diffusion_decoder` | NATTEN transformer VAE decoder (`--vae_variant diffusion`) |
| | `gemma4_text` | Gemma-4 12B text encoder |
| HunyuanVideo-1.5 | `hunyuan_video1_5_dit` / `_vae` / `_byt5` / `_siglip` | 1.5 DiT / VAE / byT5 glyph encoder / SigLIP vision encoder |
| HunyuanVideo 1.0 | `hunyuan_video_dit` / `hunyuan_video_vae` | 1.0 DiT (T2V **and** I2V checkpoints) / VAE |
| | `hunyuan_video_llama_text` / `hunyuan_video_clip_text` | T2V: Llama3-8B `.language_model` tower / CLIP-L pooled text |
| | `hunyuan_video_llava_llama_text` / `hunyuan_video_clip_vision` / `hunyuan_video_llava_projector` | I2V: full multimodal LLaVA — language model / CLIP ViT-L/14-336 vision tower / 2-layer projector |

---

## `map_*_keys` functions

Each `map_<...>_keys(pt_state_dict: dict) -> dict` takes a **raw PyTorch
state_dict already in memory** and returns the Flax pytree —
`load_torch_checkpoint_to_jax` is just `_load_pt_state_dict(path)` piped into
the matching one. Call a `map_*` directly when you already hold a state_dict
(e.g. you loaded shards yourself, or you're translating a checkpoint held by
another process).

```
map_wan_dit_keys              map_wan_t5_keys
map_wan2_1_dit_keys           map_wan2_1_vae_keys           map_wan2_1_clip_keys
map_wan2_2_vae_keys           map_wan2_2_vae_diffusers_keys
map_cosmos2_5_dit_keys        map_cosmos3_dit_keys         map_reason1_text_encoder_keys
map_cogvideox_dit_keys        map_cogvideox_vae_keys
map_ltx_video_dit_keys        map_ltx_video_vae_keys       map_ltx_video_t5_keys
map_ltx2_5_dit_keys           map_ltx2_5_connector_keys    map_ltx2_5_vae_keys
map_ltx2_5_diffusion_decoder_keys                          map_gemma4_text_keys
map_hunyuan_video1_5_dit_keys map_hunyuan_video1_5_vae_keys
map_hunyuan_video1_5_byt5_keys map_hunyuan_video1_5_siglip_keys
map_hunyuan_video_dit_keys    map_hunyuan_video_vae_keys
map_hunyuan_video_llama_text_keys      map_hunyuan_video_clip_text_keys
map_hunyuan_video_llava_llama_text_keys map_hunyuan_video_clip_vision_keys
map_hunyuan_video_llava_projector_keys
```

`map_wan_dit_keys` is what `model_type="wan_dit"` dispatches to (2.1 and 2.2
share DiT key names); `map_wan2_1_dit_keys` is also exported for callers that
want the 2.1-specific mapper by name.

---

## `convert_pt_tensor_to_jax`

```python
convert_pt_tensor_to_jax(key: str, pt_array) -> numpy.ndarray
```

Converts **one** state_dict tensor to Flax layout, inferring the transform
from `key`:

| Input | Output |
| --- | --- |
| key ends `.gamma` (channel-first RMSNorm scale, e.g. `(dim,1,1)`) | flattened to `(dim,)` |
| key is `modulation` / `*.modulation` (Wan raw AdaLN `nn.Parameter`) | passed through unchanged |
| 5D (`Conv3d` `(O,I,T,H,W)`) | `(T,H,W,I,O)` |
| 4D (`Conv2d` `(O,I,H,W)`) | `(H,W,I,O)` |
| 2D (`Linear` `(O,I)`) | `(I,O)` |
| 1D bias / norm scale | unchanged |

Returns a **numpy** array (deliberately not `jax.Array`; see
`load_torch_checkpoint_to_jax`'s note). This is the per-leaf primitive the
`map_*` functions call.

## `pt_tensor_to_numpy`

```python
pt_tensor_to_numpy(pt_array) -> numpy.ndarray
```

Materializes a `torch.Tensor` (any device/dtype) or an array-like as numpy,
**preserving bfloat16** via `ml_dtypes` (reinterprets the raw `uint16` bits
rather than upcasting to float32). No layout change — that's
`convert_pt_tensor_to_jax`'s job.
