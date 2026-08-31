"""CogVideoX's text encoder (Flax/JAX).

CogVideoX conditions on a frozen `t5-v1.1-xxl` encoder -- the *same*
architecture (`d_model=4096`, `num_heads=64`, `d_kv=64`, `d_ff=10240`,
`num_layers=24`, gated-GELU FFN, shared relative-position bias, no biases,
`layer_norm_epsilon=1e-6`) as the `PixArt-alpha/PixArt-XL-2-1024-MS`
text encoder that `vidax.models.ltx_video.t5` already ports and that
`vidax.translator.mappings.map_ltx_video_t5_keys` already translates
(confirmed 1:1 against CogVideoX-5b's `text_encoder/` key list).

So there is nothing model-specific to add: this module just re-exports
`T5Encoder` and the tokenizer wrapper from `ltx_video.t5`. The only
CogVideoX-specific choice is the tokenizer sequence length -- 226
(`max_text_seq_length`), padded to `max_length` with `add_special_tokens`
(the diffusers `_get_t5_prompt_embeds` convention). `PixArtT5Tokenizer`
already does `padding="max_length", truncation=True, max_length=seq_len`
with `AutoTokenizer`'s default `add_special_tokens=True`, so
`CogVideoXT5Tokenizer(path, seq_len=226)` is exactly right.

Load its weights with
`load_torch_checkpoint_to_jax(<repo>/text_encoder/model.safetensors.index.json,
model_type="ltx_video_t5")`.
"""
from vidax.models.ltx_video.t5 import PixArtT5Tokenizer as CogVideoXT5Tokenizer
from vidax.models.ltx_video.t5 import T5Block, T5Encoder

MAX_TEXT_SEQ_LENGTH = 226

__all__ = ["T5Encoder", "T5Block", "CogVideoXT5Tokenizer", "MAX_TEXT_SEQ_LENGTH"]
