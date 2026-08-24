"""Gemma-4 (12B, `gemma4-12b-ltx-v1`) text encoder (Flax/JAX), text-only.

A structural port of `transformers.models.gemma4_unified.modeling_
gemma4_unified.{Gemma4UnifiedTextModel,Gemma4UnifiedTextDecoderLayer,
Gemma4UnifiedTextAttention,Gemma4UnifiedTextMLP,Gemma4UnifiedRMSNorm,
Gemma4UnifiedTextRotaryEmbedding}` (installed in the `ltx2-verify` conda
env's `transformers==5.14.1`; LTX-2.5 loads this exact HF architecture via
`AutoModelForImageTextToText`, not a custom implementation of its own -- see
`refs/LTX-2-main/packages/ltx-core/src/ltx_core/text_encoders/gemma/
encoders/{base_encoder,encoder_configurator}.py`). Only the text tower is
ported (`Gemma4UnifiedTextModel`, no `lm_head`, no vision/audio towers --
`LTXGemmaTextEncoder.encode` calls `self.model.model` for exactly this
reason: it skips both).

Real config, read from the checkpoint's own embedded `gemma_config` JSON
metadata (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`), not assumed --
see `vidax.models.ltx2_5.configs.GEMMA4_TEXT_CONFIG`:
`hidden_size=3840`, `num_hidden_layers=48`, `num_attention_heads=16`,
`num_key_value_heads=8`, `head_dim=256` (sliding/local layers),
`global_head_dim=512`/`num_global_key_value_heads=1` (full/global layers,
every 6th layer per `layer_types`), `intermediate_size=15360`,
`hidden_activation="gelu_pytorch_tanh"`, `attention_bias=False`,
`attention_k_eq_v=True` (full-attention layers only), `rms_norm_eps=1e-6`,
`sliding_window=1024`. Architecturally load-bearing quirks, easy to miss
from the class names alone:

- **Fixed `scaling=1.0`** in attention (not the usual `head_dim**-0.5`) --
  `q_norm`'s learnable per-head RMSNorm scale already does that job.
- **Sandwich norm.** Both self-attention and the MLP are wrapped
  `residual + post_norm(sublayer(pre_norm(residual)))` -- *four* RMSNorms
  per layer (`input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm`), not the usual
  two.
- **Per-layer `layer_scalar`.** A trained (not fixed-at-1.0, verified
  against the real checkpoint: `0.053`, `0.356`, `0.050` for layers
  `0`/`5`/`47`) scalar multiplying each layer's output before the next
  layer -- must be loaded, not skipped as a vestigial buffer.
  the checkpoint carries real values.
- **`attention_k_eq_v`** (full/global layers only): no separate `v_proj`
  weight -- V reuses K's *pre-`k_norm`, pre-RoPE* projection output, then
  gets its *own* (always weightless, `with_scale=False`) `v_norm`. Verified
  against the real checkpoint: layer 5 (a `full_attention` layer) has no
  `self_attn.v_proj.weight` key at all.
- **`q_norm`/`k_norm`/`v_norm` are per-head** (`RMSNorm(head_dim)`, applied
  after the `(B, S, H, D)` reshape but before the `(B, H, S, D)` transpose),
  unlike LTX's own DiT `q_norm`/`k_norm` which normalize the full
  concatenated `inner_dim`.
- **"proportional" RoPE** on full/global layers (`partial_rotary_factor=
  0.25`, `theta=1e6`) doesn't slice the head dim -- it zeros out the
  trailing `(1 - 0.25) * head_dim // 2` inverse-frequency entries
  (`cos=1, sin=0` there), so the *same* rotate-half math as sliding layers
  applies uniformly across the full `head_dim`, just with most of it
  rotated by angle 0. Sliding/local layers use plain NeoX-style RoPE
  (`theta=1e4`, full rotation, no zeroing).
- **Scaled embedding.** `embed_tokens(ids) * sqrt(hidden_size)`, with the
  scale itself rounded to the embedding weight's dtype *before*
  multiplying (`hidden_size**0.5` computed once in float32, then cast down)
  -- reproduces a known, intentionally-preserved bf16 rounding quirk
  (`sqrt(3840)` rounds to a slightly different bf16 value than the true
  float32 result), not a numerical bug to "fix".

`extract_video_features` implements `FeatureExtractorV2` (`norm_and_concat_
per_token_rms` + `video_aggregate_embed`, both from the same checkpoint
file) -- the per-token, per-layer RMS-normalize-and-concatenate-then-project
step that turns Gemma's 49 layers' worth of `(B, S, 3840)` hidden states
into a single `(B, S, 4096)` tensor for
`vidax.models.ltx2_5.connector.Embeddings1DConnector`.
"""
import json
from typing import List, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def _rms_norm(x: jnp.ndarray, eps: float, scale: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x32 = x.astype(jnp.float32)
    mean_sq = jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps
    normed = x32 * jax.lax.rsqrt(mean_sq)
    if scale is not None:
        normed = normed * scale.astype(jnp.float32)
    return normed.astype(x.dtype)


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """`x`: (B, S, H, D); `cos`/`sin`: (B, S, D), broadcast over heads."""
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return x * cos + _rotate_half(x) * sin


def _default_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    return 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))


def _proportional_inv_freq(head_dim: int, theta: float, partial_rotary_factor: float) -> np.ndarray:
    """See module docstring: zeros the trailing inverse-frequency entries
    rather than slicing the head dim -- matches
    `transformers.modeling_rope_utils._compute_proportional_rope_parameters`.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq_rotated = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim))
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        return np.concatenate([inv_freq_rotated, np.zeros(nope_angles, dtype=np.float64)])
    return inv_freq_rotated


def _rope_cos_sin(position_ids: jnp.ndarray, inv_freq: np.ndarray, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """`position_ids`: (B, S) int/float. Returns (cos, sin), each (B, S, head_dim)."""
    inv_freq_j = jnp.asarray(inv_freq, dtype=jnp.float32)
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq_j[None, None, :]  # (B, S, head_dim//2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, S, head_dim)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def build_attention_bias(
    attention_mask: jnp.ndarray, sliding_window: Optional[int], dtype: jnp.dtype,
) -> jnp.ndarray:
    """Additive `(B, 1, S, S)` causal (+ optional sliding-window) bias:
    `0.0` where query `i` may attend to key `j`, else the dtype's most
    negative value. `attention_mask` is `(B, S)`, `1` = real token, `0` =
    padding (padding tokens are never attended to, regardless of position).
    """
    b, s = attention_mask.shape
    i = jnp.arange(s)[:, None]
    j = jnp.arange(s)[None, :]
    allowed = j <= i
    if sliding_window is not None:
        allowed = allowed & (j > i - sliding_window)
    allowed = allowed[None, None, :, :] & (attention_mask.astype(bool)[:, None, None, :])
    return jnp.where(allowed, jnp.zeros((), dtype=dtype), jnp.finfo(dtype).min)


class Gemma4MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
        gate = jax.nn.gelu(gate, approximate=True)
        up = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)
        return nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(gate * up)


class Gemma4Attention(nn.Module):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    k_eq_v: bool
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, attention_bias: jnp.ndarray,
    ) -> jnp.ndarray:
        b, s, _ = x.shape
        h, kvh, d = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        n_rep = h // kvh

        q = nn.Dense(h * d, use_bias=False, name="q_proj")(x).reshape(b, s, h, d)
        q_scale = self.param("q_norm_scale", nn.initializers.ones, (d,))
        q = _rms_norm(q, self.eps, q_scale)
        q = _apply_rotary(q, cos, sin)

        k_raw = nn.Dense(kvh * d, use_bias=False, name="k_proj")(x).reshape(b, s, kvh, d)
        if self.k_eq_v:
            v_raw = k_raw
        else:
            v_raw = nn.Dense(kvh * d, use_bias=False, name="v_proj")(x).reshape(b, s, kvh, d)

        k_scale = self.param("k_norm_scale", nn.initializers.ones, (d,))
        k = _rms_norm(k_raw, self.eps, k_scale)
        k = _apply_rotary(k, cos, sin)
        v = _rms_norm(v_raw, self.eps, None)  # v_norm: with_scale=False always.

        q = q.transpose(0, 2, 1, 3)  # (B, H, S, D)
        k = jnp.repeat(k.transpose(0, 2, 1, 3), n_rep, axis=1)  # (B, H, S, D)
        v = jnp.repeat(v.transpose(0, 2, 1, 3), n_rep, axis=1)

        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32)  # scaling == 1.0, see module docstring.
        logits = logits + attention_bias.astype(jnp.float32)
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, h * d)
        return nn.Dense(self.hidden_size, use_bias=False, name="o_proj")(out)


class Gemma4DecoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    k_eq_v: bool
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, attention_bias: jnp.ndarray,
    ) -> jnp.ndarray:
        input_norm_scale = self.param("input_layernorm_scale", nn.initializers.ones, (self.hidden_size,))
        residual = x
        h = _rms_norm(x, self.eps, input_norm_scale)
        h = Gemma4Attention(
            self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.head_dim,
            k_eq_v=self.k_eq_v, eps=self.eps, compute_dtype=self.compute_dtype, name="self_attn")(
                h, cos, sin, attention_bias)
        post_attn_scale = self.param("post_attention_layernorm_scale", nn.initializers.ones, (self.hidden_size,))
        h = _rms_norm(h, self.eps, post_attn_scale)
        x = residual + h

        residual = x
        pre_ff_scale = self.param("pre_feedforward_layernorm_scale", nn.initializers.ones, (self.hidden_size,))
        h = _rms_norm(x, self.eps, pre_ff_scale)
        h = Gemma4MLP(self.hidden_size, self.intermediate_size, compute_dtype=self.compute_dtype, name="mlp")(h)
        post_ff_scale = self.param("post_feedforward_layernorm_scale", nn.initializers.ones, (self.hidden_size,))
        h = _rms_norm(h, self.eps, post_ff_scale)
        x = residual + h

        layer_scalar = self.param("layer_scalar", nn.initializers.ones, (1,))
        return x * layer_scalar.astype(x.dtype)


class Gemma4TextModel(nn.Module):
    """`Gemma4UnifiedTextModel` (text tower only). Config-driven -- pass
    `vidax.models.ltx2_5.configs.GEMMA4_TEXT_CONFIG` (read from the real
    checkpoint's embedded `gemma_config`) as constructor kwargs.

    Returns every layer's hidden state (49 = embedding + 48 decoder layers),
    for `extract_video_features` -- not just the final one.
    """
    vocab_size: int = 262144
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int = 1
    layer_types: Sequence[str] = ()  # "sliding_attention" | "full_attention", length num_hidden_layers.
    sliding_window: int = 1024
    rope_theta_sliding: float = 10000.0
    rope_theta_full: float = 1000000.0
    partial_rotary_factor_full: float = 0.25
    eps: float = 1e-6
    compute_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.embed_tokens = nn.Embed(self.vocab_size, self.hidden_size, name="embed_tokens")
        self.layers = [
            Gemma4DecoderLayer(
                hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=(
                    self.num_global_key_value_heads if self.layer_types[i] == "full_attention"
                    else self.num_key_value_heads),
                head_dim=self.head_dim if self.layer_types[i] == "sliding_attention" else self.global_head_dim,
                k_eq_v=(self.layer_types[i] == "full_attention"),
                eps=self.eps, compute_dtype=self.compute_dtype, name=f"layers_{i}")
            for i in range(self.num_hidden_layers)
        ]
        self.final_norm_scale = self.param("norm_scale", nn.initializers.ones, (self.hidden_size,))

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Args:
            input_ids: (B, S) int32 token ids.
            attention_mask: (B, S) 1 = real token, 0 = padding.

        Returns:
            List of `num_hidden_layers + 1` tensors, each (B, S, hidden_size)
            -- the scaled embedding output, then each decoder layer's output.
        """
        embed_scale = jnp.asarray(self.hidden_size ** 0.5, dtype=jnp.float32).astype(self.compute_dtype)
        x = self.embed_tokens(input_ids).astype(self.compute_dtype) * embed_scale
        hidden_states = [x]

        position_ids = jnp.broadcast_to(jnp.arange(input_ids.shape[1])[None, :], input_ids.shape)
        cos_sliding, sin_sliding = _rope_cos_sin(
            position_ids, _default_inv_freq(self.head_dim, self.rope_theta_sliding), self.compute_dtype)
        cos_full, sin_full = _rope_cos_sin(
            position_ids,
            _proportional_inv_freq(self.global_head_dim, self.rope_theta_full, self.partial_rotary_factor_full),
            self.compute_dtype)

        bias_sliding = build_attention_bias(attention_mask, self.sliding_window, jnp.float32)
        bias_full = build_attention_bias(attention_mask, None, jnp.float32)

        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == "sliding_attention":
                x = layer(x, cos_sliding, sin_sliding, bias_sliding)
            else:
                x = layer(x, cos_full, sin_full, bias_full)
            hidden_states.append(x)

        hidden_states[-1] = _rms_norm(hidden_states[-1], self.eps, self.final_norm_scale)
        return hidden_states


def extract_video_features(
    hidden_states: Sequence[jnp.ndarray],
    attention_mask: jnp.ndarray,
    video_aggregate_kernel: jnp.ndarray,
    video_aggregate_bias: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """`FeatureExtractorV2` (22B / `PER_TOKEN_RMS`): stack every layer's
    hidden state, per-token-per-layer RMSNorm, rescale, concat across
    layers, then a single Linear projection (`text_embedding_projection.
    video_aggregate_embed` in the Gemma checkpoint) to `cross_attention_dim`
    -- see `ltx_core.text_encoders.gemma.feature_extractor.
    {norm_and_concat_per_token_rms,FeatureExtractorV2}`.

    Args:
        hidden_states: `num_hidden_layers + 1` tensors, each (B, S, D)
            (`Gemma4TextModel`'s output).
        attention_mask: (B, S) 1 = real token, 0 = padding.
        video_aggregate_kernel: (D * num_layers, out_dim) -- already
            transposed from PyTorch's (out_dim, in_dim) `Linear.weight` by
            the translator.
        video_aggregate_bias: (out_dim,).

    Returns:
        (B, S, out_dim) -- ready for
        `vidax.models.ltx2_5.connector.Embeddings1DConnector`.
    """
    stacked = jnp.stack(hidden_states, axis=-1)  # (B, S, D, L)
    b, s, d, l = stacked.shape
    variance = jnp.mean(jnp.square(stacked.astype(jnp.float32)), axis=2, keepdims=True)  # (B, S, 1, L)
    normed = stacked.astype(jnp.float32) * jax.lax.rsqrt(variance + eps)
    normed = normed.reshape(b, s, d * l).astype(stacked.dtype)
    mask3d = attention_mask.astype(bool)[..., None]
    normed = jnp.where(mask3d, normed, jnp.zeros_like(normed))

    out_dim = video_aggregate_kernel.shape[-1]
    scale = jnp.sqrt(out_dim / (d * l)).astype(normed.dtype)
    scaled = normed * scale
    return jnp.einsum("bsi,io->bso", scaled, video_aggregate_kernel) + video_aggregate_bias


# `tokenizer_json`/`hf_asset__tokenizer_config.json` -- a full HuggingFace
# `tokenizers.Tokenizer` (32MB vocab+merges) and its small sidecar config,
# embedded as raw-byte `uint8` tensors directly inside the Gemma
# checkpoint (`gemma_assets.py`'s `GemmaAssets.from_single_file`/
# `build_gemma_hf_tokenizer` in the reference) -- not a separate
# tokenizer directory the way LTX-Video's PixArt T5 tokenizer ships.
_TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
_TOKENIZER_CONFIG_ASSET_KEY = "hf_asset__tokenizer_config.json"
_TOKENIZER_MAX_LENGTH = 1024
# Keys the reference explicitly drops before forwarding `tokenizer_config
# .json`'s contents as `PreTrainedTokenizerFast` kwargs (see
# `gemma_assets._TOKENIZER_CONFIG_SKIP`) -- mostly HF-internal bookkeeping
# that either doesn't apply here or is set explicitly below instead.
_TOKENIZER_CONFIG_SKIP = frozenset({
    "tokenizer_class", "auto_map", "model_max_length", "backend",
    "is_local", "local_files_only", "processor_class", "added_tokens_decoder",
})


class Gemma4Tokenizer:
    """Extracts and wraps the Gemma checkpoint's embedded HF tokenizer.
    Mirrors `vidax.models.ltx_video.t5.PixArtT5Tokenizer`'s shape/API
    (fixed-length, zero-padded `(ids, attention_mask)` arrays) -- a
    separate class, not a shared base, per this port's file-independence
    design (same reasoning as that class's own docstring).
    """

    def __init__(self, checkpoint_path: str, seq_len: int = _TOKENIZER_MAX_LENGTH):
        import safetensors
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast  # Optional dependency; install the `text` extra.

        with safetensors.safe_open(checkpoint_path, framework="numpy") as f:
            keys = set(f.keys())
            if _TOKENIZER_JSON_TENSOR_KEY not in keys:
                raise ValueError(f"{checkpoint_path} has no {_TOKENIZER_JSON_TENSOR_KEY!r} tensor.")
            tokenizer_json_bytes = f.get_tensor(_TOKENIZER_JSON_TENSOR_KEY).astype(np.uint8).tobytes()
            tokenizer_cfg = {}
            if _TOKENIZER_CONFIG_ASSET_KEY in keys:
                cfg_bytes = f.get_tensor(_TOKENIZER_CONFIG_ASSET_KEY).astype(np.uint8).tobytes()
                tokenizer_cfg = json.loads(cfg_bytes.decode("utf-8"))

        kwargs = {k: v for k, v in tokenizer_cfg.items() if k not in _TOKENIZER_CONFIG_SKIP}
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(tokenizer_json_bytes),
            model_max_length=seq_len, **kwargs)
        self.seq_len = seq_len

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts, return_tensors="np", padding="max_length", truncation=True, max_length=self.seq_len)
        return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)
