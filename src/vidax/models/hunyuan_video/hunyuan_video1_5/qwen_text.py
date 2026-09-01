"""HunyuanVideo-1.5's MLLM text tower: Qwen2.5-VL-7B-Instruct, text-only
(no vision tower, no LM head) -- port of the "llm" branch of
``hyvideo/models/text_encoders/__init__.py``'s ``TextEncoder``.

This is architecturally the **same** model Cosmos-Predict2.5-2B's Reason1
text encoder uses (a Qwen2.5-VL-7B-Instruct text-only decoder tower), so
the actual transformer is reused unmodified from
``vidax.models.cosmos2_5.reason1.Qwen2TextModel`` rather than re-ported --
only the prompt-templating/crop_start/hidden-state-selection glue below is
HunyuanVideo-1.5-specific (different from Cosmos's own usage of the same
tower: HunyuanVideo does not mean/std-normalize-and-concat all 28 layers,
it selects a single intermediate layer and does not apply a final norm).

Reference call site (``hyvideo/pipelines/hunyuan_video_pipeline.py:1592-1597``):
``TextEncoder(text_encoder_type="llm", max_length=1000,
prompt_template=PROMPT_TEMPLATE['li-dit-encode-image-json'],
prompt_template_video=PROMPT_TEMPLATE['li-dit-encode-video-json'],
hidden_state_skip_layer=2, apply_final_norm=False)``.
"""
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp

from vidax.models.cosmos2_5.reason1 import Qwen2TextModel

# `hidden_state_skip_layer=2`: HF `output_hidden_states=True` returns a
# tuple of length num_layers+1 (index 0 = embedding output, index i = after
# decoder layer i, with the *last* entry additionally having the final norm
# applied -- see `Qwen2TextModel`'s own docstring, same convention). The
# reference indexes `hidden_states[-(skip_layer+1)]` = index -3, i.e. the
# raw (pre-final-norm) output of the third-from-last entry.
HIDDEN_STATE_SKIP_LAYER = 2

# System prompts for the two `PROMPT_TEMPLATE` entries HunyuanVideo-1.5
# ships (`li-dit-encode-image-json` for I2V's conditioning-image prompt
# path, `li-dit-encode-video-json` for T2V/the main video prompt) --
# transcribed verbatim from `hyvideo/models/text_encoders/__init__.py`.
_SYSTEM_PROMPT_IMAGE = (
    "You are a helpful assistant. Describe the image by detailing the following aspects:         "
    "1. The main content and theme of the image.         "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
    "3. The background environment, light, style and atmosphere."
)
_SYSTEM_PROMPT_VIDEO = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:         "
    "1. The main content and theme of the video.         "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
    "4. background environment, light, style and atmosphere.         "
    "5. camera angles, movements, and transitions used in the video."
)

_USER_MARKER = "<|im_start|>user\n"


class HunyuanVideoMLLMTokenizer:
    """Applies the reference's fixed system-prompt chat template, computes
    ``crop_start`` (position right after the ``<|im_start|>user\\n`` marker
    -- the reference auto-detects this once and reuses it; here it's
    computed fresh per call, cheap relative to the 7B forward pass), and
    tokenizes to a fixed length (``max_length + crop_start``, matching
    ``text2tokens``'s second-pass padding, so the crop always leaves
    exactly ``max_length`` tokens).

    Args:
        tokenizer_path: Path/repo id for Qwen2.5-VL-7B-Instruct's tokenizer.
        max_length: Prompt-content length *after* cropping the
            system/user-marker tokens (``self.text_len`` in the reference
            pipeline, 1000 for HunyuanVideo-1.5).
        data_type: "image" (I2V conditioning-image caption path) or
            "video" (T2V / the main prompt) -- selects the system prompt.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 1000, data_type: str = "video"):
        from transformers import AutoTokenizer  # Optional dependency; install the `text` extra.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.system_prompt = _SYSTEM_PROMPT_IMAGE if data_type == "image" else _SYSTEM_PROMPT_VIDEO
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        marker_ids = self.tokenizer(_USER_MARKER, add_special_tokens=False)["input_ids"]
        self._marker_ids = marker_ids

    def _crop_start(self, ids: List[int]) -> int:
        n = len(self._marker_ids)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == self._marker_ids:
                return i + n
        return 0

    def __call__(self, texts: Union[str, List[str]]) -> Tuple["jnp.ndarray", "jnp.ndarray", int]:
        """Returns (input_ids, attention_mask, crop_start) -- ``input_ids``/
        ``attention_mask`` shaped ``(B, crop_start + max_length)`` (crop_start
        is computed from the first prompt in the batch and applied to all,
        matching the reference's per-`TextEncoder`-instance single
        crop_start cache).
        """
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]

        conversations = [
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": t}]
            for t in texts
        ]
        # First pass (unpadded) just to compute crop_start from prompt 0 --
        # matches the reference's own single-instance crop_start cache
        # (computed once, reused for the whole batch/session).
        probe_ids = self.tokenizer.apply_chat_template(
            conversations[0], add_generation_prompt=True, tokenize=True)
        if hasattr(probe_ids, "get") or isinstance(probe_ids, dict):
            probe_ids = probe_ids["input_ids"]
        crop_start = self._crop_start(list(probe_ids))

        total_len = crop_start + self.max_length
        ids_batch, mask_batch = [], []
        for conv in conversations:
            ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "get") or isinstance(ids, dict):
                ids = ids["input_ids"]
            ids = list(ids)
            mask = [1] * len(ids)
            if len(ids) < total_len:
                pad = total_len - len(ids)
                ids = ids + [self.pad_id] * pad
                mask = mask + [0] * pad
            else:
                ids = ids[:total_len]
                mask = mask[:total_len]
            ids_batch.append(ids)
            mask_batch.append(mask)

        return (np.asarray(ids_batch, dtype=np.int32),
                np.asarray(mask_batch, dtype=np.int32),
                crop_start)


def extract_hunyuan_mllm_embeddings(
    params, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, crop_start: int,
    model: Optional[Qwen2TextModel] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Runs the Qwen2.5-VL-7B text tower and reproduces
    ``TextEncoder.encode``'s skip-layer + crop pipeline.

    Args:
        params: Flax param tree for `Qwen2TextModel` (e.g. from
            `map_reason1_text_encoder_keys` -- same checkpoint format as
            Cosmos-Predict2.5's Reason1, both being plain Qwen2.5-VL-7B-
            Instruct text towers).
        input_ids, attention_mask: From `HunyuanVideoMLLMTokenizer`.
        crop_start: From `HunyuanVideoMLLMTokenizer` (the same value used
            to build `input_ids`'s padding length).
        model: Optional pre-constructed `Qwen2TextModel`.

    Returns:
        (text_states, text_mask): (B, max_length, 3584) hidden states and
        (B, max_length) mask, both already cropped -- feed straight into
        `HunyuanVideo15DiT.__call__`'s `text_states`/`encoder_attention_mask`.
    """
    if model is None:
        model = Qwen2TextModel()
    hidden_states = model.apply({"params": params["params"]}, input_ids)

    # hidden_states[-(skip+1)] with skip=2 -> index -3, matching
    # `TextEncoder.encode`'s `outputs.hidden_states[-(hidden_state_skip_layer
    # + 1)]`. `apply_final_norm=False` in the reference, so no extra norm.
    last_hidden_state = hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]

    if crop_start > 0:
        last_hidden_state = last_hidden_state[:, crop_start:]
        attention_mask = attention_mask[:, crop_start:]

    return last_hidden_state, attention_mask
