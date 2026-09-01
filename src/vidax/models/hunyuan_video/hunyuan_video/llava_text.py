"""HunyuanVideo-I2V's ``"llm-i2v"`` text/vision encoder pipeline --
tokenizer expansion, image preprocessing (two separate pipelines, see
below), and the embedding-extraction glue that turns a raw prompt +
reference image into the DiT's ``text_states``/``text_mask``.

See ``llava_vision.py``'s module docstring for the vision tower/projector
architecture and ``llama_text.py``'s ``LlamaTextModel.__call__``'s
``image_embeds``/``image_start``/``image_end`` params for the splice
mechanism this module drives.

**Real `transformers` version mismatch** (see the plan file's "HunyuanVideo
1.0 I2V" progress log): this checkpoint's own config.json was built
against `transformers==4.40.1`; the installed verify env has a much newer
version whose `LlavaModel.forward` requires the image placeholder to
already be pre-expanded (raises otherwise) -- the *old* 4.40.1-era
`_merge_input_ids_with_image_features` dynamically expanded a single
`<image>` token into `image_emb_len` (576) positions and grew the
sequence, which is what this checkpoint's own hardcoded
`image_emb_start=5`/`image_emb_end=581` constants (`constants.py`) are
only consistent with. `LlavaPromptTokenizer` below implements that
expansion explicitly (insert 575 more `image_token_id` copies at the raw
`<image>` position) rather than relying on any installed `transformers`
version's own (possibly incompatible) merge behavior.

**Two separate image preprocessing pipelines for the same reference
photo** (confirmed by reading `inference.py` + `TextEncoder.encode`'s
`i2v_mode` branch directly, not assumed):
  1. ``preprocess_image_for_vae``: resized/center-cropped to the video's
     own target resolution bucket (`get_closest_ratio`/
     `generate_crop_size_list`, ported from `utils/data_utils.py`),
     normalized to [-1, 1] -- feeds `HunyuanVideoVAE.encode` for
     `token_replace`'s `img_latents`.
  2. ``preprocess_image_for_llava``: resized/center-cropped to CLIP's own
     fixed 336x336 (aspect-ratio-preserving resize to shortest-edge 336,
     then center-crop -- standard `CLIPImageProcessor` behavior), CLIP
     dataset mean/std normalized (`wan.wan2_1.clip_vision.CLIP_MEAN`/
     `CLIP_STD`, reused directly -- same constants) -- feeds
     `llava_vision.ClipVisionModel`.
Do not conflate these -- they use different target sizes and different
normalization conventions.
"""
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vidax.models.hunyuan_video.hunyuan_video.llama_text import LlamaTextModel
from vidax.models.hunyuan_video.hunyuan_video.llava_vision import (
    ClipVisionModel, LlavaMultiModalProjector, get_llava_image_features,
)
from vidax.models.wan.wan2_1.clip_vision import CLIP_MEAN, CLIP_STD

IMAGE_TOKEN_INDEX = 128257  # xtuner/llava-llama-3-8b-v1_1-transformers's config.json.
IMAGE_EMB_LEN = 576  # (336 // 14) ** 2 -- constants.py's `image_emb_len`.
DOUBLE_RETURN_TOKEN_ID = 271  # "\n\n" -- see this module's docstring.

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
CROP_START = 103  # constants.py's "dit-llm-encode-video-i2v" crop_start.
IMAGE_EMB_START = 5
LLM_TOKENIZE_MAX_LENGTH = 256 + CROP_START  # matches T2V's own text_len(256)+crop_start convention.


# --- Resolution bucketing (ported from hyvideo/utils/data_utils.py -- plain
# Python/numpy, no model code) ---

def align_to(value: float, alignment: int) -> int:
    return int(math.ceil(value / alignment) * alignment)


def generate_crop_size_list(base_size: int = 256, patch_size: int = 32, max_ratio: float = 4.0) -> List[Tuple[int, int]]:
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def get_closest_ratio(height: float, width: float, ratios: np.ndarray, buckets: List[Tuple[int, int]]):
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio
    if aspect_ratio >= 1:
        indices = [(i, x) for i, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(i, x) for i, x in enumerate(diff_ratios) if x > 0]
    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    return buckets[closest_ratio_id], ratios[closest_ratio_id]


_BUCKET_HW_BASE_SIZE = {"720p": 960, "540p": 720, "360p": 480}


def compute_i2v_closest_size(image: Image.Image, resolution: str = "720p") -> Tuple[int, int]:
    """Returns (width, height) -- the bucketed target resolution
    (`generate_crop_size_list`'s own (w, h) convention) closest to the
    reference image's own aspect ratio, matching `inference.py`'s
    `closest_size` exactly."""
    base = _BUCKET_HW_BASE_SIZE[resolution]
    crop_size_list = generate_crop_size_list(base, 32)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for w, h in crop_size_list])
    origin_w, origin_h = image.size
    closest_size, _ = get_closest_ratio(origin_h, origin_w, aspect_ratios, crop_size_list)
    return closest_size  # (w, h)


# --- Image preprocessing (two separate pipelines, see module docstring) ---

def preprocess_image_for_vae(image: Image.Image, closest_size: Tuple[int, int]) -> jnp.ndarray:
    """`Resize(min(closest_size)) -> CenterCrop(closest_size) ->
    Normalize([0.5],[0.5])` -- matches `inference.py`'s `ref_image_transform`
    exactly. Returns (1, 1, H, W, 3) float32 in [-1, 1] (channel-last,
    leading frame axis of 1 -- matches `HunyuanVideoVAE.encode`'s
    ``(B, T, H, W, C)`` convention)."""
    w, h = closest_size
    resize_short = min(w, h)
    ow, oh = image.size
    scale = resize_short / min(ow, oh)
    new_w, new_h = round(ow * scale), round(oh * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - w) // 2, (new_h - h) // 2
    image = image.crop((left, top, left + w, top + h))
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return jnp.asarray(x)[None, None]


def preprocess_image_for_llava(image: Image.Image, image_size: int = 336) -> jnp.ndarray:
    """Standard `CLIPImageProcessor` preprocessing: aspect-ratio-preserving
    resize to `shortest_edge=image_size`, center-crop to
    `(image_size, image_size)`, rescale by 1/255, CLIP dataset mean/std
    normalize. Returns (1, image_size, image_size, 3) float32."""
    ow, oh = image.size
    scale = image_size / min(ow, oh)
    new_w, new_h = round(ow * scale), round(oh * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - image_size) // 2, (new_h - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))
    x = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    x = (x - mean) / std
    return jnp.asarray(x)[None]


# --- Tokenizer expansion ---

class LlavaPromptTokenizer:
    """Wraps a real `AutoTokenizer` (loaded from the full LLaVA checkpoint
    dir): formats the video I2V chat template, tokenizes, then expands the
    single `<image>` placeholder into `IMAGE_EMB_LEN` (576) repeated
    `IMAGE_TOKEN_INDEX` positions -- see this module's docstring for why
    this expansion is done explicitly rather than relying on any
    installed `transformers` version's own (possibly incompatible) merge
    behavior. Always right-padded (see `llama_text.py`'s established
    "causal mask alone is sufficient under right-padding" argument)."""

    def __init__(self, tokenizer_dir: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side="right")

    def __call__(self, prompts: List[str], max_length: int = LLM_TOKENIZE_MAX_LENGTH):
        """Returns ``(expanded_ids, image_start, image_end, raw_ids,
        raw_attention_mask)``.

        ``expanded_ids`` (and the image span) are in *post-expansion*
        coordinates -- feed straight to ``LlamaTextModel``. ``raw_ids``/
        ``raw_attention_mask`` are the *pre-expansion* tokenizer output
        (single `<image>` token, length `max_length`) -- kept separately
        because the reference's own index math
        (``extract_hunyuan_llava_embeddings``) operates on the attention
        mask in *pre-expansion* coordinates while operating on the model's
        hidden states in *post-expansion* coordinates -- two deliberately
        different (but numerically consistent) coordinate spaces, not a
        simplification opportunity; see this module's docstring.
        """
        texts = [PROMPT_TEMPLATE_ENCODE_VIDEO_I2V.format(p) for p in prompts]
        enc = self.tokenizer(texts, return_tensors="np", padding="max_length",
                              max_length=max_length, truncation=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]

        expanded_ids = []
        image_pos = None
        for b in range(ids.shape[0]):
            row = ids[b]
            pos = int(np.where(row == self.tokenizer.encode("<image>", add_special_tokens=False)[0])[0][0])
            if image_pos is None:
                image_pos = pos
            assert pos == image_pos, "image placeholder position must be identical across a batch"
            new_row = np.concatenate([row[:pos], np.full(IMAGE_EMB_LEN, IMAGE_TOKEN_INDEX, dtype=row.dtype), row[pos + 1:]])
            expanded_ids.append(new_row)
        return np.stack(expanded_ids), image_pos, image_pos + IMAGE_EMB_LEN, ids, mask


def extract_hunyuan_llava_embeddings(
    llama_params, clip_params, projector_params,
    expanded_ids: np.ndarray, raw_ids: np.ndarray, raw_attention_mask: np.ndarray,
    image_start: int, image_end: int, pixel_values_clip: jnp.ndarray,
    llama_model: LlamaTextModel, clip_model: ClipVisionModel, projector: LlavaMultiModalProjector,
    hidden_state_skip_layer: int = 2, image_embed_interleave: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Full I2V embedding extraction, mirroring `TextEncoder.encode`'s
    `i2v_mode` branch for `data_type="video"` exactly (see this module's
    docstring and the plan file's architecture notes for the index math's
    provenance):

    1. Project the reference image through the CLIP tower + multimodal
       projector (`get_llava_image_features`, `vision_feature_layer=-2`).
    2. Run `expanded_ids` (post-expansion, see `LlavaPromptTokenizer`)
       through the Llama decoder with those projected features spliced in
       at `[image_start:image_end)`.
    3. Select `hidden_states[-(hidden_state_skip_layer+1)]` (same
       convention as T2V).
    4. Split that single layer's hidden states into an "image" region and
       a "text" region, excising the trailing assistant-header boilerplate
       (found via the last "\\n\\n" token, `DOUBLE_RETURN_TOKEN_ID`) --
       **the reference deliberately mixes two coordinate spaces here**:
       the hidden-state slice bounds are in *post-expansion* coordinates
       (found via `raw_ids`' pre-expansion position + the constant
       `IMAGE_EMB_LEN - 1` shift, matching every position after the image
       block moving right by that much), while the attention-mask slice
       bounds are the *raw pre-expansion* positions applied directly to
       `raw_attention_mask` (never expanded) -- both conventions yield the
       same real span length by construction (see the plan file), so this
       is not a bug to "simplify" away.

    Returns (text_states, text_mask), each batch-first, ready for
    `HunyuanVideoDiT`'s `text_states`/`encoder_attention_mask` inputs.
    """
    image_features = get_llava_image_features(clip_params, clip_model, projector_params, projector, pixel_values_clip)

    expanded_ids_j = jnp.asarray(expanded_ids)
    hidden_states = llama_model.apply(
        llama_params, expanded_ids_j, image_embeds=image_features, image_start=image_start, image_end=image_end)
    last_hidden_state = hidden_states[-(hidden_state_skip_layer + 1)]

    b = raw_ids.shape[0]
    text_states, text_masks = [], []
    for i in range(b):
        double_return_positions = np.where(raw_ids[i] == DOUBLE_RETURN_TOKEN_ID)[0]
        last_double_return = int(double_return_positions[-1])  # pre-expansion coordinates.

        text_crop_start = CROP_START - 1 + IMAGE_EMB_LEN
        assistant_crop_start = last_double_return - 1 + IMAGE_EMB_LEN - 4
        assistant_crop_end = last_double_return - 1 + IMAGE_EMB_LEN
        attn_assistant_crop_start = last_double_return - 4
        attn_assistant_crop_end = last_double_return

        text_hs = jnp.concatenate([
            last_hidden_state[i, text_crop_start:assistant_crop_start],
            last_hidden_state[i, assistant_crop_end:],
        ], axis=0)
        text_mask_row = jnp.concatenate([
            jnp.asarray(raw_attention_mask[i, CROP_START:attn_assistant_crop_start]),
            jnp.asarray(raw_attention_mask[i, attn_assistant_crop_end:]),
        ], axis=0)

        image_hs = last_hidden_state[i, image_start:image_end]
        image_mask_row = jnp.ones((image_hs.shape[0],), dtype=text_mask_row.dtype)
        if 0 < image_embed_interleave < 6:
            image_hs = image_hs[::image_embed_interleave]
            image_mask_row = image_mask_row[::image_embed_interleave]

        text_states.append(jnp.concatenate([image_hs, text_hs], axis=0))
        text_masks.append(jnp.concatenate([image_mask_row, text_mask_row], axis=0))

    return jnp.stack(text_states), jnp.stack(text_masks)
