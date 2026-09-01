# HunyuanVideo (1.0) Debugging Lessons

Findings from porting HunyuanVideo (1.0)'s DiT/VAE/text-encoders to
JAX/Flax. See
[`docs/models/hunyuan_video_1_0.md`](../models/hunyuan_video_1_0.md) for the
full port and its architecture.

## A padded-tokenizer test can look like a real numeric bug when it's actually a padding-side mismatch

An early `LlamaTextModel` check (`tokenizer(text, padding="max_length",
max_length=32)`, *without* explicitly setting `padding_side="right"`)
showed a real-looking divergence (correlation 0.978, max abs diff 14)
against the real PyTorch reference. Re-running the identical model code on
an **unpadded** prompt gave a bit-exact match (correlation 1.0 to 13
decimal places) — proving the model code itself was already correct, and
the divergence was from the tokenizer silently defaulting to its saved
`padding_side` (plausibly `"left"` for a chat/generation-oriented
checkpoint like llava-llama-3), which breaks this port's "causal mask
alone is sufficient under right-padding" argument (a *left*-padded
sequence lets every valid, later token's causal-visible span include the
leading padding as keys, which does change its output). `LlamaPromptTokenizer`
in `examples/generate_hunyuan_video_1_0.py` explicitly passes
`padding_side="right"` at `AutoTokenizer.from_pretrained(...)` for exactly
this reason — general lesson for any decoder-only text encoder reused
across model families: never trust a checkpoint's own saved tokenizer
default padding side without checking it explicitly matches what the
masking implementation assumes.
