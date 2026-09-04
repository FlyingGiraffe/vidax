"""Benchmark harness for LTX-Video 0.9.8 (2B-distilled, 13B-dev,
13B-distilled) -- t2v, i2v. Imports and calls the real
`examples/generate_ltx_video.py`'s `main(args)` (no separate
reimplementation of the generation loop -- see `benchmarks/common.py`'s
module docstring).

Usage:
    python benchmarks/run_ltx_video.py --model_size 2B-distilled --task t2v
    python benchmarks/run_ltx_video.py --model_size 13B-dev --task t2v
    VIDAX_CHECKPOINT_DIR=/path/to/checkpoints python benchmarks/run_ltx_video.py --model_size 13B-distilled --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_ltx_video  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODEL_SIZES = ("2B-distilled", "13B-dev", "13B-distilled")
TASKS = ("t2v", "i2v")

# Each variant's own released checkpoint repo/file name (see
# docs/models/ltx_video.md).
_CHECKPOINT_REPOS = {
    "2B-distilled": ("LTX-Video-0.9.8-2B-distilled", "ltxv-2b-0.9.8-distilled.safetensors"),
    "13B-dev": ("LTX-Video-0.9.8-13B-dev", "ltxv-13b-0.9.8-dev.safetensors"),
    "13B-distilled": ("LTX-Video-0.9.8-13B-distilled", "ltxv-13b-0.9.8-distilled.safetensors"),
}
# Distilled checkpoints are trained to run well in very few steps with no
# CFG (guidance_scale=1.0 -- see refs/LTX-Video-main's own distilled
# configs' `guidance_scale: 1`); "dev" needs real classifier-free guidance
# and more steps, matching the reference's own multi-scale first_pass
# step count (this script runs single-scale/single-pass -- see
# generate_ltx_video.py's module docstring for what that simplifies away).
_IS_DISTILLED = {"2B-distilled": True, "13B-dev": False, "13B-distilled": True}
# The 13B checkpoints' bf16 weights alone (~26GB) don't fit replicated on a
# single TPU v4 chip at all -- see generate_ltx_video.py's
# --tensor_parallel_size. 2B fits *weight-wise* replicated on one chip, but
# not the reference's full 704x1216x121-token activation footprint on top
# of it (confirmed OOM at --tensor_parallel_size 1) -- tp=4 shards both and
# comfortably fits all three variants at the same reference resolution, so
# every row below is directly comparable.
_DEFAULT_TP_SIZE = {"2B-distilled": 4, "13B-dev": 4, "13B-distilled": 4}


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    repo_name, checkpoint_file = _CHECKPOINT_REPOS[args.model_size]
    checkpoint_path = os.path.join(checkpoint_dir, repo_name, checkpoint_file)
    t5_checkpoint_path = os.path.join(
        checkpoint_dir, "PixArt-XL-2-1024-MS", "text_encoder", "model.safetensors.index.json")

    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP_SIZE[args.model_size]
    distilled = _IS_DISTILLED[args.model_size]

    size_slug = args.model_size.lower().replace("-", "_")
    ns = argparse.Namespace(
        checkpoint_path=checkpoint_path,
        t5_checkpoint_path=t5_checkpoint_path,
        tokenizer_path=None,
        prompt=[common.STANDARD_T2V_PROMPT],
        negative_prompt=generate_ltx_video.DEFAULT_NEGATIVE_PROMPT,
        image_path=None,
        conditioning_strength=1.0,
        guidance_scale=1.0 if distilled else 3.0,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        tensor_parallel_size=tp_size,
        seed=0,
        num_steps=8 if distilled else 30,
        sampler="LinearQuadratic",
        shift=None,
        text_max_tokens=256,
        decode_timestep=0.05,
        decode_noise_scale=None,
        # The reference's own InferenceConfig default (height=704,
        # width=1216, num_frames=121) -- confirmed to fit this repo's v4-8
        # test machine at --tensor_parallel_size as set above, for every
        # variant (see docs/benchmarking.md).
        height=704,
        width=1216,
        num_frames=121,
        fps=24,
        output_path=common.output_path("ltx_video", "0.9.8", size_slug, args.task),
    )
    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = [common.STANDARD_I2V_PROMPT]
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="2B-distilled", choices=MODEL_SIZES)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    run_args = build_args(args)
    size_slug = args.model_size.lower().replace("-", "_")
    common.run_benchmark(
        model="ltx_video", version="0.9.8", size=size_slug, task=args.task,
        main_fn=generate_ltx_video.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
