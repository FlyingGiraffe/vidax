"""Benchmark harness for CogVideoX (2b / 5b / 5b-I2V / 1.5-5B / 1.5-5B-I2V),
t2v + i2v. Imports and calls the real `examples/generate_cogvideox.py`'s
`main(args)` (no reimplementation of the generation loop -- see
`benchmarks/common.py`'s module docstring).

Usage:
    python benchmarks/run_cogvideox.py --model_size 5b --task t2v
    VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints python benchmarks/run_cogvideox.py --model_size 1.5-5b-i2v --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_cogvideox  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODEL_SIZES = ("2b", "5b", "5b-i2v", "1.5-5b", "1.5-5b-i2v")
TASKS = ("t2v", "i2v")

# Each variant's downloaded diffusers repo directory name.
_REPO = {
    "2b": "CogVideoX-2b",
    "5b": "CogVideoX-5b",
    "5b-i2v": "CogVideoX-5b-I2V",
    "1.5-5b": "CogVideoX1.5-5B",
    "1.5-5b-i2v": "CogVideoX1.5-5B-I2V",
}
# The t5-v1.1-xxl text encoder + tokenizer are identical across every repo;
# the benchmark always sources them from CogVideoX-5b (the one full download).
_T5_REPO = "CogVideoX-5b"

# Default: shard attention heads across every device (the example caps this
# to a divisor of the head count -- 2b's 30 heads -> tp=2 on a v4-8).
_DEFAULT_TP = {k: None for k in MODEL_SIZES}


def _size_slug(model_size: str, task: str) -> str:
    """`5b-i2v` + task `i2v` -> `5b` (the task already carries the i2v suffix,
    so the result slug is `cogvideox_5b_i2v`, not `cogvideox_5b_i2v_i2v`)."""
    slug = model_size
    if task == "i2v" and slug.endswith("-i2v"):
        slug = slug[: -len("-i2v")]
    return slug.replace("-", "_").replace(".", "_")


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    ckpt_dir = common.resolve_checkpoint_dir(args)
    model_dir = os.path.join(ckpt_dir, _REPO[args.model_size])
    t5_dir = os.path.join(ckpt_dir, _T5_REPO, "text_encoder")
    tok_dir = os.path.join(ckpt_dir, _T5_REPO, "tokenizer")
    is_15 = args.model_size.startswith("1.5")
    size_slug = _size_slug(args.model_size, args.task)

    # CogVideoX-1.5 (patch_size_t=2, "slice" RoPE, 81 frames) is benchmarked at
    # its *native* 1360x768 (~45k visual tokens) via DeepSpeed-Ulysses sequence
    # parallelism (--sequence_parallel_size 4 on a v4-8): the per-block
    # activations don't fit a v4 chip at full resolution otherwise, and the
    # non-SP DiT step's XLA graph over ~45k tokens never finished compiling in
    # testing. SP and Megatron TP are mutually exclusive for CogVideoX, so the
    # 1.5 rows run tp=1. The 1.0 rows (2b / 5b / 5b-i2v) fit natively at
    # 720x480 and keep the default all-device Megatron TP.
    if is_15:
        sp = args.sequence_parallel_size if args.sequence_parallel_size is not None else 4
        tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else 1
        width, height, num_frames = 1360, 768, 81
    else:
        sp = args.sequence_parallel_size if args.sequence_parallel_size is not None else 1
        tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP[args.model_size]
        width, height, num_frames = 720, 480, 49

    ns = argparse.Namespace(
        model_dir=model_dir, variant=args.model_size, t5_dir=t5_dir, tokenizer_dir=tok_dir,
        prompt=[common.STANDARD_T2V_PROMPT], negative_prompt=generate_cogvideox.DEFAULT_NEGATIVE_PROMPT,
        image_path=None,
        num_frames=num_frames,
        height=height, width=width,
        sequence_parallel_size=sp,
        num_inference_steps=50, num_steps=50,  # num_steps: read by common.run_benchmark for per_step_s
        guidance_scale=6.0, scheduler="dpm", use_dynamic_cfg=True,
        # i2v: rescale the saved video to the standard image's aspect ratio (as
        # every other model's i2v benchmark does -- see the portrait resolutions
        # in the i2v rows of docs/benchmarking.md). Host-side, post-decode, so it
        # doesn't touch generation_s / peak_hbm.
        match_image_aspect=True,
        tensor_parallel_size=tp, dtype="bfloat16", dit_dtype="bfloat16", seed=42, fps=16,
        output_path=common.output_path("cogvideox", "", size_slug, args.task),
    )
    if args.task == "i2v":
        if "i2v" not in args.model_size:
            raise SystemExit(f"--task i2v requires an i2v model_size (got {args.model_size})")
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = [common.STANDARD_I2V_PROMPT]
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="5b", choices=MODEL_SIZES)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    run_args = build_args(args)
    size_slug = _size_slug(args.model_size, args.task)
    common.run_benchmark(
        model="cogvideox", version="", size=size_slug, task=args.task,
        main_fn=generate_cogvideox.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
