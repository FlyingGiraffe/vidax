"""Benchmark harness for Wan2.1 -- T2V (1.3B/14B) and I2V (14B only).
Imports and calls the real `examples/generate_wan2_1_t2v.py` /
`generate_wan2_1_i2v.py` `main(args)` (no separate reimplementation of the
generation loop -- see `benchmarks/common.py`'s module docstring).

Usage:
    python benchmarks/run_wan2_1.py --model_size 1.3B --task t2v
    python benchmarks/run_wan2_1.py --model_size 14B --task t2v
    python benchmarks/run_wan2_1.py --model_size 14B --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_wan2_1_t2v  # noqa: E402
import generate_wan2_1_i2v  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TASKS = ("t2v", "i2v")


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4
    sp_size = args.sequence_parallel_size if args.sequence_parallel_size is not None else 1

    if args.task == "t2v":
        repo_name = f"Wan2.1-T2V-{args.model_size}"
        repo_dir = os.path.join(checkpoint_dir, repo_name)
        dit_checkpoint_path = (
            os.path.join(repo_dir, "diffusion_pytorch_model.safetensors") if args.model_size == "1.3B"
            else os.path.join(repo_dir, "diffusion_pytorch_model.safetensors.index.json"))
        return argparse.Namespace(
            model_size=args.model_size,
            dit_checkpoint_path=dit_checkpoint_path,
            vae_checkpoint_path=os.path.join(repo_dir, "Wan2.1_VAE.pth"),
            t5_checkpoint_path=os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=None,
            prompt=[common.STANDARD_T2V_PROMPT],
            negative_prompt=generate_wan2_1_t2v.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0,
            tensor_parallel_size=tp_size,
            sequence_parallel_size=sp_size,
            dtype="bfloat16",
            seed=0,
            num_steps=50,
            shift=5.0,
            height=480,
            width=832,
            num_frames=81,
            output_path=common.output_path("wan", "2.1", args.model_size.lower(), "t2v"),
        )
    else:  # i2v, 14B only
        repo_dir = os.path.join(checkpoint_dir, "Wan2.1-I2V-14B-480P")
        return argparse.Namespace(
            dit_checkpoint_path=os.path.join(repo_dir, "diffusion_pytorch_model.safetensors"),
            vae_checkpoint_path=os.path.join(repo_dir, "Wan2.1_VAE.pth"),
            t5_checkpoint_path=os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            clip_checkpoint_path=os.path.join(
                repo_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            tokenizer_path=None,
            image_path=common.STANDARD_IMAGE_PATH,
            prompt=common.STANDARD_I2V_PROMPT,
            negative_prompt=generate_wan2_1_i2v.DEFAULT_NEGATIVE_PROMPT,
            tensor_parallel_size=tp_size,
            sequence_parallel_size=sp_size,
            dtype="bfloat16",
            seed=0,
            num_steps=40,
            shift=5.0,
            guide_scale=5.0,
            max_area=720 * 1280,
            num_frames=81,
            output_path=common.output_path("wan", "2.1", "14b", "i2v"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="1.3B", choices=["1.3B", "14B"],
                         help="Ignored for --task i2v (Wan2.1 i2v only ships a 14B checkpoint).")
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    run_args = build_args(args)
    size = args.model_size.lower() if args.task == "t2v" else "14b"
    main_fn = generate_wan2_1_t2v.main if args.task == "t2v" else generate_wan2_1_i2v.main
    common.run_benchmark(
        model="wan", version="2.1", size=size, task=args.task, main_fn=main_fn, args=run_args,
        num_runs=args.num_runs)


if __name__ == "__main__":
    main()
