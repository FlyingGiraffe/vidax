"""Benchmark harness for Cosmos 3 (Nano / Edge) -- T2V and I2V. Imports and
calls the real `examples/generate_cosmos3.py`'s `main(args)` (no separate
reimplementation of the generation loop -- see `benchmarks/common.py`'s
module docstring).

Usage:
    python benchmarks/run_cosmos3.py --model_size nano --task t2v
    python benchmarks/run_cosmos3.py --model_size edge --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_cosmos3  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TASKS = ("t2v", "i2v")


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    repo_name = "Cosmos3-Nano" if args.model_size == "nano" else "Cosmos3-Edge"
    repo_dir = os.path.join(checkpoint_dir, repo_name)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4

    ns = argparse.Namespace(
        model_size=args.model_size,
        dit_checkpoint_path=os.path.join(
            repo_dir, "transformer", "diffusion_pytorch_model.safetensors.index.json"),
        vae_checkpoint_path=os.path.join(repo_dir, "vae", "diffusion_pytorch_model.safetensors"),
        tokenizer_path=os.path.join(repo_dir, "text_tokenizer"),
        image_path=None,
        prompt=common.STANDARD_T2V_PROMPT,
        negative_prompt=generate_cosmos3.DEFAULT_NEGATIVE_PROMPT,
        max_text_len=128,
        guide_scale=6.0,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        seed=0,
        num_steps=35,
        karras_sigma_min=0.147,
        karras_sigma_max=200.0,
        height=704,
        width=1280,
        num_frames=93,
        fps=24.0,
        output_path=common.output_path("cosmos", "3", args.model_size, args.task),
    )
    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = common.STANDARD_I2V_PROMPT
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="nano", choices=["nano", "edge"])
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="cosmos", version="3", size=args.model_size, task=args.task,
        main_fn=generate_cosmos3.main, args=run_args)


if __name__ == "__main__":
    main()
