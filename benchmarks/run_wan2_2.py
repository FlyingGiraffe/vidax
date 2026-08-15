"""Benchmark harness for Wan2.2 -- TI2V-5B (t2v/i2v) and A14B (t2v/i2v,
two-expert MoE). Imports and calls the real `examples/generate_wan2_2_ti2v.py`
/ `generate_wan2_2_t2v_a14b.py` / `generate_wan2_2_i2v_a14b.py` `main(args)`
(no separate reimplementation of the generation loop -- see
`benchmarks/common.py`'s module docstring).

Usage:
    python benchmarks/run_wan2_2.py --model_size 5B --task t2v
    python benchmarks/run_wan2_2.py --model_size 5B --task i2v
    python benchmarks/run_wan2_2.py --model_size A14B --task t2v
    python benchmarks/run_wan2_2.py --model_size A14B --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_wan2_2_ti2v  # noqa: E402
import generate_wan2_2_t2v_a14b  # noqa: E402
import generate_wan2_2_i2v_a14b  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODEL_SIZES = ("5B", "A14B")
TASKS = ("t2v", "i2v")


def build_args_5b(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    repo_dir = os.path.join(checkpoint_dir, "Wan2.2-TI2V-5B")
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 1
    sp_size = args.sequence_parallel_size if args.sequence_parallel_size is not None else 4
    ns = argparse.Namespace(
        dit_checkpoint_path=os.path.join(repo_dir, "diffusion_pytorch_model.safetensors.index.json"),
        vae_checkpoint_path=os.path.join(repo_dir, "Wan2.2_VAE.pth"),
        t5_checkpoint_path=os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=None,
        image_path=None,
        max_area=704 * 1280,
        prompt=[common.STANDARD_T2V_PROMPT],
        negative_prompt=generate_wan2_2_ti2v.DEFAULT_NEGATIVE_PROMPT,
        guide_scale=5.0,
        tensor_parallel_size=tp_size,
        sequence_parallel_size=sp_size,
        dtype="bfloat16",
        seed=0,
        num_steps=None,  # script default: 50 t2v / 40 i2v
        shift=5.0,
        height=704,
        width=1280,
        num_frames=121,
        fps=24,
        output_path=common.output_path("wan", "2.2", "5b-ti2v", args.task),
    )
    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = [common.STANDARD_I2V_PROMPT]
    return ns


def build_args_a14b(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    repo_name = "Wan2.2-I2V-A14B" if args.task == "i2v" else "Wan2.2-T2V-A14B"
    repo_dir = os.path.join(checkpoint_dir, repo_name)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 2
    sp_size = args.sequence_parallel_size if args.sequence_parallel_size is not None else 2
    common_kwargs = dict(
        high_noise_dit_checkpoint_path=os.path.join(
            repo_dir, "high_noise_model", "diffusion_pytorch_model.safetensors.index.json"),
        low_noise_dit_checkpoint_path=os.path.join(
            repo_dir, "low_noise_model", "diffusion_pytorch_model.safetensors.index.json"),
        vae_checkpoint_path=os.path.join(repo_dir, "Wan2.1_VAE.pth"),
        t5_checkpoint_path=os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=None,
        tensor_parallel_size=tp_size,
        sequence_parallel_size=sp_size,
        dtype="bfloat16",
        seed=0,
        num_frames=81,
        output_path=common.output_path("wan", "2.2", "a14b", args.task),
    )
    if args.task == "t2v":
        return argparse.Namespace(
            prompt=[common.STANDARD_T2V_PROMPT],
            negative_prompt=generate_wan2_2_t2v_a14b.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0, boundary=0.875, num_steps=50, shift=12.0,
            height=720, width=1280, **common_kwargs)
    else:
        return argparse.Namespace(
            image_path=common.STANDARD_IMAGE_PATH,
            prompt=common.STANDARD_I2V_PROMPT,
            negative_prompt=generate_wan2_2_i2v_a14b.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0, boundary=0.900, num_steps=40, shift=5.0,
            max_area=720 * 1280, **common_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="5B", choices=MODEL_SIZES)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    if args.model_size == "5B":
        run_args = build_args_5b(args)
        main_fn = generate_wan2_2_ti2v.main
        size = "5b-ti2v"
    else:
        run_args = build_args_a14b(args)
        main_fn = generate_wan2_2_t2v_a14b.main if args.task == "t2v" else generate_wan2_2_i2v_a14b.main
        size = "a14b"

    common.run_benchmark(
        model="wan", version="2.2", size=size, task=args.task, main_fn=main_fn, args=run_args,
        num_runs=args.num_runs)


if __name__ == "__main__":
    main()
