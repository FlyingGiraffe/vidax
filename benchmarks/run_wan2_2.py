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
    # tp=4/sp=1 (all 4 chips shard weights, none go to sequence-parallel) is
    # the default -- sp=4/tp=1 was tried first (this is TI2V-5B's largest
    # token count of all Wan models) but leaves the dit_dtype=float32 DiT
    # weights (~20GB) fully unsharded/replicated per chip, OOMing during T5
    # encoding before the DiT even runs. tp=2/sp=2 gets past that but still
    # OOMs inside the DiT sampling step itself (39.4G required vs 30.75G
    # available) at the full 121-frame/704x1280 reference config. Only
    # tp=4/sp=1 fits -- confirmed working end-to-end at full resolution/
    # frame count.
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4
    sp_size = args.sequence_parallel_size if args.sequence_parallel_size is not None else 1
    # `generate_wan2_2_ti2v.py`'s own `--num_steps` default resolves `None`
    # to 50 (t2v) / 40 (i2v) internally, but `common.run_benchmark`'s
    # per-step-time calculation reads `args.num_steps` directly off this
    # harness's own Namespace, not what the script resolved it to -- passing
    # `None` through leaves `per_step_s` empty in the results (it silently
    # divides by `None` instead of the real step count). Resolve explicitly
    # here so the harness's own args carry the actual value used.
    num_steps = 40 if args.task == "i2v" else 50
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
        dit_dtype="float32",
        seed=0,
        num_steps=num_steps,
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
    num_frames = args.num_frames if args.num_frames is not None else 81
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
        dit_dtype="float32",
        seed=0,
        num_frames=num_frames,
        # See generate_wan2_2_i2v_a14b.py's identical flags -- offloading
        # composes with A14B's two-expert MoE switch, see
        # docs/weight_offloading.md. Now wired up for T2V too (see
        # generate_wan2_2_t2v_a14b.py).
        offload_dit_weights=args.offload_dit_weights,
        offload_chunk_size=args.offload_chunk_size,
        output_path=common.output_path(
            "wan", "2.2",
            "a14b_720p" if (
                (args.task == "i2v" and args.i2v_resolution == "720P")
                or (args.task == "t2v" and args.t2v_resolution == "720P")
            ) else "a14b",
            args.task),
    )
    if args.task == "t2v":
        # `--t2v_resolution` picks which (height, width) to benchmark --
        # mirrors `--i2v_resolution`'s existing pattern below. 720P is
        # A14B T2V's own native default (`generate_wan2_2_t2v_a14b.py`'s
        # `--height`/`--width` defaults); 480P is the smaller config that
        # fits this 4-chip machine without needing to reduce frame count
        # (see docs/benchmarking.md's `§` footnote).
        height, width = (480, 832) if args.t2v_resolution == "480P" else (720, 1280)
        return argparse.Namespace(
            prompt=[common.STANDARD_T2V_PROMPT],
            negative_prompt=generate_wan2_2_t2v_a14b.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0, boundary=0.875, num_steps=50, shift=12.0,
            height=height, width=width, **common_kwargs)
    else:
        # `--i2v_resolution` picks which max_area to benchmark -- see
        # generate_wan2_1_i2v.py's identical flag for the pattern. Unlike
        # Wan2.1's I2V-14B, A14B ships one checkpoint used at either
        # resolution (no separate 480P/720P repos), and its reference
        # shift=5.0 doesn't vary by resolution.
        max_area = 480 * 832 if args.i2v_resolution == "480P" else 720 * 1280
        return argparse.Namespace(
            image_path=common.STANDARD_IMAGE_PATH,
            prompt=common.STANDARD_I2V_PROMPT,
            negative_prompt=generate_wan2_2_i2v_a14b.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0, boundary=0.900, num_steps=40, shift=5.0,
            max_area=max_area, **common_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="5B", choices=MODEL_SIZES)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    parser.add_argument("--offload_dit_weights", action="store_true", help="A14B only (see generate_wan2_2_i2v_a14b.py's identical flag; not yet wired up for A14B T2V). Per-layer weight offloading, composed with the two-expert MoE switch and with --sequence_parallel_size -- see docs/weight_offloading.md's 'A14B (Wan2.2)' section.")
    parser.add_argument("--offload_chunk_size", type=int, default=1, help="A14B only. See generate_wan2_2_i2v_a14b.py's identical flag.")
    parser.add_argument("--num_frames", type=int, default=None, help="A14B only. Overrides the benchmark script's own default (81).")
    parser.add_argument("--i2v_resolution", type=str, default="720P", choices=["480P", "720P"], help="A14B I2V only -- picks --max_area (480*832 or 720*1280). See generate_wan2_1_i2v.py's identical flag for the pattern.")
    parser.add_argument("--t2v_resolution", type=str, default="720P", choices=["480P", "720P"], help="A14B T2V only -- picks --height/--width (480x832 or 720x1280, the script's own native default).")
    args = parser.parse_args()

    if args.model_size == "5B":
        run_args = build_args_5b(args)
        main_fn = generate_wan2_2_ti2v.main
        size = "5b-ti2v"
    else:
        run_args = build_args_a14b(args)
        main_fn = generate_wan2_2_t2v_a14b.main if args.task == "t2v" else generate_wan2_2_i2v_a14b.main
        is_720p = (
            (args.task == "i2v" and args.i2v_resolution == "720P")
            or (args.task == "t2v" and args.t2v_resolution == "720P"))
        size = "a14b_720p" if is_720p else "a14b"

    common.run_benchmark(
        model="wan", version="2.2", size=size, task=args.task, main_fn=main_fn, args=run_args,
        num_runs=args.num_runs)


if __name__ == "__main__":
    main()
