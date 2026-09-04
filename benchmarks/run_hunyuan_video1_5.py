"""Benchmark harness for HunyuanVideo-1.5 (480p/720p, T2V/I2V). Imports and
calls the real `examples/generate_hunyuan_video1_5.py`'s `main(args)` (no
separate reimplementation of the generation loop -- see `benchmarks/
common.py`'s module docstring).

Usage:
    python benchmarks/run_hunyuan_video1_5.py --resolution 480p --task t2v
    VIDAX_CHECKPOINT_DIR=/path/to/checkpoints python benchmarks/run_hunyuan_video1_5.py --resolution 720p --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_hunyuan_video1_5  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

RESOLUTIONS = ("480p", "720p")
TASKS = ("t2v", "i2v")

# (height, width) per --resolution -- matches examples/generate_hunyuan_video1_5.py's
# own --height/--width defaults for 480p; 720p is this repo's standard
# 720p-scale token count (see docs/benchmarking.md's other 720p rows).
_RESOLUTION_HW = {"480p": (480, 832), "720p": (720, 1280)}

# Reference default (`hyvideo_pipeline.py`'s own num_frames=121, ~5s @ 24fps)
# -- fits at both resolutions with --tensor_parallel_size 4 +
# --vae_tile_latent_size 8 (see docs/lessons/hunyuan_video1_5_debugging.md's
# "Tensor parallelism"/"VAE tile size vs. TP memory headroom" sections;
# confirmed real end-to-end at 480p, not yet separately confirmed at 720p --
# --vae_tile_latent_size may need to shrink further there).
NUM_FRAMES = 121
# The 8.3B DiT alone doesn't fit replicated on one TPU v4 chip -- TP is
# required, not a resolution-dependent choice.
_DEFAULT_TP_SIZE = 4
_DEFAULT_VAE_TILE_LATENT_SIZE = 8


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = os.path.join(common.resolve_checkpoint_dir(args), "HunyuanVideo-1.5")
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP_SIZE

    ns = argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        siglip_checkpoint_dir=(
            os.path.join(common.resolve_checkpoint_dir(args), "FLUX.1-Redux-dev")
            if args.task == "i2v" else None
        ),
        tensor_parallel_size=tp_size,
        resolution=args.resolution,
        prompt=common.STANDARD_T2V_PROMPT,
        negative_prompt="",
        image_path=None,
        # T2V: fixed --height/--width at the resolution's own default. I2V:
        # left None so generate_hunyuan_video1_5.main derives (height,
        # width) from the standard conditioning image's own aspect ratio
        # via --max_area (matching generate_wan2_1_i2v.py's benchmark
        # convention) -- a fixed landscape height/width would otherwise
        # silently squish the (portrait) standard image.
        height=None if args.task == "i2v" else _RESOLUTION_HW[args.resolution][0],
        width=None if args.task == "i2v" else _RESOLUTION_HW[args.resolution][1],
        max_area=_RESOLUTION_HW[args.resolution][0] * _RESOLUTION_HW[args.resolution][1],
        num_frames=args.num_frames if args.num_frames is not None else NUM_FRAMES,
        fps=24,
        num_steps=30,
        shift=None,
        guidance_scale=6.0,
        seed=0,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        vae_tile_latent_size=args.vae_tile_latent_size or _DEFAULT_VAE_TILE_LATENT_SIZE,
        output_path=common.output_path("hunyuan_video", "1.5", args.resolution, args.task),
    )
    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = common.STANDARD_I2V_PROMPT
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--resolution", type=str, default="480p", choices=RESOLUTIONS)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Overrides the reference's own 121-frame default.")
    parser.add_argument(
        "--vae_tile_latent_size", type=int, default=None,
        help="Overrides the default VAE spatial-tile size (8) -- see "
             "docs/lessons/hunyuan_video1_5_debugging.md's 'VAE tile size vs. TP memory headroom'.")
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="hunyuan_video", version="1.5", size=args.resolution, task=args.task,
        main_fn=generate_hunyuan_video1_5.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
