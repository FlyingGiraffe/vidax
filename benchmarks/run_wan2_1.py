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
        # T2V-14B ships as a single checkpoint used at any resolution
        # (unlike I2V's two separately-tuned 480P/720P checkpoints below) --
        # `--t2v_resolution` just picks which (height, width) to benchmark
        # it at, defaulting to this repo's existing 480P row.
        height, width = (720, 1280) if args.t2v_resolution == "720P" else (480, 832)
        size_slug = args.model_size.lower() if args.t2v_resolution == "480P" else f"{args.model_size.lower()}_720p"
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
            dit_dtype="float32",
            offload_dit_weights=args.offload_dit_weights,
            offload_chunk_size=args.offload_chunk_size,
            seed=0,
            num_steps=50,
            shift=5.0,
            height=height,
            width=width,
            num_frames=81,
            output_path=common.output_path("wan", "2.1", size_slug, "t2v"),
        )
    else:  # i2v, 14B only -- ships as two separate checkpoints (480P/720P),
        # each trained/tuned at its own resolution range (identical
        # architecture/config.json, different weights -- see
        # docs/models/wan2_1.md#i2v-14b). --i2v_resolution picks which one.
        repo_dir = os.path.join(checkpoint_dir, f"Wan2.1-I2V-14B-{args.i2v_resolution}")
        max_area = 832 * 480 if args.i2v_resolution == "480P" else 720 * 1280
        shift = 3.0 if args.i2v_resolution == "480P" else 5.0
        # I2V derives its actual output resolution from the conditioning
        # image's aspect ratio (compute_latent_grid), not a --height/--width
        # flag -- computed here just so common.run_benchmark's result JSON
        # records the real resolution instead of "NonexNone".
        from PIL import Image
        with Image.open(common.STANDARD_IMAGE_PATH) as im:
            image_w, image_h = im.size
        pixel_h, pixel_w, _, _ = generate_wan2_1_i2v.compute_latent_grid(
            image_h, image_w, max_area, vae_stride=(4, 8, 8), patch_size=(1, 2, 2))
        return argparse.Namespace(
            dit_checkpoint_path=os.path.join(repo_dir, "diffusion_pytorch_model.safetensors.index.json"),
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
            dit_dtype="float32",
            offload_dit_weights=args.offload_dit_weights,
            offload_chunk_size=args.offload_chunk_size,
            seed=0,
            num_steps=40,
            shift=shift,
            guide_scale=5.0,
            max_area=max_area,
            num_frames=81,
            height=pixel_h,
            width=pixel_w,
            output_path=common.output_path(
                "wan", "2.1", f"14b_{args.i2v_resolution.lower()}", "i2v"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="1.3B", choices=["1.3B", "14B"],
                         help="Ignored for --task i2v (Wan2.1 i2v only ships a 14B checkpoint).")
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    parser.add_argument("--i2v_resolution", type=str, default="480P", choices=["480P", "720P"],
                         help="Ignored for --task t2v. I2V-14B ships as two separate checkpoints "
                              "(Wan2.1-I2V-14B-480P/720P), each trained/tuned at its own "
                              "resolution range -- picks which one to load and benchmark.")
    parser.add_argument("--t2v_resolution", type=str, default="480P", choices=["480P", "720P"],
                         help="Ignored for --task i2v. T2V-14B/1.3B ship as a single checkpoint "
                              "used at any resolution (unlike i2v's two separate checkpoints) -- "
                              "picks (height, width) to benchmark it at: 480x832 (this repo's "
                              "existing default row) or native 720x1280.")
    parser.add_argument("--offload_dit_weights", action="store_true",
                         help="Passed through to generate_wan2_1_t2v.py/_i2v.py's identical flag: "
                              "offload the DiT's per-block weights into HBM one chunk at a time "
                              "instead of keeping the whole tree resident. Needed at native 720P "
                              "(see docs/weight_offloading.md); not needed at 480P.")
    parser.add_argument("--offload_chunk_size", type=int, default=1,
                         help="Passed through to generate_wan2_1_t2v.py/_i2v.py's identical flag: "
                              "number of consecutive DiT blocks offloaded together. Ignored unless "
                              "--offload_dit_weights is also set.")
    args = parser.parse_args()

    run_args = build_args(args)
    if args.task == "t2v":
        size = args.model_size.lower() if args.t2v_resolution == "480P" else f"{args.model_size.lower()}_720p"
    else:
        size = f"14b_{args.i2v_resolution.lower()}"
    main_fn = generate_wan2_1_t2v.main if args.task == "t2v" else generate_wan2_1_i2v.main
    common.run_benchmark(
        model="wan", version="2.1", size=size, task=args.task, main_fn=main_fn, args=run_args,
        num_runs=args.num_runs)


if __name__ == "__main__":
    main()
