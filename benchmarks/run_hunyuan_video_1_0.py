"""Benchmark harness for HunyuanVideo 1.0 (T2V only, one checkpoint
variant -- `tencent/HunyuanVideo`'s `hunyuan-video-t2v-720p`). Imports and
calls the real `examples/generate_hunyuan_video_1_0.py`'s `main(args)` (no
separate reimplementation of the generation loop -- see `benchmarks/
common.py`'s module docstring).

Usage:
    python benchmarks/run_hunyuan_video_1_0.py
    VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints python benchmarks/run_hunyuan_video_1_0.py --num_runs 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_hunyuan_video_1_0  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# The reference's own default `--video-size (720, 1280)` / `--video-length
# 129` OOMs on this dev box's TPU v4-8 at `--tensor_parallel_size 4`: the
# 13B DiT (TP-sharded, ~6.5GB/chip bf16) plus the 8B Llama text tower
# (replicated, unsharded, ~16GB/chip bf16 -- no existing TP wiring for it)
# already resident leaves only ~5GB/chip free, and the reference's own
# 129-frame/720p token count's activations (both cond+uncond branches)
# need much more than that -- confirmed via direct measurement: 720x1280
# even at just 5 frames needs 7.27G with 5.06G free; 480x832 needs 6.40G
# at 13 frames, 9.33G at 21 frames (all OOM), but *does* fit at 5 frames.
# `--height 480 --width 832 --num_frames 5` below is therefore this port's
# largest *confirmed-working* config on this hardware, not the reference's
# real default -- see docs/models/hunyuan_video_1_0.md's Architecture notes
# and docs/lessons/hunyuan_video_1_debugging.md for the full numbers.
# Closing this gap needs weight offloading and/or sequence parallelism
# (out of scope this batch, same precedent as HunyuanVideo-1.5's own
# "TP only, no offloading yet" scope cut -- see docs/benchmarking.md's
# "Why some rows need offloading" table).
_HEIGHT, _WIDTH = 480, 832
NUM_FRAMES = 5
_DEFAULT_TP_SIZE = 4  # 13B DiT doesn't fit replicated on one TPU v4 chip.
_DEFAULT_VAE_TILE_LATENT_SIZE = 8


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    root = common.resolve_checkpoint_dir(args)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP_SIZE

    return argparse.Namespace(
        checkpoint_dir=os.path.join(root, "HunyuanVideo"),
        text_encoder_dir=os.path.join(root, "HunyuanVideo", "text_encoder"),
        clip_checkpoint_dir=os.path.join(root, "HunyuanVideo", "clip-vit-large-patch14"),
        model="HYVideo-T/2-cfgdistill",
        tensor_parallel_size=tp_size,
        prompt=common.STANDARD_T2V_PROMPT,
        negative_prompt=None,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=args.num_frames if args.num_frames is not None else NUM_FRAMES,
        fps=24,
        num_steps=30,
        shift=7.0,
        guidance_scale=1.0,
        embedded_guidance_scale=6.0,
        seed=0,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        vae_tile_latent_size=args.vae_tile_latent_size or _DEFAULT_VAE_TILE_LATENT_SIZE,
        output_path=common.output_path("hunyuan_video", "_1", "720p", "t2v"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Overrides the reference's own 129-frame default.")
    parser.add_argument(
        "--vae_tile_latent_size", type=int, default=None,
        help="Overrides the default VAE spatial-tile size (8).")
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="hunyuan_video", version="_1", size="720p", task="t2v",
        main_fn=generate_hunyuan_video_1_0.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
