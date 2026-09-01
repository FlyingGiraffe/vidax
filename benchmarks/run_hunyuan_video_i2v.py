"""Benchmark harness for HunyuanVideo-I2V (`token_replace` mode, the
released `hunyuan-video-i2v-720p` checkpoint's only shipped config).
Imports and calls the real `examples/generate_hunyuan_video_i2v.py`'s
`main(args)` (no separate reimplementation of the generation loop -- see
`benchmarks/common.py`'s module docstring).

Usage:
    python benchmarks/run_hunyuan_video_i2v.py
    VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints python benchmarks/run_hunyuan_video_i2v.py --num_runs 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_hunyuan_video_i2v  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Same real-default reasoning as run_hunyuan_video.py's T2V row: the 13B
# DiT needs --offload_dit_weights to fit the reference's real 129-frame
# default in HBM on this 4-chip hardware -- see docs/weight_offloading.md
# and docs/lessons/hunyuan_video_debugging.md. `--flow-shift`/
# `--embedded-cfg-scale` defaults here are I2V-specific (17.0/6.0, not
# T2V's 7.0/6.0) -- confirmed against the real
# scripts/run_sample_image2video_dynamic.sh launch script, not the bare
# (differently-defaulted) argparse flags -- see
# generate_hunyuan_video_i2v.py's module docstring.
NUM_FRAMES = 129  # The reference's own `--video-length` default.
NUM_STEPS = 50  # The reference's own `--infer-steps` default.
I2V_RESOLUTION = "720p"
_DEFAULT_TP_SIZE = 4  # 13B DiT doesn't fit replicated on one TPU v4 chip.
_DEFAULT_VAE_TILE_LATENT_SIZE = 8


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    root = common.resolve_checkpoint_dir(args)
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP_SIZE

    return argparse.Namespace(
        checkpoint_dir=os.path.join(root, "HunyuanVideo-I2V"),
        vae_checkpoint_dir=None,  # defaults to --checkpoint_dir; byte-identical to T2V's VAE.
        llava_checkpoint_dir=os.path.join(root, "llava-llama-3-8b-v1_1-transformers"),
        clip_checkpoint_dir=os.path.join(root, "HunyuanVideo", "clip-vit-large-patch14"),
        model="HYVideo-T/2-cfgdistill",
        tensor_parallel_size=tp_size,
        image_path=common.STANDARD_IMAGE_PATH,
        i2v_resolution=I2V_RESOLUTION,
        prompt=common.STANDARD_I2V_PROMPT,
        negative_prompt=None,
        negative_prompt_default=(
            "deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs"),
        num_frames=args.num_frames if args.num_frames is not None else NUM_FRAMES,
        fps=24,
        num_steps=args.num_steps if args.num_steps is not None else NUM_STEPS,
        shift=17.0,
        guidance_scale=1.0,
        embedded_guidance_scale=6.0,
        hidden_state_skip_layer=2,
        image_embed_interleave=4,
        seed=0,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        vae_tile_latent_size=args.vae_tile_latent_size or _DEFAULT_VAE_TILE_LATENT_SIZE,
        offload_dit_weights=not args.no_offload,
        offload_chunk_size_double=args.offload_chunk_size_double,
        offload_chunk_size_single=args.offload_chunk_size_single,
        output_path=common.output_path("hunyuan_video", "", "720p", "i2v"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Overrides the reference's own 129-frame default.")
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Overrides the reference's own 50-step default.")
    parser.add_argument(
        "--vae_tile_latent_size", type=int, default=None,
        help="Overrides the default VAE spatial-tile size (8).")
    parser.add_argument(
        "--no_offload", action="store_true",
        help="Disable --offload_dit_weights (on by default -- required to fit the real 129-frame/720p "
             "default in HBM, see this module's docstring).")
    parser.add_argument(
        "--offload_chunk_size_double", type=int, default=None,
        help="Overrides the default double-stream offload chunk size (20, i.e. all in one chunk).")
    parser.add_argument(
        "--offload_chunk_size_single", type=int, default=None,
        help="Overrides the default single-stream offload chunk size (40, i.e. all in one chunk).")
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="hunyuan_video", version="", size="720p", task="i2v",
        main_fn=generate_hunyuan_video_i2v.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
