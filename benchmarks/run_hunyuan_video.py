"""Benchmark harness for HunyuanVideo 1.0 (T2V only, one checkpoint
variant -- `tencent/HunyuanVideo`'s `hunyuan-video-t2v-720p`). Imports and
calls the real `examples/generate_hunyuan_video.py`'s `main(args)` (no
separate reimplementation of the generation loop -- see `benchmarks/
common.py`'s module docstring).

Usage:
    python benchmarks/run_hunyuan_video.py
    VIDAX_CHECKPOINT_DIR=/path/to/checkpoints python benchmarks/run_hunyuan_video.py --num_runs 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_hunyuan_video  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# The reference's own default `--video-size (720, 1280)` / `--video-length
# 129` OOMs on this dev box's TPU v4-8 at `--tensor_parallel_size 4` if the
# DiT's double/single-stream block weights stay fully HBM-resident, even
# after freeing the 8B Llama text tower's ~16GB/chip (replicated, unsharded
# -- no TP wiring for it) once prompt encoding is done (see
# `generate_hunyuan_video.py`'s `del encode, llama_model, ...` and
# docs/lessons/hunyuan_video_debugging.md): 129 frames needs 122.86G of HLO
# temporaries against 30.75G available with the DiT fully resident. Closed
# with `--offload_dit_weights` (see docs/weight_offloading.md): the 20
# double-stream / 40 single-stream blocks' weights stay host-resident and
# stream into HBM `--offload_chunk_size_{double,single}` blocks at a time
# instead -- confirmed working end-to-end at the real 129-frame/720p/
# 50-step default. Both default to their stream's full depth (one chunk
# for all 20 double blocks, one chunk for all 40 single blocks -- the two
# streams' chunk sizes are independent, not a single shared value), which
# empirically beat smaller chunk sizes by ~2-3x/step (fewer, larger
# host-to-device transfers and jax.jit dispatches) while still fitting --
# see docs/lessons/hunyuan_video_debugging.md for the full numbers.
_HEIGHT, _WIDTH = 720, 1280
NUM_FRAMES = 129  # The reference's own `--video-length` default.
NUM_STEPS = 50  # The reference's own `--infer-steps` default.
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
        num_steps=args.num_steps if args.num_steps is not None else NUM_STEPS,
        shift=7.0,
        guidance_scale=1.0,
        embedded_guidance_scale=6.0,
        seed=0,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        vae_tile_latent_size=args.vae_tile_latent_size or _DEFAULT_VAE_TILE_LATENT_SIZE,
        offload_dit_weights=not args.no_offload,
        offload_chunk_size_double=args.offload_chunk_size_double,
        offload_chunk_size_single=args.offload_chunk_size_single,
        output_path=common.output_path("hunyuan_video", "", "720p", "t2v"),
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
        help="Disable --offload_dit_weights (on by default -- required to fit the reference's real "
             "129-frame/720p default in HBM, see this module's docstring). Only useful for a smaller "
             "--num_frames override that fits fully resident.")
    parser.add_argument(
        "--offload_chunk_size_double", type=int, default=None,
        help="Overrides the default double-stream offload chunk size (20, i.e. all in one chunk).")
    parser.add_argument(
        "--offload_chunk_size_single", type=int, default=None,
        help="Overrides the default single-stream offload chunk size (40, i.e. all in one chunk).")
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="hunyuan_video", version="", size="720p", task="t2v",
        main_fn=generate_hunyuan_video.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
