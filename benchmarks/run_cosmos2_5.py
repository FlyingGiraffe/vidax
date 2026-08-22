"""Benchmark harness for Cosmos-Predict2.5 (2B / 14B) -- text2world,
image2world, video2world. Imports and calls the real
`examples/generate_cosmos2_5.py`'s `main(args)` (no separate reimplementation
of the generation loop -- see `benchmarks/common.py`'s module docstring).

Usage:
    python benchmarks/run_cosmos2_5.py --model_size 2B --task t2v
    python benchmarks/run_cosmos2_5.py --model_size 14B --task i2v
    VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints python benchmarks/run_cosmos2_5.py --model_size 14B --task v2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_cosmos2_5  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TASKS = ("t2v", "i2v", "v2v")


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = common.resolve_checkpoint_dir(args)
    repo_name = f"Cosmos-Predict2.5-{args.model_size}"
    dit_checkpoint_path = common.find_single(
        os.path.join(checkpoint_dir, repo_name, "base", "pre-trained", "*_ema_bf16.pt"))
    vae_checkpoint_path = os.path.join(checkpoint_dir, "Cosmos-Predict2.5-2B", "tokenizer.pth")
    reason1_checkpoint_path = os.path.join(
        checkpoint_dir, "Cosmos-Reason1-7B", "model.safetensors.index.json")

    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else 4
    sp_size = args.sequence_parallel_size if args.sequence_parallel_size is not None else 1

    # 14B at the reference's full 704x1280x93-frame default doesn't fit this
    # repo's 4-chip v4-8 test machine fully device-resident at any
    # --tensor_parallel_size/--sequence_parallel_size split (4x1 needs
    # ~22.6G/chip with 18.5G free; 2x2 needs ~13.0G/chip with only 9.1G free
    # -- weight-sharding four ways beats splitting the difference with
    # sequence-parallel here, since 14B's weights dominate over its
    # activations at this frame count). With --offload_dit_weights (see
    # generate_cosmos2_5.py's identical flag, docs/weight_offloading.md),
    # the DiT's weights no longer need to be fully resident at once, so the
    # full reference 93 frames fits -- see docs/benchmarking.md's Cosmos-
    # Predict2.5 14B row. Without it, 45 frames (still native 704x1280, same
    # "1 + 4k" VAE-valid frame count family) remains the largest that fits
    # at --tensor_parallel_size 4.
    num_frames = 93 if (args.model_size == "2B" or args.offload_dit_weights) else 45

    ns = argparse.Namespace(
        model_size=args.model_size,
        dit_checkpoint_path=dit_checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        reason1_checkpoint_path=reason1_checkpoint_path,
        tokenizer_path=None,
        image_path=None,
        video_path=None,
        num_conditional_latent_frames=1,
        max_area=704 * 1280,
        prompt=[common.STANDARD_T2V_PROMPT],
        negative_prompt=generate_cosmos2_5.DEFAULT_NEGATIVE_PROMPT,
        guide_scale=7.0,
        tensor_parallel_size=tp_size,
        sequence_parallel_size=sp_size,
        dtype="bfloat16",
        seed=0,
        num_steps=35,
        solver_order=2,
        shift=5.0,
        height=704,
        width=1280,
        num_frames=num_frames,
        fps=16,
        offload_dit_weights=args.offload_dit_weights,
        offload_chunk_size=args.offload_chunk_size,
        output_path=common.output_path("cosmos", "2.5", args.model_size.lower(), args.task),
    )

    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = [common.STANDARD_I2V_PROMPT]
    elif args.task == "v2v":
        ns.video_path = common.STANDARD_VIDEO_PATH
        ns.num_conditional_latent_frames = 2
        ns.prompt = [common.STANDARD_V2V_PROMPT]
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="2B", choices=["2B", "14B"])
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    parser.add_argument("--offload_dit_weights", action="store_true", help="See generate_cosmos2_5.py's identical flag; lets 14B fit the full reference 93-frame count instead of the reduced 45. See docs/weight_offloading.md.")
    parser.add_argument("--offload_chunk_size", type=int, default=1, help="See generate_cosmos2_5.py's identical flag.")
    args = parser.parse_args()

    run_args = build_args(args)
    common.run_benchmark(
        model="cosmos", version="2.5", size=args.model_size.lower(), task=args.task,
        main_fn=generate_cosmos2_5.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
