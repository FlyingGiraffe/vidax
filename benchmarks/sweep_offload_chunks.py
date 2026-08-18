"""Sweeps `--offload_chunk_size` for `--offload_dit_weights` (see
docs/weight_offloading.md) at one fixed resolution/model, measuring
per-step time and peak HBM/chip at each chunk size -- answers "does
grouping more blocks per offloaded HBM buffer actually help throughput,
and how much extra HBM does it cost" empirically instead of guessing.

Reuses `examples/generate_wan2_1_t2v.py`'s real `main(args)` (no separate
generation-loop reimplementation, matching every other `benchmarks/run_*.py`
script) and `benchmarks/common.py`'s timing/HBM helpers -- only the sweep
loop and result-table formatting are specific to this script, since
`run_wan2_1.py`'s own per-model/task/size result-file naming isn't set up
for "same config, N chunk sizes" the way this needs.

Only ever needs `--num_runs 1` per chunk size (unlike every other
`benchmarks/run_*.py` row): each native-720P/14B run already takes minutes
to tens of minutes even at a handful of steps, so 5x that for negligible
extra precision isn't worth it here either -- same reasoning as
`docs/benchmarking.md`'s native-720P rows.

Usage:
    python benchmarks/sweep_offload_chunks.py
    python benchmarks/sweep_offload_chunks.py --chunk_sizes 1,2,4,8,20,40 --num_steps 5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_wan2_1_t2v  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                         help=f"See run_wan2_1.py's identical flag. Defaults to the "
                              f"'{common.CHECKPOINT_DIR_ENV_VAR}' environment variable if set, else "
                              f"'{common.DEFAULT_CHECKPOINT_DIR}'.")
    parser.add_argument("--chunk_sizes", type=str, default="1,2,4,8,20,40",
                         help="Comma-separated --offload_chunk_size values to sweep. Each must "
                              "divide the 14B DiT's num_layers (40).")
    parser.add_argument("--num_steps", type=int, default=5,
                         help="Sampling steps per chunk size -- kept small since only steady-state "
                              "per-step time/peak-HBM are of interest here, not final output "
                              "quality (already verified separately, see docs/weight_offloading.md).")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or os.environ.get(
        common.CHECKPOINT_DIR_ENV_VAR) or common.DEFAULT_CHECKPOINT_DIR
    repo_dir = os.path.join(checkpoint_dir, "Wan2.1-T2V-14B")
    chunk_sizes = [int(c) for c in args.chunk_sizes.split(",")]

    out_dir = os.path.join(common.OUT_DIR, "wan2_1_14b_720p_t2v_offload_sweep")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for chunk_size in chunk_sizes:
        run_args = argparse.Namespace(
            model_size="14B",
            dit_checkpoint_path=os.path.join(
                repo_dir, "diffusion_pytorch_model.safetensors.index.json"),
            vae_checkpoint_path=os.path.join(repo_dir, "Wan2.1_VAE.pth"),
            t5_checkpoint_path=os.path.join(repo_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=None,
            prompt=[common.STANDARD_T2V_PROMPT],
            negative_prompt=generate_wan2_1_t2v.DEFAULT_NEGATIVE_PROMPT,
            guide_scale=5.0,
            tensor_parallel_size=args.tensor_parallel_size,
            sequence_parallel_size=1,
            dtype="bfloat16",
            dit_dtype="float32",
            offload_dit_weights=True,
            offload_chunk_size=chunk_size,
            seed=0,
            num_steps=args.num_steps,
            shift=5.0,
            height=args.height,
            width=args.width,
            num_frames=81,
            output_path=os.path.join(out_dir, f"chunk{chunk_size}.mp4"),
        )

        common.clear_jax_compilation_cache()
        timing = common.instrument_jit()
        t_wall0 = time.perf_counter()
        generate_wan2_1_t2v.main(run_args)
        wall_s = time.perf_counter() - t_wall0

        result = {
            "offload_chunk_size": chunk_size,
            "compile_s": timing.compile_s,
            "generation_s": timing.generation_s,
            "per_step_s": timing.generation_s / args.num_steps,
            "peak_hbm_gb": common.peak_hbm_per_chip_gb(),
            "wall_s": wall_s,
        }
        results.append(result)
        print(f"offload_chunk_size={chunk_size}: {json.dumps(result)}")

    result_path = os.path.join(common.RESULTS_DIR, "wan2_1_14b_720p_t2v_offload_sweep.json")
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({
            "model": "wan", "version": "2.1", "size": "14b_720p", "task": "t2v",
            "num_steps": args.num_steps, "tensor_parallel_size": args.tensor_parallel_size,
            "height": args.height, "width": args.width, "results": results,
        }, f, indent=2)
    print(f"\n=== Wrote {result_path} ===")

    print("\n| offload_chunk_size | Per-step (s) | Peak HBM/chip (GB) |")
    print("| ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['offload_chunk_size']} | {r['per_step_s']:.1f} | {r['peak_hbm_gb']:.1f} |")


if __name__ == "__main__":
    main()
