"""Runs every model/variant/task's benchmark, in the same order as the
model-support table in the root `README.md`, skipping any whose checkpoints
aren't present under `--checkpoint_dir` (so this is safe to run on a machine
that only has a subset of checkpoints downloaded -- e.g. this repo's own v4-8
dev machine currently only has Cosmos-Predict2.5's 2B/14B).

Each benchmark runs in its own subprocess (not an in-process import loop):
different models need different mesh shapes/parallelism, and running them
back-to-back in one process would leak JAX's global device/compilation state
between runs -- a fresh process per run is the only way to get a genuinely
clean compile-time measurement every time (matches `common.clear_jax_compilation_cache`'s
same goal, one level up).

Usage:
    python benchmarks/run_all.py
    VIDAX_CHECKPOINT_DIR=/path/to/checkpoints python benchmarks/run_all.py
"""
import argparse
import os
import subprocess
import sys

import common

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

# (script, extra_args, checkpoint_subdir_to_check_for) -- ordered exactly like
# the root README's model-support table.
RUN_MATRIX = [
    ("run_cosmos3.py", ["--model_size", "nano", "--task", "t2v"], "Cosmos3-Nano"),
    ("run_cosmos3.py", ["--model_size", "nano", "--task", "i2v"], "Cosmos3-Nano"),
    ("run_cosmos3.py", ["--model_size", "edge", "--task", "t2v"], "Cosmos3-Edge"),
    ("run_cosmos3.py", ["--model_size", "edge", "--task", "i2v"], "Cosmos3-Edge"),
    ("run_cosmos2_5.py", ["--model_size", "14B", "--task", "t2v"], "Cosmos-Predict2.5-14B"),
    ("run_cosmos2_5.py", ["--model_size", "2B", "--task", "t2v"], "Cosmos-Predict2.5-2B"),
    ("run_wan2_2.py", ["--model_size", "A14B", "--task", "t2v"], "Wan2.2-T2V-A14B"),
    ("run_wan2_2.py", ["--model_size", "A14B", "--task", "i2v"], "Wan2.2-I2V-A14B"),
    ("run_wan2_2.py", ["--model_size", "5B", "--task", "t2v"], "Wan2.2-TI2V-5B"),
    ("run_wan2_2.py", ["--model_size", "5B", "--task", "i2v"], "Wan2.2-TI2V-5B"),
    ("run_wan2_1.py", ["--model_size", "14B", "--task", "t2v"], "Wan2.1-T2V-14B"),
    ("run_wan2_1.py", ["--task", "i2v"], "Wan2.1-I2V-14B-480P"),
    ("run_wan2_1.py", ["--model_size", "1.3B", "--task", "t2v"], "Wan2.1-T2V-1.3B"),
    ("run_hunyuan_video1_5.py", ["--resolution", "480p", "--task", "t2v"], "HunyuanVideo-1.5"),
    ("run_hunyuan_video1_5.py", ["--resolution", "480p", "--task", "i2v"], "HunyuanVideo-1.5"),
    ("run_hunyuan_video1_5.py", ["--resolution", "720p", "--task", "t2v"], "HunyuanVideo-1.5"),
    ("run_hunyuan_video1_5.py", ["--resolution", "720p", "--task", "i2v"], "HunyuanVideo-1.5"),
    ("run_hunyuan_video.py", [], "HunyuanVideo"),
    ("run_cogvideox.py", ["--model_size", "5b", "--task", "t2v"], "CogVideoX-5b"),
    ("run_cogvideox.py", ["--model_size", "2b", "--task", "t2v"], "CogVideoX-2b"),
    ("run_cogvideox.py", ["--model_size", "1.5-5b", "--task", "t2v"], "CogVideoX1.5-5B"),
    ("run_cogvideox.py", ["--model_size", "5b-i2v", "--task", "i2v"], "CogVideoX-5b-I2V"),
    ("run_cogvideox.py", ["--model_size", "1.5-5b-i2v", "--task", "i2v"], "CogVideoX1.5-5B-I2V"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    args = parser.parse_args()

    checkpoint_dir = common.resolve_checkpoint_dir(args)
    print(f"Using checkpoint_dir={checkpoint_dir}")

    ran, skipped, failed = [], [], []
    for script, extra_args, checkpoint_subdir in RUN_MATRIX:
        if not os.path.isdir(os.path.join(checkpoint_dir, checkpoint_subdir)):
            print(f"SKIP {script} {' '.join(extra_args)} -- {checkpoint_subdir} not found under {checkpoint_dir}")
            skipped.append((script, extra_args))
            continue
        cmd = [sys.executable, os.path.join(BENCHMARKS_DIR, script), *extra_args,
               "--checkpoint_dir", checkpoint_dir, "--num_runs", str(args.num_runs)]
        print(f"RUN {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append((script, extra_args))
        else:
            ran.append((script, extra_args))

    print(f"\nDone. Ran {len(ran)}, skipped {len(skipped)}, failed {len(failed)}.")
    if failed:
        print("Failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
