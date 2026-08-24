"""Benchmark harness for LTX-2.5 (22B-dev, 22B-distilled) -- t2v, i2v.
Imports and calls the real `examples/generate_ltx2_5.py`'s `main(args)` (no
separate reimplementation of the generation loop -- see `benchmarks/
common.py`'s module docstring).

Usage:
    python benchmarks/run_ltx2_5.py --model_size 22B-distilled --task t2v
    python benchmarks/run_ltx2_5.py --model_size 22B-dev --task t2v
    VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints python benchmarks/run_ltx2_5.py --model_size 22B-distilled --task i2v
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))
import generate_ltx2_5  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from vidax.schedulers.ltx2_5_ancestral_euler import DEV_NUM_STEPS, DISTILLED_SIGMA_VALUES  # noqa: E402

MODEL_SIZES = ("22B-dev", "22B-distilled")
TASKS = ("t2v", "i2v")

# Each variant's own released checkpoint file name (see docs/models/ltx2_5.md).
_DIT_CHECKPOINT_FILES = {
    "22B-dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "22B-distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
}
_VAE_CHECKPOINT_FILE = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
_TEXT_ENCODER_CHECKPOINT_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
_SAMPLER = {"22B-dev": "dev", "22B-distilled": "distilled"}
# Both the 22B DiT's bf16 weights (~44GB) and the 12B Gemma-4 encoder's
# (~24GB) don't fit replicated on a single TPU v4 chip at all -- tp=4 is
# required, not a resolution-dependent choice the way LTX-Video's 2B row
# was (see docs/models/ltx2_5.md's Tensor parallelism section).
_DEFAULT_TP_SIZE = {"22B-dev": 4, "22B-distilled": 4}


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = os.path.join(common.resolve_checkpoint_dir(args), "LTX-2.5")
    dit_checkpoint_path = os.path.join(checkpoint_dir, _DIT_CHECKPOINT_FILES[args.model_size])
    vae_checkpoint_path = os.path.join(checkpoint_dir, _VAE_CHECKPOINT_FILE)
    text_encoder_checkpoint_path = os.path.join(checkpoint_dir, _TEXT_ENCODER_CHECKPOINT_FILE)

    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else _DEFAULT_TP_SIZE[args.model_size]
    sampler = _SAMPLER[args.model_size]

    size_slug = args.model_size.lower().replace("-", "_")
    ns = argparse.Namespace(
        dit_checkpoint_path=dit_checkpoint_path,
        vae_checkpoint_path=vae_checkpoint_path,
        text_encoder_checkpoint_path=text_encoder_checkpoint_path,
        tensor_parallel_size=tp_size,
        prompt=[common.STANDARD_T2V_PROMPT],
        negative_prompt=generate_ltx2_5.DEFAULT_NEGATIVE_PROMPT,
        image_path=None,
        conditioning_strength=1.0,
        text_max_tokens=256,
        dtype="bfloat16",
        dit_dtype="bfloat16",
        seed=0,
        sampler=sampler,
        sigmas=None,
        eta=None,
        guidance_scale=None,
        # Not the reference's own single-stage default (height=704,
        # width=1216, num_frames=121, the same resolution LTX-Video's
        # benchmark rows use) -- that OOMs even at tp=4 on this repo's
        # reference v4-8 test machine for a 22B model (47.2GB required vs.
        # 30.75GB/chip available; LTX-Video's 13B fit at the same
        # resolution/tp, this doesn't). 121 frames alone OOMs too, even at a
        # smaller 480x832 (21.3GB required vs. 13.8GB free) -- frame count
        # dominates the token count this needs to shrink. Scaled down to
        # this port's own confirmed-fitting real end-to-end test
        # configuration instead -- see docs/benchmarking.md's own note that
        # these columns are "not necessarily what fits this hardware today".
        height=320,
        width=544,
        num_frames=25,
        fps=24,
        output_path=common.output_path("ltx2_5", "", size_slug, args.task),
        # Not consumed by generate_ltx2_5.main() itself (step count comes
        # from --sampler's own sigma schedule length) -- set here purely so
        # common.run_benchmark's `getattr(args, "num_steps", None)` can
        # compute per_step_s, matching every other model's benchmark row.
        num_steps=(len(DISTILLED_SIGMA_VALUES) - 1) if sampler == "distilled" else DEV_NUM_STEPS,
    )
    if args.task == "i2v":
        ns.image_path = common.STANDARD_IMAGE_PATH
        ns.prompt = [common.STANDARD_I2V_PROMPT]
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--model_size", type=str, default="22B-distilled", choices=MODEL_SIZES)
    parser.add_argument("--task", type=str, default="t2v", choices=TASKS)
    args = parser.parse_args()

    run_args = build_args(args)
    size_slug = args.model_size.lower().replace("-", "_")
    common.run_benchmark(
        model="ltx2_5", version="", size=size_slug, task=args.task,
        main_fn=generate_ltx2_5.main, args=run_args, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
