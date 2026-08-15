"""Shared harness for `benchmarks/run_*.py` -- checkpoint-directory
resolution, standardized prompts/conditioning assets, compile-vs-generation
timing, peak-HBM-per-chip measurement, and output-video filenames.

Every `docs/models/*.md` usage example assumes checkpoints live under
`./checkpoints/<repo-name>/...` (the layout a fresh `huggingface-cli
download` produces) -- that convention is what ships in the library release,
and stays the default here too. On this particular development machine the
checkpoints instead live on a mounted SSD
(`/mnt/disks/tpu_ssd/checkpoints/`), so `resolve_checkpoint_dir` also honors
a `--checkpoint_dir` CLI flag / `VIDAX_CHECKPOINT_DIR` environment variable
override -- e.g. `export VIDAX_CHECKPOINT_DIR=/mnt/disks/tpu_ssd/checkpoints`
before running any `benchmarks/run_*.py` script on this machine, with no
change to the scripts themselves and no change to what a released-library
user sees by default.

Timing methodology (matches `docs/benchmarking.md`'s Metrics section):
every `benchmarks/run_*.py` script imports the *real* `examples/generate_*.py`
module and calls its unmodified `main(args)` -- there is deliberately no
second, parallel implementation of any model's generation loop to keep in
sync. `instrument_jit()` transparently wraps `jax.jit` (every example
script's jitted step functions are created via a bare `@jax.jit`/`jax.jit(...)`
inside `main`, so patching the `jax` module's `jit` attribute before calling
`main` is sufficient -- no example-script changes needed) so that each
distinct jitted function's *first* call is timed separately from every later
call: the first-call total across every jitted function used during a run
(DiT step, VAE decode-chunk, prompt encode, ...) is reported as `compile_s`,
everything after is `generation_s`. This is a slightly broader "compile"
bucket than a single isolated step function (it also captures the VAE's own
first-chunk compile, etc.) but matches what a real cold-start caller actually
pays, and is called out here so the number's meaning stays unambiguous.
"""
import argparse
import glob
import json
import os
import time

import jax

CHECKPOINT_DIR_ENV_VAR = "VIDAX_CHECKPOINT_DIR"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "examples", "assets")
OUT_DIR = os.path.join(REPO_ROOT, "out")
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

# --- Standardized prompt/conditioning assets, shared by every model's
# benchmark script so results are comparable across model families. ---
STANDARD_T2V_PROMPT = "A majestic red panda climbing a bamboo tree in the snow, 4k"

STANDARD_IMAGE_PATH = os.path.join(ASSETS_DIR, "cat.jpg")
STANDARD_I2V_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a "
    "surfboard. The fluffy-furred feline gazes directly at the camera with "
    "a relaxed expression. Blurred beach scenery forms the background "
    "featuring crystal-clear waters, distant green hills, and a blue sky "
    "dotted with white clouds. The cat assumes a naturally relaxed posture, "
    "as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing "
    "atmosphere of the seaside."
)

STANDARD_VIDEO_PATH = os.path.join(ASSETS_DIR, "driving.mp4")
STANDARD_V2V_PROMPT = (
    "The car continues driving smoothly down the residential street, "
    "passing parked cars and houses under sunny daylight, steady forward motion."
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help=f"Root checkpoint directory (each model's own subdirectory, e.g. "
             f"'<checkpoint_dir>/Wan2.1-T2V-1.3B/...', lives under this). "
             f"Defaults to the '{CHECKPOINT_DIR_ENV_VAR}' environment variable if "
             f"set, else '{DEFAULT_CHECKPOINT_DIR}' (the layout every docs/models/*.md "
             f"usage example assumes).")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None,
        help="Overrides the benchmark script's own default --tensor_parallel_size "
             "(picked to fit this repo's reference v4-8 test machine -- see "
             "docs/hardware_and_sharding.md). Must divide num_devices.")
    parser.add_argument(
        "--sequence_parallel_size", type=int, default=None,
        help="Overrides the benchmark script's own default --sequence_parallel_size.")
    parser.add_argument(
        "--num_runs", type=int, default=5,
        help="Number of independent end-to-end runs (each with a cleared JAX "
             "compilation cache) to average metrics over.")


def resolve_checkpoint_dir(args: argparse.Namespace) -> str:
    return args.checkpoint_dir or os.environ.get(CHECKPOINT_DIR_ENV_VAR) or DEFAULT_CHECKPOINT_DIR


def find_single(pattern: str) -> str:
    """Globs `pattern` and returns the one match, erroring clearly if the
    checkpoint layout doesn't match what was expected (zero matches) or is
    ambiguous (more than one -- e.g. multiple downloaded revisions)."""
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matched '{pattern}' -- check --checkpoint_dir / "
            f"{CHECKPOINT_DIR_ENV_VAR} points at a directory containing the expected "
            f"checkpoint layout (see docs/models/*.md for what each model expects).")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous checkpoint: '{pattern}' matched {len(matches)} files: "
            f"{matches}. Pass an explicit path instead of relying on the glob default.")
    return matches[0]


def result_slug(model: str, version: str, size: str, task: str) -> str:
    """Standardized filename slug, e.g. `('cosmos', '3', 'nano', 't2v')` ->
    `cosmos3_nano_t2v`, `('cosmos', '2.5', '14b', 't2v')` -> `cosmos2_5_14b_t2v`,
    `('wan', '2.1', '1.3b', 't2v')` -> `wan2_1_1.3b_t2v`. Model and version are
    merged into one token (no separator, `.` in version -> `_`) since together
    they name one released model family; size and task stay separate tokens."""
    model_version = f"{model}{version.replace('.', '_')}" if version else model
    return "_".join(p for p in (model_version, size, task) if p)


def output_path(model: str, version: str, size: str, task: str, ext: str = "mp4") -> str:
    """Standardized base `out/` filename (see `result_slug`).
    `run_benchmark` derives each individual run's actual output path from
    this (`out/<slug>/<slug>_<run>.<ext>`) -- this function itself doesn't
    create anything on disk."""
    return os.path.join(OUT_DIR, f"{result_slug(model, version, size, task)}.{ext}")


# --- Compile-vs-generation timing ---

class _Timing:
    def __init__(self):
        self.compile_s = 0.0
        self.generation_s = 0.0
        self.num_generation_calls = 0

    def as_dict(self) -> dict:
        return {
            "compile_s": self.compile_s,
            "generation_s": self.generation_s,
            "num_generation_calls": self.num_generation_calls,
        }


def instrument_jit() -> _Timing:
    """Monkeypatches `jax.jit` so that every distinct jitted function's first
    call is timed into `.compile_s` and every later call accumulates into
    `.generation_s` -- see module docstring for the full reasoning. Must be
    called *before* the target `examples/generate_*.py` module's `main(args)`
    runs (its `@jax.jit`/`jax.jit(...)` call sites execute when `main` runs,
    not at module import time, so patching right before the `main(args)`
    call is sufficient and doesn't require reordering any imports).
    """
    timing = _Timing()
    real_jit = jax.jit

    def patched_jit(fun, *jit_args, **jit_kwargs):
        jitted = real_jit(fun, *jit_args, **jit_kwargs)
        state = {"compiled": False}

        def wrapper(*call_args, **call_kwargs):
            t0 = time.perf_counter()
            out = jitted(*call_args, **call_kwargs)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            if not state["compiled"]:
                timing.compile_s += dt
                state["compiled"] = True
            else:
                timing.generation_s += dt
                timing.num_generation_calls += 1
            return out
        return wrapper

    jax.jit = patched_jit
    return timing


def peak_hbm_per_chip_gb() -> float:
    """Max, across every local device, of that device's `peak_bytes_in_use`
    -- the binding per-chip memory watermark for whichever
    --tensor_parallel_size/--sequence_parallel_size split the run used."""
    peaks = []
    for d in jax.local_devices():
        stats = d.memory_stats()
        if stats is not None and "peak_bytes_in_use" in stats:
            peaks.append(stats["peak_bytes_in_use"])
    return max(peaks) / 1e9 if peaks else float("nan")


def clear_jax_compilation_cache() -> None:
    """Removes vidax's persistent JAX compilation cache
    (`vidax.core.sharding.configure_jax_cache`'s default directory) so a
    benchmark run always measures a genuine cold compile, not a cache hit
    left over from an earlier run/process at the same shape/sharding."""
    import shutil
    cache_dir = os.path.expanduser("~/.cache/vidax/jax")
    shutil.rmtree(cache_dir, ignore_errors=True)


def _avg(runs: list, key: str):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def run_benchmark(
    model: str, version: str, size: str, task: str, main_fn, args: argparse.Namespace,
    num_runs: int = 5,
) -> dict:
    """Runs `examples/generate_*.py`'s `main(args)` `num_runs` times (each
    under `instrument_jit()`, with a cleared JAX compilation cache -- so
    every run is an independent, comparable cold start, not just the first),
    saves each run's video to `out/<slug>/<slug>_<run>.mp4`, and writes one
    aggregated JSON result file under `benchmarks/results/` containing every
    run's raw metrics plus their average. Returns that result dict (also
    printed to stdout).
    """
    base_output_path = args.output_path
    out_dir, ext = os.path.splitext(base_output_path)
    ext = ext.lstrip(".") or "mp4"
    slug = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for i in range(1, num_runs + 1):
        clear_jax_compilation_cache()
        args.output_path = os.path.join(out_dir, f"{slug}_{i}.{ext}")
        timing = instrument_jit()
        t_wall0 = time.perf_counter()
        main_fn(args)
        wall_s = time.perf_counter() - t_wall0

        num_steps = getattr(args, "num_steps", None)
        run_result = {
            "run": i,
            "compile_s": timing.compile_s,
            "generation_s": timing.generation_s,
            "per_step_s": (timing.generation_s / num_steps) if num_steps else None,
            "peak_hbm_gb": peak_hbm_per_chip_gb(),
            "wall_s": wall_s,
            "output_path": args.output_path,
        }
        runs.append(run_result)
        print(f"  run {i}/{num_runs}: {json.dumps(run_result)}")

    result = {
        "model": model, "version": version, "size": size, "task": task,
        "jax_version": jax.__version__,
        "device_kind": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "resolution": f"{getattr(args, 'width', None)}x{getattr(args, 'height', None)}",
        "num_frames": getattr(args, "num_frames", None),
        "num_steps": getattr(args, "num_steps", None),
        "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
        "sequence_parallel_size": getattr(args, "sequence_parallel_size", None),
        "output_dir": out_dir,
        "num_runs": num_runs,
        "runs": runs,
        "avg": {
            "compile_s": _avg(runs, "compile_s"),
            "generation_s": _avg(runs, "generation_s"),
            "per_step_s": _avg(runs, "per_step_s"),
            "peak_hbm_gb": _avg(runs, "peak_hbm_gb"),
            "wall_s": _avg(runs, "wall_s"),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{result_slug(model, version, size, task)}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Benchmark result ({result_path}) ===")
    print(json.dumps(result, indent=2))
    return result
