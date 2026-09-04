# Contributing to `vidax`

Contributions are welcome — bug fixes, new model families, sharding/kernel
improvements, and docs. This guide covers the dev setup and the conventions the
codebase already follows.

## Dev setup

```bash
git clone https://github.com/FlyingGiraffe/vidax.git
cd vidax
pip install -e ".[dev]"          # + ".[tpu]" as well, on a TPU VM
```

`torch`, `transformers`, and `sentencepiece` are ordinary dependencies (used
only to deserialize checkpoints and tokenize text — never inside model code),
so a plain editable install is enough to import and lint everything. Running a
model end-to-end additionally needs a TPU and the relevant checkpoints; see
[`docs/models/`](docs/models/).

Before pushing:

```bash
ruff check .
python -m compileall -q src examples benchmarks
python -m build && python -m twine check dist/*     # if you touched packaging
```

## Repository layout

Standard `src`-layout. One subpackage per model family under
`src/vidax/models/`, mirrored one-for-one by `src/vidax/translator/mappings/`,
with a usage guide under `docs/models/` and a standalone script under
`examples/`. The model-agnostic primitives live in `src/vidax/core/`
(attention, RoPE, sharding) and `src/vidax/schedulers/`. See
[`docs/directory_layout.md`](docs/directory_layout.md) for the full tree and
[`docs/index.md`](docs/index.md) for the doc map.

## Conventions

- **Explicit Flax PyTrees.** Model code is plain `flax.linen` Modules with
  explicit parameter structure — no dynamic registration, no framework magic.
  The translator relies on the pytree matching the checkpoint 1:1.
- **No `torch`/`transformers` inside `vidax.models`.** They are for checkpoint
  IO and tokenization only, and live in `vidax.translator` and the per-model
  text-encoder wrappers.
- **Comments describe current behavior**, not debugging history. Postmortems
  and war stories go in [`docs/lessons/`](docs/lessons/), referenced from code
  where useful.
- **Verify against the reference.** Every ported model is checked by an exact
  checkpoint key/shape match plus a bit-exact or real end-to-end comparison
  against the upstream implementation.
- Line length target is 100 (`ruff` is currently scoped to pyflakes checks
  only — see `[tool.ruff]` in `pyproject.toml`).

## Adding a model family

1. `src/vidax/models/<family>/` — `dit.py`, `vae.py`, `configs.py` (named
   presets or a checkpoint-metadata loader), text/vision encoder wrappers as
   needed. Reuse `vidax.core` and `vidax.schedulers`; add a new scheduler only
   if the family's schedule genuinely differs.
2. `src/vidax/translator/mappings/<family>.py` — the state-dict key mapping,
   registered in `mappings/__init__.py`'s `load_torch_checkpoint_to_jax`
   dispatch (`model_type=` strings).
3. `examples/generate_<family>*.py` — one standalone script per genuinely
   distinct task/checkpoint, following the structure of the existing scripts
   (argparse CLI, mesh build, sharded sampling loop with `donate_argnums`).
4. `docs/models/<family>.md` — full CLI reference and checkpoint sources.
5. `benchmarks/run_<family>.py` — using the shared harness in
   `benchmarks/common.py`.
6. `docs/lessons/<family>_debugging.md` — what was subtle or went wrong.
7. Add the rows to `README.md`'s support table and
   [`docs/benchmarking.md`](docs/benchmarking.md); add a `CHANGELOG.md`
   `## [Unreleased]` entry.

## Pull requests

- Keep the PR focused; note any reference-mismatch caveats.
- Update the relevant docs and add a `CHANGELOG.md` `## [Unreleased]` line.
- CI runs `ruff`, `compileall`, an import smoke test, and a package build
  across Python 3.10–3.12. It does **not** run models (no TPU / no
  checkpoints in CI).

## Releases

Maintainers only — see [`docs/releasing.md`](docs/releasing.md).
