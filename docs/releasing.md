# Releasing `vidax`

This is the runbook for cutting a version, tagging it, and (optionally)
publishing to PyPI. It also explains how to handle the two kinds of change that
land between releases.

## Versioning scheme

`vidax` follows [Semantic Versioning](https://semver.org). While the project is
pre-1.0 (`0.MINOR.PATCH`):

| Change | Bump | Example |
| --- | --- | --- |
| New model family / new engine capability / breaking API change | **MINOR** | `0.1.0` → `0.2.0` |
| Bug fix, dependency bump, doc edit, benchmark-number refresh | **PATCH** | `0.1.0` → `0.1.1` |

The version lives in **one place**: `__version__` in
[`src/vidax/__init__.py`](../src/vidax/__init__.py). `pyproject.toml` reads it
from there (`[tool.hatch.version]`), so there is nothing else to bump.

## The two kinds of between-release change

### 1. "Next version" changes — adding/altering code

A new model, a new flag, an API change. These get their own release:

1. Develop on a branch, merge to `main`.
2. Add an entry under `## [Unreleased]` in
   [`CHANGELOG.md`](../CHANGELOG.md) as you go.
3. When ready, follow **Cutting a release** below with a MINOR bump.

### 2. "Fold into current release" changes — nothing code-facing

Refreshed benchmark numbers, corrected README links, typo fixes, expanded
docs. These do **not** need their own version:

- Commit them straight to `main`. They are live on GitHub immediately — that
  is the whole distribution channel for docs.
- Note them under `## [Unreleased]` in `CHANGELOG.md`.
- They reach PyPI only when the *next* release of any kind goes out. A PyPI
  release lagging `main` by a handful of doc commits is normal and expected —
  do not cut a patch release just to sync docs.
- If a doc fix is genuinely urgent for PyPI users (e.g. a wrong install
  command), bundle it into a PATCH release.

**Never move a tag that has been pushed.** `v0.1.0` always points at the exact
tree that was released as `0.1.0`. Corrections go into the next version.

## Cutting a release

Prerequisites: a clean `main`, `pip install -e ".[dev]"` (gives `build` +
`twine`), and — for the manual PyPI upload path — a PyPI account with an API
token in `~/.pypirc` or `TWINE_PASSWORD`.

1. **Changelog.** Move the `## [Unreleased]` items under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading. Update the compare/tag links at the
   bottom of the file. Leave a fresh empty `## [Unreleased]` section.
2. **Bump the version.** Edit `__version__` in `src/vidax/__init__.py`.
3. **Commit & tag.**
   ```bash
   git commit -am "Release X.Y.Z"
   git tag -a vX.Y.Z -m "vidax X.Y.Z"
   git push origin main --follow-tags
   ```
4. **Build & check.**
   ```bash
   rm -rf dist
   python -m build
   python -m twine check dist/*
   ```
5. **Publish (manual).** Dry-run on TestPyPI first, then the real index:
   ```bash
   python -m twine upload -r testpypi dist/*
   python -m twine upload dist/*
   ```
6. **GitHub Release.** Create a release from the `vX.Y.Z` tag and paste that
   version's `CHANGELOG.md` section as the notes.

## Automated publishing via `.github/workflows/publish.yml`

The workflow builds the package and uploads it to PyPI via **PyPI Trusted
Publishing** (OIDC — no token stored in the repo). Set it up once:

1. On PyPI: **Your projects → vidax → Publishing → Add a new pending
   publisher** — GitHub, owner `FlyingGiraffe`, repo `vidax`, workflow
   `publish.yml`, environment `pypi`. (For a project that doesn't exist yet,
   this is a *pending* publisher — the first successful run creates it.)
2. On GitHub: create an Environment named `pypi` (Settings → Environments),
   optionally with a required reviewer for a manual approval gate.
3. Run it: **Actions → Publish to PyPI → Run workflow**. This replaces step 5
   above.
4. Optional — to also publish automatically on every `v*` tag, uncomment the
   `push: tags` trigger at the top of `publish.yml`.

Until step 1–2 are done, use the **Publish (manual)** `twine` commands in step
5; the workflow's `publish` job needs the `pypi` environment to succeed.

## First-time PyPI registration

The name `vidax` is currently unregistered. The first `twine upload` (or the
first successful Trusted-Publishing run) registers it. Consider uploading
`0.1.0` to TestPyPI first to confirm the metadata renders correctly:

```bash
python -m twine upload -r testpypi dist/*
pip install -i https://test.pypi.org/simple/ --no-deps vidax
```
