# AtlasRoute: working rules for this repository

## What AtlasRoute is (and is not)

- A **validation, calibration, abstention and reporting layer** around cell-state mapping.
  It is not an integration algorithm and not a differential-expression method; mapping
  engines are backends behind `atlasroute.backends.base.MappingBackend`.
- **Donors, not cells, are the replication unit.** Every metric that reaches a user must
  exist at donor level; cell-level accuracy is reported separately, never as a proxy.
- **Freeze before query.** A reference is written once (`freeze()`), checksummed, and never
  redefined by a query.
- **Abstention is a result.** `unknown` is a valid assignment, never a failure.
- Disease-, tissue- and atlas-agnostic. Project-specific keys (NVU: `donor_id`,
  `state_cluster`, `broad_cell_class`, `X_scVI`) live in the benchmark repository's
  configs, never in this package.

## House style (matches ../IsoGraph and ../rfmix_reader)

- `src/atlasroute/` layout; PEP 621 `[project]` metadata with the poetry-core backend;
  version lives only in `pyproject.toml` and is read through `importlib.metadata`.
- Every dependency has a floor and a major cap. `torch` and `scvi-tools` are never hard
  dependencies; optional features raise `ImportError("... pip install 'atlasroute[extra]'")`.
- `__init__.py` exports lazily (`_lazy` dict + `__getattr__` + `TYPE_CHECKING` block);
  `import atlasroute` must not import scanpy, matplotlib, sklearn or torch.
- `from __future__ import annotations`; `X | None` and `list[str]` typing; NumPy-style
  docstrings on every public object; keyword-only arguments on public entry points.
- `logger = logging.getLogger(__name__)`; no `print`. `warnings.warn(..., stacklevel=2)`
  for user-facing hazards.
- Dataclasses for configs and results; every AnnData key written goes through `schema.py`.
- ruff: line length 100, rules `E F I B UP N C4`, ignore `N803 N806 E501`.
- Tests: pytest + hypothesis; `slow` marker behind `--run-slow`; `tests/test_imports.py`
  runs import hygiene in a subprocess (silent import, works with torch and scvi blocked).
- Synthetic data comes from `atlasroute.datasets.simulate_reference`; no downloads in tests.

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
make lint PY=.venv/bin/python
make test PY=.venv/bin/python
make test-slow PY=.venv/bin/python
make docs PY=.venv/bin/python
```

## Do not

- Put benchmark analyses, notebooks with real data or manuscript material here; they belong
  in `../../projects/AtlasRoute-benchmarking/` and `../../manuscripts/atlasroute-manuscript/`.
- Refit the latent space per fold in v0.1 (`refit_latent=True` is reserved for the scVI backend).
- Pickle anything in the frozen reference format.
- Add a CLI, create GitHub remotes, or bump the version outside `make release`.
- Add trailer lines (Co-Authored-By or similar) to commit messages. Subject plus optional body only.

## Related repositories

- Benchmark and application analyses: `/home/kynon/Documents/github/projects/AtlasRoute-benchmarking/`
- Manuscript (Manubot, Genome Biology): `/home/kynon/Documents/github/manuscripts/atlasroute-manuscript/`
- First application, the NVU reference: `/home/kynon/Documents/github/projects/nvu-neuroimmune-reference/`
