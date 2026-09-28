# AtlasRoute

[![CI](https://github.com/heart-gen/AtlasRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/heart-gen/AtlasRoute/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/atlasroute.svg)](https://pypi.org/project/atlasroute/)
[![Python](https://img.shields.io/pypi/pyversions/atlasroute.svg)](https://pypi.org/project/atlasroute/)
[![Docs](https://readthedocs.org/projects/atlasroute/badge/?version=latest)](https://atlasroute.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

AtlasRoute is a donor-aware framework for building, freezing, validating and transporting
cellular-state references across single-cell and single-nucleus datasets. Cellular states
are routinely defined in one cohort and applied to another, but standard workflows say
little about whether those states reproduce across donors, whether a query is represented
by the reference, or whether downstream disease associations rest on robust states rather
than uncertain label transfer. AtlasRoute treats donors, not cells, as the replication
unit and treats label transfer as one component of a reference-validation and transport
framework. It is tissue-, disease- and atlas-agnostic.

## Core capabilities

- **Donor-aware reference validation.** Leave-one-donor-out recovery of every state,
  with donor x state recall, per-donor balanced accuracy, state-level medians, minima
  and recovery rates, and a rule-based `lodo_stability` reading aid.
- **The optimistic control, side by side.** A stratified random split over cells
  (the conventional benchmark) is one call away, so the donor effect shows as a gap.
- **Pluggable mapping backends.** A k-nearest-neighbour classifier in a frozen latent
  space ships in 0.1; scVI/scANVI is planned. AtlasRoute does not invent an integration
  method; it validates whichever one you use.
- **Abstention.** Held-out cells below a confidence floor are predicted `unknown`
  instead of forced into the nearest state (data-driven query thresholds in 0.2).
- **Frozen references.** An immutable, checksummed, pickle-free `.atlasroute` bundle of
  latent coordinates, labels, donor statistics and validation tables.
- **Seeded synthetic references** for tests, tutorials and controlled benchmarks.

## Installation

```bash
pip install atlasroute
```

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, hypothesis, ruff | tests and linting |
| `docs` | sphinx, myst-parser, sphinx-rtd-theme | building the docs |
| `scvi` | scvi-tools (pulls in torch) | the scVI backend (0.2) |
| `all` | all of the above | |

torch is never a direct dependency; install the build that matches your machine before
adding the `scvi` extra. Python 3.11 to 3.14.

## Quickstart

```python
import atlasroute as ar
from atlasroute.datasets import simulate_reference

adata = simulate_reference(n_donors=8, n_states=5, n_cells_per_donor=200, donor_effect=0.7, seed=1)

ref = ar.Reference(adata, donor_key="donor_id", state_key="cell_state", latent_key="X_latent").fit()

res = ref.leave_one_donor_out()          # one fold per donor
print(res.summary())
res.state_metrics[["n_donors_reference", "median_donor_recall", "min_donor_recall", "lodo_stability"]]

ctrl = ref.random_split(n_splits=5)      # the cell-level, optimistic control
ctrl.overall["accuracy"] - res.overall["accuracy"]   # the donor effect

ref.freeze("my_reference.atlasroute")    # immutable bundle; ar.Reference.load(...) restores it
```

With real data, point `latent_key` at a precomputed embedding (`"X_scVI"`, `"X_pca_harmony"`)
and `donor_key` / `state_key` at your `obs` columns. Leave `latent_key=None` to compute a
PCA from counts.

Outputs of a validation run:

| Attribute | Content |
|---|---|
| `state_metrics` | one row per state: donor support, median/min/variance of donor recall, recovery rate, pooled precision/recall/F1, `lodo_stability` |
| `donor_metrics` | one row per (donor, state): recall, precision, F1, `scored` |
| `donor_summary` | one row per donor: accuracy, balanced accuracy, abstention rate |
| `confusion_matrix` | true x predicted states over every held-out cell |
| `cell_predictions` | per cell: prediction, probability, second best, margin, neighbour support |

## Documentation

Full documentation, including the concepts, the validation metrics and the frozen
reference format, lives at [atlasroute.readthedocs.io](https://atlasroute.readthedocs.io/en/latest/).

## Development

```bash
git clone https://github.com/heart-gen/AtlasRoute.git && cd AtlasRoute
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
make lint PY=.venv/bin/python
make test PY=.venv/bin/python          # add test-slow for the larger synthetic run
make docs PY=.venv/bin/python
```

The benchmark analyses and the manuscript live in separate repositories
(`AtlasRoute-benchmarking`, `atlasroute-manuscript`).

## Citation

See [`CITATION.cff`](CITATION.cff). A preprint will be linked here once available.

## Funding

This work is supported by the National Institute on Minority Health and Health
Disparities (R00 MD016964).

## Limitations

- Version 0.1 validates references only; query transport, calibration, novelty detection,
  abundance and pseudobulk outputs are planned for 0.2.
- The latent space is fixed across leave-one-donor-out folds (fitted with every donor
  present), so 0.1 measures classifier transfer within a given embedding: an optimistic
  bound on embedding transfer. Per-fold refitting is reserved for the scVI backend.
- The `lodo_stability` categories are descriptive defaults, not biological claims.
