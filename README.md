# StateTrace

[![CI](https://github.com/heart-gen/StateTrace/actions/workflows/ci.yml/badge.svg)](https://github.com/heart-gen/StateTrace/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/statetrace.svg)](https://pypi.org/project/statetrace/)
[![Python](https://img.shields.io/pypi/pyversions/statetrace.svg)](https://pypi.org/project/statetrace/)
[![Docs](https://readthedocs.org/projects/statetrace/badge/?version=latest)](https://statetrace.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

StateTrace is a donor- and cohort-aware framework for determining whether cellular states
discovered in one dataset remain reproducible, supported, phenotypically preserved and suitable
for downstream inference in another. Mapping tools (scPoli, scANVI/scArches, Symphony, Azimuth,
CellTypist, or a user classifier) decide which state a cell belongs to; two-sample tests such as
[scXMatch](https://doi.org/10.1093/bioinformatics/btag706) and the E-distance decide whether two
sets of profiles differ. StateTrace takes that evidence and answers what neither was designed to
answer: **is a cellular state defined in one human cohort reproducible and transportable enough
to support inference in another?** It treats donors, not cells, as the replication unit; it
reports reproducibility across reference donors, recovery in an independent cohort, robustness
to the mapping backend, and whether the query version of a state is the same, a shifted version
of it or out of reference; and it gates which states may enter differential abundance,
pseudobulk expression or genetic analyses. It does not map cells and it is not a two-sample
test. It is tissue-, disease- and atlas-agnostic.

## Core capabilities

- **Grouped validation.** Leave-one-donor-out and leave-one-study-out holdouts with group x
  state recall, per-group balanced accuracy, and a rule-based `reproducibility` category
  (stable / moderate / fragile / insufficient); the conventional random cell split is one call
  away so the optimism of standard practice is visible as a gap.
- **Backend-agnostic.** `Predictions.from_scpoli / from_scanvi / from_symphony / from_azimuth /
  from_celltypist / from_obs` normalise any tool's labels and confidence; external mappers can
  also be refitted inside every holdout fold through `AnnDataBackend`. A kNN baseline is built in.
- **Cross-cohort transport validation.** Per state: reference reproducibility, query donors
  recovering it, uncertain and out-of-reference fractions, centroid shift, and a transport status.
- **Backend robustness.** `compare_backends` reports cell agreement, per-state concordance and
  whether the transport status agrees across mappers.
- **Shift and novelty diagnostics.** Absent, shifted and novel populations are distinguished
  by a donor x state pseudobulk distance in a frozen expression space built from the reference
  alone, calibrated against held-out reference donors (it decides the status), with
  reference-distance z-scores in the mapping latent as secondary evidence.
- **Two-sample engines (optional, reported).** `TransportConfig(two_sample=TwoSampleConfig(...))`
  runs the E-distance permutation test and, when installed, scXMatch's cross-match test for every
  state, at cell level and on donor profiles, inside the frozen space. Their p-values sit next to
  the status and never decide it: a cell-level test calls genuinely new, unshifted donors
  "different", which is a statement about cells, not about transporting the state.
- **Inference gating.** `supported`, `supported_with_caution`, `shifted`, `insufficient`,
  `absent`; `route.pseudobulk(states="supported")` and `route.state_abundance(states="supported")`.
- **Transport report.** Standardised CSV tables plus a Markdown summary of the evidence for every
  state, suitable to accompany a publication.
- **Frozen references** (checksummed, pickle-free `.statetrace` bundles) and **seeded
  simulations** of reference/query cohort pairs with planted stable, shifted, minority, absent and
  novel states.

## Installation

```bash
pip install statetrace
```

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, pytest-cov, hypothesis, ruff | tests and linting |

The optional scXMatch engine needs the `scxmatch` package, which installs only from conda
because it depends on graph-tool: `conda install scxmatch -c conda-forge -c bioconda`.
| `docs` | sphinx, myst-parser, sphinx-rtd-theme | building the docs |
| `scvi` | scvi-tools (pulls in torch) | the scANVI refit-per-fold backend (planned) |
| `all` | all of the above | |

torch is never a direct dependency; install the build that matches your machine before adding
the `scvi` extra. Python 3.11 to 3.14.

## Quickstart

```python
import statetrace as st
from statetrace.datasets import simulate_transport

ref_adata, query_adata, truth = simulate_transport(
    n_ref_groups=8, n_query_groups=6, n_states=5, n_latent=10,
    shifted_states={"S1": 9.0}, absent_states=["S4"], minority_states={"S3": 0.25}, seed=5,
)

# 1. reference: validate by donor, then freeze
ref = st.Reference(ref_adata, group_key="donor_id", state_key="cell_state", latent_key="X_latent").fit()
res = ref.leave_one_group_out()
res.state_metrics[["n_groups_reference", "median_group_recall", "group_recovery_rate", "reproducibility"]]
ref.freeze("my_reference.statetrace")

# 2. transport into an independent cohort (built-in kNN, or predictions from any tool)
route = st.validate_transport(ref, query_adata, group_key="donor_id")
route.states[["ref_reproducibility", "groups_recovered", "query_groups", "shift_level", "ood_fraction", "status"]]

preds = st.Predictions.from_scpoli(query_adata)          # after running scPoli yourself
route_scpoli = st.validate_transport(ref, query_adata, group_key="donor_id", predictions=preds)
st.compare_backends({"knn": route, "scpoli": route_scpoli}).summary()

# 3. gate downstream inference
route.supported_states
pb = route.pseudobulk(states="supported")                # donor x state pseudobulk AnnData
route.report("results/statetrace_report")
```

The transport table reads, for example:

| state | ref_reproducibility | groups_recovered | shift_level | ood_fraction | status |
|---|---|---|---|---|---|
| S0 | stable | 6/6 | low | 0.00 | supported |
| S1 | stable | 6/6 | high | 0.01 | shifted |
| S3 | insufficient | 6/6 | low | 0.00 | insufficient |
| S4 | stable | 0/6 | unknown | | absent |

With real data, point `latent_key` at a shared embedding (`"X_scVI"`) and `group_key` /
`state_key` at your `obs` columns; `study_key` enables leave-one-study-out.

## Documentation

[statetrace.readthedocs.io](https://statetrace.readthedocs.io/en/latest/): concepts, grouped
validation, transport validation and gating rules, the frozen reference format, API.

## Development

```bash
git clone https://github.com/heart-gen/StateTrace.git && cd StateTrace
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
make lint PY=.venv/bin/python
make test PY=.venv/bin/python          # add test-slow for the larger synthetic run
make docs PY=.venv/bin/python
```

Benchmarks (scPoli, scANVI/scArches and Symphony as mapping backends, scConform as the
uncertainty benchmark, scXMatch and the E-distance as the distribution-shift benchmark, with
gated release criteria) and the manuscript live in separate repositories (`StateTrace-benchmarking`,
`statetrace-manuscript`).

## Citation

See [`CITATION.cff`](CITATION.cff). A preprint will be linked here once available.

## Funding

This work is supported by the National Institute on Minority Health and Health Disparities
(R00 MD016964).

## Limitations

- With the built-in kNN the latent space is fixed across holdout folds (fitted with every donor
  present), an optimistic bound on embedding transfer; per-fold refitting needs an
  `AnnDataBackend` (scANVI planned).
- Shift and novelty diagnostics need the query in the reference latent space; with labels-only
  predictions from an external tool, gating uses group support and confidence alone.
- The `reproducibility` and `status` rules are descriptive defaults to be predeclared, not
  biological claims.
