# Tutorials

The tutorials are notebooks under [`docs/tutorials/`](../docs/tutorials/), committed with their
outputs and rendered in the documentation. They run on simulated data in seconds; re-run them
with `make tutorials PY=.venv/bin/python` (needs the `docs` extra).

| Notebook | Covers |
|---|---|
| `01_validate_reference.ipynb` | `Reference.fit()` -> `leave_one_group_out()` vs `random_split()`; reproducibility categories; `state_metrics`, `group_metrics` |
| `02_freeze_and_reload.ipynb` | `freeze()` / `Reference.load()`; the files of a `.statetrace` directory, read without StateTrace; checksum verification |
| `03_transport_query.ipynb` | `validate_transport()`; statuses against the planted truth; cell categories; `state_abundance()`, `pseudobulk()`; an external mapper through `Predictions`; `compare_backends()`; a withheld population; `report()`, `to_obs()` |
| `04_two_h5ads_to_report.ipynb` | the whole workflow from a reference and a query `.h5ad` to a transport report |

Planned: the NVU -> HIV/OUD application on real data (the data and analysis live in the
benchmark repository).
