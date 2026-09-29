# Roadmap

Releases are gated by the benchmark repository: StateTrace advances to the next layer only when
it is comparable to or better than the main comparators on the previous layer's task.

| Version | Content | Benchmark gate |
|---|---|---|
| 0.1 | `StateReference`, grouped holdout (donor, study), random-split control, reproducibility categories, frozen format, kNN baseline | G0 simulation parity |
| 0.2 | `Predictions` adapters (scPoli, scANVI, Symphony, Azimuth, CellTypist), `validate_transport`, shift diagnostics, inference gating, `compare_backends`, abundance and pseudobulk, transport report, feature harmonisation | G1 semi-synthetic perturbations on real data |
| 0.3 | scANVI `AnnDataBackend` with per-fold refit, group-aware calibration tables, downstream-validity helpers (type I error / replication summaries) | G2 public cross-cohort (PBMC, HLCA) |
| 0.4 | hierarchical states, plotting, CLI | G3 brain and G4 application |
| later | genetics annotation export, spatial and multi-omic references | — |

## Release criteria for v1.0

Measured before the stable release and tracked in the benchmark repository's `UTILITY.md`.

| KPI | Target |
|---|---:|
| Core code test coverage | > 80 % |
| Core workflow reproducible from the tutorial | 100 % |
| Installation | one command (PyPI, later conda-forge) |
| Tutorial runtime on the standard dataset | minutes on a workstation |
| Input conventions | AnnData / scverse-compatible |
| Mapping backends consumed through `Predictions` | >= 3 (scPoli, scANVI, Symphony) |
| Public functions fully documented | ~100 % |
| Example independent datasets | >= 3 |
| Automated CI | yes |
| Core workflow (`validate` -> `validate_transport` -> `report`) | <= 15 public API calls, no project-specific code |
| Usability | a non-author produces and interprets a transport report from two h5ads and the docs |
