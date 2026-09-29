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
