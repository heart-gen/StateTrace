# Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 Skeleton | `Reference`, metadata validation, latent resolution, kNN backend, frozen format | 0.1.0 |
| 2 Donor-aware validation | leave-one-donor-out, random-split control, donor x state metrics, stability rule | 0.1.0 |
| 3 Calibration | confidence calibration from holdouts, reference-distance metrics, unknown thresholds | 0.2 |
| 4 Transport | frozen reference -> independent query, `transport()`, `TransportResult`, `harmonize_features()` | 0.2 |
| 5 Biological outputs | `state_abundance()`, `pseudobulk()`, HIV/OUD workflow | 0.2 |
| 6 Packaging | tutorials, PyPI release, benchmark dataset | 0.2 |
| 7 Extensions | scVI backend with per-fold refit, genetics annotation export, spatial and multi-omic references | later |

Version 0.1 is complete when a labelled AnnData reference can be validated by donor
holdout, frozen and reloaded. Version 0.2 adds everything a query needs.
