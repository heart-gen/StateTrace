# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- scANVI `AnnDataBackend` with per-fold latent refitting (`statetrace[scvi]`).
- Downstream-validity helpers (type I error and replication summaries for gated vs ungated analyses).
- Plotting and a CLI.

## [0.1.0] - 2026-09-28

Renamed from AtlasRoute and re-scoped from "donor-aware reference mapping" to a
backend-independent framework for validating cellular-state transportability and gating
downstream inference.

### Added
- `StateReference` / `Reference`: metadata validation with group and study support, fixed-latent
  resolution (obsm key or PCA fallback), kNN baseline, `fit()`, state centroids and spread.
- Grouped validation: `leave_one_group_out()` (donor), `leave_one_study_out()`, `random_split()`
  control; `ValidationResult` with group x state, per-group, state-level and confusion tables and
  the `reproducibility` category; `summarize_holdout()` for held-out tables produced by external tools.
- `Predictions`: backend-agnostic container with `from_scpoli`, `from_scanvi`, `from_symphony`,
  `from_azimuth`, `from_celltypist`, `from_obs`, `from_dataframe`, `from_arrays`.
- `MappingBackend`, `AnnDataBackend`, `CallableBackend` protocols; `KNNBackend`; registry.
- `validate_transport()` and `TransportRoute`: per-cell categories (supported / uncertain /
  out_of_reference), per-donor recovery, per-state evidence and transport status
  (supported / supported_with_caution / shifted / insufficient / absent); `TransportConfig`.
- Distribution-shift diagnostics (`novelty`): standardised reference distances, centroid shift in
  per-dimension sd units, and removal of a whole-cohort latent offset before scoring.
- `compare_backends()` for backend robustness.
- `state_abundance()`, `pseudobulk()` with status-based gating; `harmonize_features()` with
  one-to-one ortholog tables; group-aware `calibration_table()` and `IsotonicCalibrator`.
- Standardised transport report (`route.report()`).
- Frozen reference format v2 (`*.statetrace`): npy/npz/JSON/CSV only, SHA-256 manifest, group and
  study keys, state geometry.
- `simulate_reference()` and `simulate_transport()` with planted stable, shifted, minority, absent
  and novel states and a truth table.

### Known limitations
- With the built-in kNN the latent is fixed across folds; grouped holdout is an optimistic bound
  on embedding transfer.
- Probability-only abstention is a weak novelty detector; the reference-distance category is the
  intended out-of-reference signal and needs the query in the reference latent.
