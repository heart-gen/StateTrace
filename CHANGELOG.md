# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- `transport()` and `TransportResult`: map an independent query onto a frozen reference.
- `Calibrator`: isotonic / Platt / quantile calibration from donor holdouts.
- Novelty statistics and `supported` / `uncertain` / `out_of_reference` categories.
- `state_abundance()` and `pseudobulk()` donor-level outputs.
- `TransportReport` results directory.
- `harmonize_features()` with one-to-one ortholog tables for cross-species transport.
- scvi-tools backend with per-fold latent refitting.

## [0.1.0] - 2026-09-28

### Added
- `AtlasReference` / `Reference`: metadata validation, fixed-latent resolution (obsm key
  or PCA fallback), k-nearest-neighbour backend, `fit()`.
- Leave-one-donor-out validation (`leave_one_donor_out()`) and the random-split control
  (`random_split()`), sharing one fold engine.
- `ValidationResult` with donor x state, per-donor, state-level and confusion tables,
  a rule-based `lodo_stability` category, `to_obs()`, `to_dir()` / `from_dir()`.
- `MappingBackend` protocol, `KNNBackend`, backend registry (`get_backend`).
- Frozen reference format (`*.atlasroute` directory or zip): npy/npz/JSON/CSV only,
  SHA-256 manifest, `freeze()` / `Reference.load()`.
- `simulate_reference()` seeded generator for tests and tutorials.
- `schema` constants for every AnnData key AtlasRoute writes.
- Stub modules documenting the planned v0.2 API.

### Known limitations
- The latent space is fixed across folds (fitted with every donor present); LODO in this
  release measures classifier transfer within a fixed embedding, an optimistic bound.
