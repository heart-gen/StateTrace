# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Frozen biological space (`statetrace.expression`): donor x state pseudobulk profiles of the
  reference, standardised gene-wise by the reference between-donor mean and shrunk sd, a PCA
  fitted on the standardised reference rows, per-donor offset removal (each donor's median
  standardised profile over its states, applied to held-out reference donors and query donors
  alike) and a held-out reference-donor null of the standardised distance. Built at `StateReference.fit()` from counts (`expression=` takes an
  `ExpressionConfig` or `False`), frozen under `expression/` (format 3; format 2 bundles still load).
- `validate_transport` now judges **state shift** in the frozen expression space whenever the
  reference carries one and the query has its genes (`TransportConfig.shift_source`,
  `expression_layer`; `feature_map=` for ortholog-mapped queries). New state columns
  `shift_source`, `expression_shift[_level|_in_reference]`, `groups_shifted`,
  `groups_scored_expression`, `latent_shift[_level|_in_reference]`; `shift` and `shift_level`
  hold the deciding value. Per donor x state: `route.expression` and
  `expression_distance` / `expression_shift_z` / `expression_shift_p` / `shifted` in `groups`.
  The transport report states which coordinate system each conclusion came from.
- Local reference support computed by StateTrace (`novelty.local_support`): the fraction of a
  query cell's nearest reference cells, in the tool's latent, that carry the assigned state.
  Used whenever the tool reports no support of its own. Cells below
  `TransportConfig.min_support` (0.5) are `uncertain` and do not count towards a donor
  observing the state, so absence detection no longer depends on the mapper abstaining. New
  columns `n_cells_supported`, `median_support` (states), `n_supported`, `median_support`
  (groups); `route.reference["support"]` records the source.
- Labels-only predictions: when a tool gives no confidence, local support stands in for it
  (`confidence_source = "support"`); with neither confidence nor a latent, donor recovery is
  judged on counts and every state is at most `supported_with_caution`
  (`confidence_source = "none"`, new state column).
- Out-of-reference call `TransportConfig.ood_score` / `ood_rule`: the score is the centroid
  distance (previous behaviour), the kNN distance to the k-th nearest reference cell, or the
  joint score (the larger of the two standardised distances); the rule is `"z"` (above
  `ood_z`) or `"quantile"` (above the held-out 95th or 99th percentile of the state,
  `ood_quantile`). `cells` gains `distance`, `knn_distance`, `knn_novelty` and
  `joint_novelty`; `route.reference` records the score and rule applied and any fallback.
  The benchmark's withheld-state curve chooses the rule to predeclare.
- Held-out evidence per reference cell from grouped validation
  (`ValidationResult.holdout_evidence`: centroid distance, kNN distance and support against
  the training donors only), with `distance_calibration` extended by kNN, joint and support
  quantiles; both are written with the frozen reference.
- Calibrated thresholds in `validate_transport`: per-state support minimum
  (`support_calibration`, the smaller of `min_support` and the held-out 5th percentile) and
  per-state out-of-reference fraction for the `novel_population` call
  (`population_calibration`: the larger of `population_floor` and the largest fraction any
  held-out reference donor produced); `support_threshold` and `ood_threshold` state columns.

### Changed
- The built-in kNN path no longer attaches its vote fraction as `Predictions.support`; local
  support is computed once, after cohort-offset removal, for every backend alike.
- `simulate_transport` regenerates the query counts from the final query latent, so the cohort
  effect and every planted shift are present in expression as well as in `obsm`. Before, only
  the latent carried them, and a mapper that saw the counts never faced the planted scenario.
- Frozen reference format version 3.

### Planned
- scANVI `AnnDataBackend` with per-fold latent refitting (`statetrace[scvi]`).
- Downstream-validity helpers (type I error and replication summaries for gated vs ungated analyses).
- Plotting.
- Cell-level frozen-expression displacement (the within-donor complement of the pseudobulk shift).

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
- Distribution-shift diagnostics (`novelty`): standardised reference distances calibrated on
  leave-one-group-out held-out cells, centroid shift in per-dimension sd units, and removal of a
  whole-cohort latent offset before scoring.
- `compare_backends()` for backend robustness.
- `state_abundance()`, `pseudobulk()` with status-based gating; `harmonize_features()` with
  one-to-one ortholog tables; group-aware `calibration_table()` and `IsotonicCalibrator`.
- Standardised transport report (`route.report()`).
- Frozen reference format v2 (`*.statetrace`): npy/npz/JSON/CSV only, SHA-256 manifest, group and
  study keys, state geometry.
- `simulate_reference()` and `simulate_transport()` with planted stable, shifted, minority, absent
  and novel states and a truth table.

### Known limitations
- Held-out novelty calibration trades a little sensitivity to novel populations near an
  existing state for a near-nominal false out-of-reference rate on new donors.
- With the built-in kNN the latent is fixed across folds; grouped holdout is an optimistic bound
  on embedding transfer.
- Probability-only abstention is a weak novelty detector; the reference-distance category is the
  intended out-of-reference signal and needs the query in the reference latent.
