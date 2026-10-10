# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Two-sample engines (`statetrace.twosample`, `TwoSampleConfig`): the E-distance permutation
  test (native; Euclidean, as in pertpy) and Rosenbaum's cross-match test through scXMatch
  (Moeller et al., Bioinformatics 2026, doi:10.1093/bioinformatics/btag706; optional, conda
  only). `TransportConfig(two_sample=...)` runs them for every reference state at cell level
  (frozen cell-level space) and on donor profiles (held-out reference donors against query
  donors), over all assigned and in-reference cells; `route.two_sample`, `two_sample.csv` and a
  report section. Reported evidence only: `status` and `STATE_COLUMNS` are unchanged.
- Frozen cell-level space in `ExpressionSpace` (`cell_sample_per_state`, `cell_components`,
  `cell_seed`; `build_cell_space`, `project_cells`) and the held-out reference donor profiles
  (`null_profiles`); `query_profiles` exposes the standardised donor profiles behind
  `score_query`.
- `simulate_transport(spread_states=...)`: a change in spread with the mean kept (truth
  `spread`, column `spread_factor`).

### Changed
- `validate_transport` raises when a query group (donor) is also a reference group, unless
  `TransportConfig(allow_group_overlap=True)` (the explicit cell-level control); it warns when
  the reference is not frozen and records the check in `route.reference["independence"]`.
- Frozen format 4: `expression/` adds `null_profiles.npy` and the cell-level space. Format 3
  bundles still load; their two-sample rows read `unsupported` until re-frozen.
- Positioning: StateTrace orchestrates shift evidence and is not a distribution-shift test
  (README, concepts, transport guide, roadmaps).

### Planned
- Stage 7 StateTrace-Spatial: segmentation-stability diagnostics, controlled boundary-error
  benchmark design and optional CellART adapter; roadmap only, schedule and version unassigned.
- scANVI `AnnDataBackend` with per-fold latent refitting (`statetrace[scvi]`).
- Downstream-validity helpers (type I error and replication summaries for gated vs ungated analyses).
- Plotting.
- Within-donor localisation of a shifted subpopulation (per-cell displacement against held-out
  reference cells); the state-level cell tests above are done.

## [0.1.0] - 2026-10-02

First tagged release. Benchmark gate G0 (simulation: easy-cell accuracy, two-arm novelty,
truth-category recovery against scANVI and scPoli) passed on this code at `ef0560a`
(StateTrace-benchmarking `00_gates/_m/G0.md`, 2026-10-02).

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

- Per-donor latent offsets (`TransportConfig.latent_group_offset`, default on): each query
  donor's own offset (median over its states of the displacement of its state means from
  the reference centroids) is removed before the distance, novelty and latent-shift
  diagnostics, and the held-out null removes each held-out reference donor's offset the same
  way. A mapper that fits the query by surgery places each donor with its own displacement,
  which a single cohort offset leaves behind; on the simulation grid that raised the false
  out-of-reference rate in the scANVI and scPoli latents to 0.10 to 0.16 against a nominal
  0.05 while the kNN latent sat at 0.04. `route.reference["latent_offset"]` reports the
  mode, the donors with their own offset and the offset size in per-dimension reference sd.
- The held-out null is assembled from the held-out *predicted* labels of grouped validation
  (`holdout_evidence` gains `assigned_state`; the calibration groups by it), not from the
  true labels. Query cells are judged by their assigned state, and cells assigned to a state
  are the ones nearest its centroid, so a true-label null under-estimated each held-out
  donor's offset relative to the query and sat too tight: on the simulation grid the
  predeclared rule's false rate in the kNN latent rose with donor effect to 0.25 before this
  change.
- Latent offsets (cohort and per-donor) are geometric medians of the per-state
  displacements (Weiszfeld), not component-wise medians, so one shifted state does not leak
  into the offset when the number of states is even. The expression space keeps its
  gene-wise median.
- Raw and residual displacement are both reported: `latent_shift_raw` and
  `expression_shift_raw` (before donor-offset removal) beside the residual `latent_shift`
  and `expression_shift` that decide the level; per donor, `latent_offset_sd` and
  `expression_offset_rms` in `groups`, with `expression_distance_raw` and
  `expression_shift_z_raw`; the report shows raw/residual per state and the offsets per
  donor.

### Changed
- The built-in kNN path no longer attaches its vote fraction as `Predictions.support`; local
  support is computed once, after cohort-offset removal, for every backend alike.
- `simulate_transport` regenerates the query counts from the final query latent, so the cohort
  effect and every planted shift are present in expression as well as in `obsm`. Before, only
  the latent carried them, and a mapper that saw the counts never faced the planted scenario.
- Frozen reference format version 3.

Earlier in the same release (2026-09-28): renamed from AtlasRoute and re-scoped from "donor-aware reference mapping" to a
backend-independent framework for validating cellular-state transportability and gating
downstream inference.

### Added at the rename (2026-09-28)
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
