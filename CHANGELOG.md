# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-19

StateTrace 0.2.0 expands reference validation and transport diagnostics
with calibrated out-of-reference detection, mapper-independent reference
reproducibility diagnostics, study-aware novelty calibration, and
complementary cell- and donor-level distribution-shift tests.

### Highlights

- **Improved reference validation.** Reproducibility can now be evaluated
  in a frozen expression space, independently of mapper-specific latent
  coordinates. Study-aware holdout calibration accounts for variation
  across represented studies, with fallback for insufficient evidence.

- **Calibrated novelty detection.** kNN distance replaces centroid
  distance as the default OOD statistic following a prespecified
  simulation-based comparison. Support-deficit and combined
  kNN-support diagnostics are available for sensitivity analysis.

- **Complementary shift evidence.** E-distance permutation testing and
  optional scXMatch cross-match testing assess cell-level and
  donor-profile distribution differences without changing state
  classifications.

- **Reference-aware caution thresholds.** Recovery and uncertainty
  warnings now use held-out reference evidence rather than fixed
  thresholds where sufficient calibration data are available.

- **Reproducibility and compatibility.** Frozen-reference format 4 adds
  cell-level expression coordinates and donor profiles. Format 3
  references remain loadable. Configuration provenance and fallback
  decisions are recorded.

### Benchmark status

Benchmark G1 was evaluated using StateTrace commit `07def17`
(StateTrace-benchmarking `00_gates/_m/G1.md`, 2026-10-16).

The benchmark passed insufficient-state recall (1.000) and
cross-backend status agreement (0.843). Withheld-state detection
did not meet its registered criterion (AUROC 0.652 versus the
registered comparison value of 0.722).

The withheld-state detection failure remains accepted as a
documented limitation, not a successful benchmark outcome.

Release defaults were selected using simulator-based criteria rather
than optimized against G1. Inspection of G1 identified an additional
mixed-neighborhood failure mode, motivating support-based diagnostics.
Those diagnostics did not satisfy the registered replacement rule and
are retained as post hoc sensitivity tools rather than new defaults.

The G1 results describe the evaluated commit and do not independently
validate additions made after that evaluation.

### Added
- Out-of-reference scores that see a population the reference lacks when it sits *between*
  reference states rather than away from them (`TransportConfig.ood_score`): `"support"`, the
  local-support deficit standardised against the held-out support of the assigned state
  (`novelty.support_novelty`), and `"knn_support"`, the larger of the standardised kNN
  distance and the support deficit. Benchmark G1 (StateTrace-benchmarking) withheld CD4 T
  cells from a PBMC reference that keeps Tregs and CD8: every mapper absorbed them into
  Treg or CD8 with kNN and centroid distances no larger than those states' own held-out
  cells (AUROC at chance), while their mixed neighbourhoods were visible; the frozen
  expression space showed the same (CD4 cells sit closer to the Treg centroid than held-out
  Tregs do), so no distance score, in any space, sees such a state. The score the benchmark
  predeclares is chosen on the simulator by a rule fixed before it runs, as `knn` was.
  `distance_calibration` gains `support_mean_holdout`, `support_sd_holdout`,
  `supportz_q95/q99_holdout` and `knn_support_q95/q99_holdout`; `cells` gains
  `support_novelty`, `knn_support_novelty` and `ood_novelty` (the standardised score behind
  the out-of-reference call, so a benchmark ranks what the rule used). Frozen references
  without the new calibration columns fall back to the centroid z rule for the new scores,
  and `route.reference["ood_fallback"]` says so.
- `simulate_transport(novel_chimera=...)`: a withheld state that takes a reference state's
  latent centroid exactly and, on a few seeded genes, a third state's expression level; the
  case of a population that shares its neighbour's programs but differs in marker genes.
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
- `LODOConfig(calibration_unit="study")`: the novelty null of a leave-one-group-out run
  (held-out distances, their quantiles, the population thresholds) is built with one study
  held out at a time, each held-out donor's latent offset removed as for a query donor, so a
  query from a new study is judged against between-study differences an offset does not
  remove. States without enough study-held-out evidence keep the group null;
  `distance_calibration["calibration_unit"]` and `route.reference["novelty_calibration_unit"]`
  record the choice. Reproducibility still comes from the group holdout. Default `"auto"`
  (see Changed).
- `LODOConfig(reproducibility_space="expression")`: the `reproducibility` category is judged
  by the grouped holdout repeated with the kNN on every reference cell in the frozen
  cell-level expression space (`expression.cell_coordinates`), built from counts alone, so
  the same reference cells get the same call whichever mapper's latent they sit in. The
  latent call is kept as `reproducibility_latent`; the expression holdout adds
  `reproducibility_expression` and `*_expression` metrics. Transport reports
  `ref_reproducibility_latent` and `ref_reproducibility_expression` beside the deciding
  `ref_reproducibility`, and `route.reference["reproducibility_space"]`. Default `"auto"`
  (see Changed).
- Elongated states and a withheld state between two states in the simulator:
  `simulate_reference` / `simulate_transport(state_anisotropy=, anisotropic_dims=)` stretch
  each state's noise along its own random directions (drawn from a separate generator, so
  every existing seed is unchanged), and `simulate_transport(novel_between={novel: (a, b,
  t)})` places a withheld state a fraction `t` of the way from `a` to `b`. With elongated
  absorbing states the centroid score loses an absorbed population that the kNN distance
  still finds (`tests/test_transport.py`: AUROC 0.66-0.88 centroid against 0.89-0.98 kNN over
  three seeds at `t=0.35`, stretch 3); the round-state grid behind the predeclared
  `ood_score` never showed this.
- `TransportConfig(caution_calibration="holdout")`: the recovery and uncertainty caution
  triggers compare the query's donors with held-out reference donors, both measured from
  StateTrace's own evidence in the mapper's latent (`novelty.group_evidence_table`), instead
  of fixed cut-offs (recovery below 0.8, uncertain fraction above 0.2): caution when the query
  is worse with one-sided `p < caution_alpha` (Fisher exact for recovery, Mann-Whitney for
  per-donor uncertain fractions) and by more than `caution_margin`. States seen in too few
  held-out donors keep the cut-offs (`caution_basis`). New state columns `caution_basis`,
  `ref_groups_observed`, `cmp_recovery_rate`, `ref_recovery_rate`, `recovery_p`,
  `cmp_uncertain_fraction`, `ref_uncertain_fraction`, `uncertain_p` and, in every mode,
  `caution_reasons` (also in the report); `route.reference["caution_calibration"]`. Default
  `"holdout"` (see Changed).

### Changed
- Distribution name on PyPI is `statetrace-sc`: PyPI refuses `statetrace` as too similar to
  the unrelated `state-trace` project (separators are ignored in the comparison). The import
  name, the repository and the documentation keep `statetrace`; `pip install statetrace-sc`.
- **Defaults that change results on the same data** (benchmark gate G1 failed on 0.1.0: the
  status differed across mappers and the centroid novelty score lost withheld populations;
  every fix was chosen on the simulator, never on the G1 data):
  - `LODOConfig.reproducibility_space` defaults to `"auto"`: the frozen cell-level expression
    space when the reference has it, else the latent. The reproducibility call no longer
    depends on which mapper's latent the reference sits in.
  - `LODOConfig.calibration_unit` defaults to `"auto"`: the study-held-out null when the
    reference has at least two studies, else the donor-held-out null.
  - `TransportConfig.caution_calibration` defaults to `"holdout"`: caution when the query's
    donors do measurably worse than held-out reference donors, not when a noisy rate crosses
    a fixed cut-off.
  - `ValidationResult.config` records the resolved values. `reproducibility_space="latent"`,
    `calibration_unit="group"`, `caution_calibration="fixed"` and `ood_score="centroid"`
    reproduce 0.1.0.
  - `TransportConfig.ood_score` defaults to `"knn"` (was `"centroid"`), set from the
    withheld-state curve on elongated states (StateTrace-benchmarking S1.3b,
    `withheld_geometry_curve.csv`; 240 runs, 5 seeds, StateTrace `93e7261`) by a rule fixed
    on 2026-10-12, before any result of that curve. Over the G1 mappers (scANVI,
    scPoli, Symphony): (1) primary = mean AUROC on elongated states with the withheld state
    between two (stretch 3, t = 0.2, 0.35, 0.5); (2) on round states at the default
    placement (the G0 geometry), mean AUROC no more than 0.02 below centroid and no mapper
    more than 0.05 below; (3) false rate of the score's q95 rule no more than 0.02 above
    centroid's; (4) among `knn` and `joint` meeting (2) and (3), the higher primary wins if it
    beats centroid by at least 0.02, `joint` when the two are within 0.01; else `centroid`.
    Result (mean AUROC; centroid / knn / joint): elongated, between 0.680 / 0.859 / 0.812;
    round 0.949 / 0.949 / 0.955, per mapper knn within 0.006 of centroid; q95 false rate
    0.048 / 0.042 / 0.047; q95 power on elongated 0.16 / 0.49 / 0.42. Both candidates were
    eligible and knn led joint by 0.047, so `knn`. Without held-out calibration the centroid
    z rule still applies (`route.reference["ood_fallback"]`).
- `validate_transport` raises when a query group (donor) is also a reference group, unless
  `TransportConfig(allow_group_overlap=True)` (the explicit cell-level control); it warns when
  the reference is not frozen and records the check in `route.reference["independence"]`.
- Frozen format 4: `expression/` adds `null_profiles.npy` and the cell-level space. Format 3
  bundles still load; their two-sample rows read `unsupported` until re-frozen.
- Positioning: StateTrace orchestrates shift evidence and is not a distribution-shift test
  (README, concepts, transport guide, roadmaps).
- matplotlib is no longer a core dependency: the new `plot` extra carries it for
  `statetrace.plotting`, which says so when it is missing. scvi-tools stays in the `scvi` extra.

### Fixed
- Python 3.11: `isinstance(backend, MappingBackend)` raised on an unfitted `KNNBackend`,
  because 3.11 checks protocol members with `hasattr`, which runs the `classes_` property.
  The backend protocols now look members up statically on 3.11, as 3.12 does.
- Expression space: when LAPACK's divide-and-conquer SVD (gesdd) does not converge, the PCA
  retries with the QR-iteration driver (gesvd). Which matrices fail depends on the CPU's BLAS
  kernels: a 210k-cell PBMC reference failed on one node type and not on another.
- `Reference.fit` no longer turns that numerical failure into the "expression space not built"
  warning (`LinAlgError` is a `ValueError`); it used to fall back to the mapping latent and so
  change the shift and reproducibility diagnostics with only a warning.

### Known limitations

**Absorbed novel populations.** StateTrace may fail to identify a withheld population when
mapping places its cells within the distance distribution of existing reference states.

In benchmark G1, withheld CD4 T cells were absorbed into retained Treg and CD8 states. The
evaluated distance-based diagnostics did not reliably distinguish these cells from held-out
reference cells. The registered withheld-state detection criterion was not met
(AUROC 0.652 versus 0.722).

Mixed-neighborhood support provided additional information about this failure mode. However,
the support-deficit and combined kNN-support scores did not satisfy their prespecified
replacement criteria. They are retained as sensitivity diagnostics; kNN distance
remains the default OOD score.

**Reference completeness.** A low out-of-reference fraction does not establish that all
biological populations are represented. Novelty estimates should be interpreted alongside
neighborhood support, reference recovery, and uncertainty diagnostics.

**Statistical interpretation.** Cell-level two-sample tests identify differences in sampled
cell distributions but do not by themselves establish donor-level replication.
Donor-profile evidence is reported separately. Neither test changes StateTrace status calls.

**Calibration and transportability.** Holdout calibration describes variation represented
in the reference. Performance may degrade for new studies, populations, or technical
conditions not adequately represented by those holdouts. Insufficient calibration evidence
triggers documented fallback behavior.

### Planned
- Stage 6 (0.5.0) StateTrace-Spatial: segmentation-stability diagnostics, controlled boundary-error
  benchmark design and optional CellART adapter; roadmap only, schedule unassigned, not a
  submission requirement.
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
