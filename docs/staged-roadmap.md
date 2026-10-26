# StateTrace staged roadmap

Core software stages, each closed by a benchmark gate, on the twelve-week schedule that ends with a
submission-ready manuscript on 2026-12-21. The benchmark repository owns the gate definitions
(`GATES.md`, `UTILITY.md`, `configs/gates.yaml`, `configs/utility_gate.yaml`) and the weekly
timeline (`TIMELINE.md`); this page tracks what the *software* must deliver for each gate,
what evidence closes the stage, and which inputs from the PI it waits on. Update the summary
table and the milestone log when a stage changes state.

Stage 6 is the planned spatial extension (StateTrace-Spatial, version 0.5.0) with its own gate and
an unassigned schedule; it is not required for the manuscript submission. The version cited at
submission is the Stage 5 manuscript release tag (0.4.x), not the 1.0.0 stable release that closes
Stage 7, so Stages 6 and 7 can follow the submission without moving it.

## Summary

| Stage | Status | Weeks | Deliverables | Promotion gate | Required evidence | Version | Last commit |
|---|---|---|---|---|---|---|---|
| 0 | complete | before 1 | Package, grouped validation (donor, study), reproducibility categories, `Predictions`, kNN baseline, frozen format, transport route and gating, report, feature harmonisation | Fresh install, silent import with torch and scvi blocked, full test suite, docs build | CI log, `tests/test_imports.py`, docs build | 0.1.0 (tagged 2026-10-02 with Stage 1) | `8dcf9bb` |
| 1 | G0 passed 2026-10-02 | 1 - 2 | Two coordinate systems: frozen expression space with per-donor offsets (done); identity-loss status; local support in the tool's latent (done); novelty scores and calibrated thresholds (done); corrected simulator (done); tag v0.1.0 | **G0** on the full simulation grid: (a) cell accuracy within 2 points of scANVI / scPoli, (b) two arms: the predeclared out-of-reference rule beats the mapper's uncertainty at matched donor-calibrated false rate, and the absorbing state shows a high frozen-expression shift, (c) >= 90 % planted statuses correct, shift evidence scored separately | `00_gates/_m/G0.md` with both SHAs; grid summary CSV; smoke and rerun reports | 0.1.0 tag, 0.1.1 fixes | `841f1dc` |
| 2 | G1 (d) FAIL accepted 2026-10-18 | 3 - 4 | Backend coverage: Symphony adapter (R -> CSV -> `from_symphony`, done), study-unit novelty null, expression-space reproducibility, held-out caution calibration, kNN out-of-reference score, support-deficit score (sensitivity only); scANVI `AnnDataBackend` with per-fold refit and the scConform / Azimuth bridges move to Stage 3 | **G1** semi-synthetic perturbations on Stephenson et al. 2021 PBMC; **U1** decided (does the problem exist) | `G1.md` (FAIL on withheld-state detection, 0.652 vs 0.722; `insufficient` and backend agreement pass; the failure accepted as a stated limitation, benchmark `GATES.md` 2026-10-18), `U1.md` pending | 0.2.0 | `9b6e0a7` |
| 3 | planned | 4 - 6 | Real-data robustness: study-level holdout on real studies; test of the per-donor offset assumption under a real platform effect; group-aware calibration tables; feature-retention reporting | **G2** PBMC and HLCA, **U2** decided (beats simple baselines at predicting replication); **G3** brain | `G2.md`, `U2.md`, `G3.md`; scIB reference table; calibration figure | 0.2.x, 0.3.0 | |
| 4 | planned | 7 - 8 | Inference validity: downstream-validity helpers (type I error under permuted null, FDR, power, effect-size bias, replication summaries); gated versus ungated pseudobulk and differential-abundance pipeline | **G4** NVU -> SCORCH application and validity layer; **U3** decided (supported states replicate, gating controls type I error) | `G4.md`, `U3.md`; error-rate and replication tables | 0.3.0 | |
| 5 | planned | 9 | Generalisation: ortholog `FeatureMap` on real BioMart tables; retention in the report; sensitivity analyses (thresholds, reduced features); optional cell-level frozen-expression displacement; **manuscript release tag, the version cited at submission** | **G5** organoid and cross-species application stages; **U4** decided (>= 3 systems, >= 3 backends) | `G5.md`, `U4.md`; frozen result set; manuscript release tag | 0.4.0 | |
| 6 | planned | unassigned; not a submission requirement | StateTrace-Spatial: boundary stability, spatial correspondence, transcript reassignment, donor/state summaries; optional CellART adapter | **G6-Spatial** (proposed): distinguish biological shift from segmentation-induced artificial novelty, with reference sensitivity assessed separately | Corruption benchmark, matched-cell and correspondence-component tables, donor/state effect sizes, independent validation, provenance and gate report | 0.5.0 | |
| 7 | planned | 10 - 12 for U5 and the submission release; 1.0.0 after Stage 6 | Usability and stable release: tutorial (two h5ads to a transport report), docs and API fixes from the usability test, PyPI and conda recipe, Zenodo DOI, clean-environment reproduction of the submission version; the stable tag once Stage 6 has closed | **U5** a non-author produces and interprets a transport report from the docs; then one outside collaborator; v1.0 release criteria ({doc}`roadmap`) | `U5.md`; tutorial runtime; `env/environment.lock.txt`; Zenodo record | 1.0.0 | |

## Promotion rule

A stage is marked `complete` only after:

1. Its gate passes on the criteria predeclared in the benchmark repository, never on criteria
   adjusted after the run.
2. The gate report (`00_gates/_m/G<n>.md` or `U<n>.md`) records the software SHA, the benchmark
   SHA and the result-set path, and is copied to the manuscript's `drafts/analysis-summaries/`.
3. The version named in the table is tagged in this repository and the `CHANGELOG` entry closed.
4. New public objects, config fields, AnnData keys (`schema.py`) and frozen-format changes are
   documented in `docs/`.
5. The milestone log below has a row for it.

A failed readiness gate changes the software and costs about a week; the schedule absorbs one.
A failed utility gate changes what the paper claims and does not stop the software work that
the next readiness gate needs.

## Development policy

- Benchmark-first: every adapter and every diagnostic runs through the same simulation grid
  before it touches real data.
- One complexity layer at a time: a stage adds either a comparator or a diagnostic, not both.
- StateTrace never competes on cell-level accuracy. Every comparison against scPoli, scANVI /
  scArches, Symphony, scConform and Azimuth is on a question they do not answer: donor-level
  reproducibility, transport into an independent cohort, shift of a state's molecular phenotype
  beyond the reference's between-donor range, and whether a state may enter a donor-level test.
- StateTrace is not a distribution-shift test. scXMatch (Moeller et al., *Bioinformatics* 2026,
  doi:10.1093/bioinformatics/btag706) and the E-distance answer whether two sets of profiles
  differ; they run as optional engines inside the frozen space (`statetrace.twosample`) and as
  the shift comparators in the benchmark (reported at G1, gated at G2, decided 2026-10-09), and
  never decide a status.
- Two coordinate systems, always: state identity from the mapping space, state shift from the
  frozen expression space. A diagnostic that needs the query present in the latent it is judged
  in is not a shift diagnostic.
- Freeze before query. Thresholds are predeclared in the benchmark configs and signed off before
  the deciding stage runs.
- No CLI, no pickle, no GitHub remotes until the release stage; commits carry no trailers.

## Current status

Stage 0 is complete. Stage 1 passed G0 on 2026-10-02 (third full grid run, software
`ef0560a`, tag v0.1.0): both mappers absorb a planted state shift in their own latent while
the frozen expression space discriminates the absorbing state from the intact ones; the
predeclared out-of-reference rule holds its nominal false rate once the null is built the way
the query is (assigned labels, per-donor geometric-median offsets). Left open from Stage 1: the
identity-loss status (PI decision), the scConform and Symphony R bridges (carried to G1), and
the optional Benchmark A vs B optimism grid. Stage 2 is unlocked and its analysis code exists
(benchmark `statetrace_benchmark/perturbation/`, 2026-10-02: prepare, donor / study / random
holdout with the mappers refitted per fold, withheld state, donor-support restriction with
sensitivity thresholds, aggregation and the G1 evaluator). The stage is registered on
Stephenson et al. 2021 PBMC (GSE132044 has two donors), the G1 criterion reads fewer than
three donors and the comparator set is fixed (Symphony in G1; scConform and Azimuth at G2).
Next: the 7.2 GB download, the Symphony adapter, the run.

2026-10-09: scXMatch was published (2026-10-03) and overlaps the expression-shift arm, so the
novelty claim moved from "detecting a shifted state" to donor- and cohort-aware
transportability. The two-sample engines (E-distance native; scXMatch when installed from
conda) entered the package as reported evidence (frozen format 4 carries the cell-level space
and the held-out donor profiles they need), `validate_transport` refuses a query that shares a
donor with the reference, and `simulate_transport` gained a spread-only scenario. The benchmark
adds scXMatch and the E-distance as shift comparators (G1 reported, G2 gated) with monotonicity
and subsampling-robustness metrics.

2026-10-18: G1 was evaluated twice on real data (Stephenson et al. 2021 PBMC). The first run (0.1.0)
failed on detection and on backend agreement; the defaults changed in 0.2.0 (study-unit novelty
null, expression-space reproducibility, held-out caution, kNN out-of-reference score, every choice
made on the simulator) brought agreement to 0.843 and `insufficient` recall to 1.000, but
withheld-state detection stayed at 0.652 against 0.722: a population the mappers place between two
reference states at within-state distances is invisible to every distance score, and the
support-deficit score (`ood_score="support"`) did not meet its pre-registered replacement rule. The
PI accepted the failure as a stated limitation rather than change the software for the gate; the
benchmark's stages 03 and 04 open on this code and state the limitation. The benchmark's
predeclared real-data out-of-reference rule fired on the same records (held-out studies of intact
states exceed 10 % out-of-reference under q95), so real-data stages use the held-out 99th percentile.

2026-10-25: G2 was evaluated on 0.2.1 (`53f7af0`) after both real-data stages were rerun from
prepare. Stage 04 (HLCA cross-study) passed every check: backend agreement 0.870, the scIB paradox
present, StateTrace's donor-level shift AUROC 0.583 against the best two-sample engine's 0.533,
technical false calls 0.033, monotonicity 0.85. Stage 03 (Ren et al. 2021, 5' reference against 3'
query cohorts) read 0.778 exact agreement against 0.80 (0.833 within the 0.05 sd shift tolerance,
reported): four of 18 pairs depend on the mapper, three at the shift cutoffs through each mapper's
assignment composition and one rare state read `insufficient` or `absent`. The PI accepted the
failure as a stated limitation; the 0.2.1 status-order correction stands and no further change is
made for the gate. Two directions are recorded for the application stages (0.3): the shift per query
donor with the status read from the donor-level interval (a data-driven boundary flag in place of
the fixed tolerance), and a consensus transport status across backends with the agreement reported
as its reliability. Neither targets G2, whose criterion is per-backend agreement.

Inputs this software waits on from the benchmark side are listed there (`INPUTS_NEEDED.md`,
`TIMELINE.md`); the ones that gate software work are the comparator sign-offs (done), the
reference objects of the application stages, the replication definitions for U2 / U3, and the
testers for U5.

## Stage 1: two coordinate systems and G0

### Objective

Close the G0 blocker for good and pass G0 on the full grid, so that the software enters real
data with a shift diagnostic that does not depend on the mapper's latent.

### Deliverables

- Frozen expression space (`statetrace.expression`): donor x state pseudobulks, reference
  standardisation, per-donor offsets, held-out-donor null, frozen under `expression/`. Done.
- Corrected simulator: query counts carry the cohort effect and planted shifts. Done.
- Identity-loss handling: a state whose cells the mapper reassigns wholesale reads `absent`
  today while a neighbour shows a high expression shift with an out-of-reference population.
  Implement whichever design the PI chooses.
- Local support computed inside `validate_transport` from a kNN in the tool's latent, so
  absence detection does not depend on the backend abstaining. Done (`min_support`).
- Novelty decision rule: the standardised distance ranks novel cells well (AUROC 0.86 at the
  hard cell) but the 3-sd rule flags under a tenth of them. The withheld-state curve over
  separation (`simulation.withheld_state`) chose the held-out 95th-percentile rule,
  predeclared 2026-09-30 (`ood_score` centroid; the joint centroid + kNN score is the
  amendment candidate). Done. The default score became `knn` on 2026-10-12, chosen by a rule
  fixed before the elongated-state curve (S1.3b) was read (see CHANGELOG).
- Calibrated novelty stack: kNN distance and joint scores from the held-out folds
  (`ValidationResult.holdout_evidence`), per-state support threshold, per-state
  `novel_population` threshold from held-out reference donors. Done.
- Labels-only `Predictions` (no confidence) no longer read `insufficient` on every state:
  local support stands in for confidence, and without a latent the states are at most
  `supported_with_caution`. Done (`confidence_source`).
- Tag v0.1.0; CI green; docs build clean.

### Checklist

- [x] `ExpressionConfig`, `ExpressionSpace`, `build_expression_space`, `score_query`
- [x] `shift_source`, `expression_shift*`, `latent_shift*`, `groups_shifted` in the state table
- [x] Frozen format version 3 with `expression/`; version 2 still loads
- [x] `simulate_transport` regenerates query counts from the final latent
- [x] Benchmark scores shift evidence separately from status
- [ ] Identity-loss status (PI decision)
- [x] Local support in the tool's latent (`novelty.local_support`, `min_support`)
- [x] Novelty decision rule from the withheld-state curve (`ood_rule`; curve in `simulation.withheld_state`; centroid q95 predeclared 2026-09-30)
- [x] kNN novelty score, support calibration, joint score, calibrated population call (`ood_score`, `support_calibration`, `population_calibration`)
- [x] Labels-only predictions path (`confidence_source`)
- [x] Full grid, first run: 450 comparator cells plus the withheld-state curve with the mappers on a local GPU (2026-09-30 to 2026-10-01, `01_simulation/_h/local_grid.sh`); G0 failed, see the milestone log
- [x] Full grid, reruns after the per-donor latent offset fix and the expression-arm amendment (local, same runner): second run 2026-10-01 08:13 to 18:46 (false rate 0.098 from a true-label null), third run 2026-10-01 19:17 to 2026-10-02 05:57 with the assigned-label null and geometric-median offsets
- [x] `00_gates/_m/G0.md`: **PASS** 2026-10-02 on `ef0560a`, recorded in `configs/gates.yaml` (`def81ff`); tag v0.1.0; wheel built with `uv build` for the cluster environment

### Commands

```bash
make lint PY=.venv/bin/python && make test PY=.venv/bin/python && make docs PY=.venv/bin/python
# benchmark repository
PYTHONPATH=. .venv/bin/python -m statetrace_benchmark.simulation.comparators --scenario disease_shifted --seed 1
PYTHONPATH=. .venv/bin/python -m statetrace_benchmark.simulation.aggregate
PYTHONPATH=. .venv/bin/python -m statetrace_benchmark.gates.check G0
```

### Expected artifacts

`reports/2026-09-29_g0_comparators_smoke.md`, `reports/2026-09-29_g0_shift_rerun.md`,
`01_simulation/_m/metrics/comparators.csv`, `01_simulation/_m/metrics/g0_summary.csv`,
`00_gates/_m/G0.md`.

### Promotion gate

G0 (a), (b) in its two-arm form of 2026-09-30, (c) as predeclared, with shift evidence
reported separately from status, on 3
comparators x 5 scenarios x 5 seeds x 9 grid cells.

### Notes

The morning smoke of 2026-09-29 concluded that surgery absorbs the shift on a simulator whose
query counts did not carry it. The rerun on the corrected simulator reached the same
conclusion on real evidence. Per-donor offset removal raised the pseudobulk test's power from
z about 1 to z above 10 at a 4-unit shift; the simulation shares donor effects across states
exactly, and Stage 3 must measure how much of that power survives a real platform effect.

## Stage 2: backend coverage and G1

### Objective

Reach the comparator set the paper promises and remove the fixed-latent optimism of grouped
holdout, so that G1 can ask whether the problem exists on real multi-donor data.

### Deliverables

- Symphony adapter in the benchmark (R -> CSV -> `Predictions.from_symphony`); the G1
  comparator set is scANVI, scPoli and Symphony (PI, 2026-10-02). scConform and Seurat/Azimuth
  moved to Stage 3 as G2 comparators.
- scANVI `AnnDataBackend`: the latent is refit inside every held-out fold (`refit_latent=True`),
  lifting the v0.1 caveat that grouped holdout is an optimistic bound on embedding transfer.
- Donor-support restriction (20 / 10 / 5 / 3 / 1 donors) driven from the benchmark on
  Stephenson et al. 2021 PBMC (130 participants, three sites; broad states carried by >= 10
  donors, sensitivities at >= 20 and >= 5), with `insufficient` assigned at fewer than 3 donors
  in >= 90 % of restrictions (amended 2026-10-02 from "at most 3", `GATES.md`).

### Promotion gate

G1: withheld-state detection >= best comparator - 0.02; `insufficient` at fewer than 3 donors
in >= 90 % of restrictions; status agreement across the three mapping backends >= 0.8. U1 decided
from S1.5, S2.4 and S2.5: at least two real settings where cell-level F1 >= 0.80 while grouped
holdout says `fragile` or `insufficient`. If U1 fails, the project stops and publishes the
negative result; Stage 3 does not start.

### Expected artifacts

`02_perturbation/_m/`, `00_gates/_m/G1.md` (FAIL on detection, accepted 2026-10-18),
`00_gates/_m/U1.md`, tag v0.2.0.

## Stage 3: real-data robustness, G2 and G3

### Objective

Show that StateTrace's conclusions do not depend on the backend on public cross-cohort data
with known protocol and study effects, and that its terms predict replication better than the
simple rules a careful analyst already uses.

### Deliverables

- Study-level holdout (`leave_one_study_out`) exercised on real studies with the report
  carrying the study-level tables.
- Per-donor offset assumption tested under a real platform effect (PBMCBench): how much of the
  expression-space power survives when donor effects are only partly shared across states;
  record the answer in the docs and the transport report's caveats.
- Group-aware calibration tables: a backend's confidence against recovery in an unseen donor.
- Feature-retention reporting in `route.report()`.
- Donor-level shift interval: the frozen-expression shift computed per query donor, the state's
  level read from the interval across donors, and the boundary flag derived from that interval
  rather than the fixed `shift_tolerance` (recorded 2026-10-25 from the G2 stage-03 reading).
  Built 2026-10-26 on `v0.3-dev` (`TransportConfig.shift_interval`, `shift_ci_low` /
  `shift_ci_high`, `shift_interval_boundary`, `status_interval_alternative`; reported only).
- Consensus transport status across backends (`Predictions` from several mappers of the same
  query): the modal status with the per-backend statuses and their agreement reported as its
  reliability; an application-stage mode, never a substitute for the per-backend criterion.
  Built 2026-10-26 on `v0.3-dev` (`robustness.consensus_status`; `compare_backends(...,
  tolerance=)` reads exact agreement, agreement within the cutoff or the interval margin, and
  the consensus with its support).
- Downstream helpers needed for U2: the replication model (`replication ~ cell_confidence +
  cell_number + donor_number + integration_score + StateTrace`) is a benchmark analysis, but
  the software must export the per-state terms it uses in one table.

### Promotion gate

G2: conclusions identical across backends for >= 80 % of states; states failing donor-level
reproducibility identified in at least one embedding that scores well on scIB. U2: likelihood
ratio at alpha 0.01 and AUROC gain >= 0.05 over the best simple baseline in >= 2 systems. If U2
fails, the software ships the rule-based check and the paper becomes a validation benchmark.
G3 repeats G2 on brain with pathology-stratum stability.

### Expected artifacts

`03_pbmc/_m/`, `04_hlca/_m/`, `05_brain/_m/`, `G2.md`, `U2.md`, `G3.md`, tags v0.2.x and v0.3.0.

## Stage 4: inference validity and G4

### Objective

Turn the status into a demonstrated gate: analyses restricted to `supported` states control
type I error where ungated analyses do not, without losing power, and supported states
replicate in the independent cohort.

### Deliverables

- Downstream-validity helpers: type I error under a permuted null, FDR, power and effect-size
  bias for gated versus ungated pseudobulk and differential-abundance analyses; replication
  summaries at the state level.
- The decision-impact rate and number needed to flag, exported per system for `UTILITY.md`.
- The biological application (a frozen brain reference transported into an independent disease
  cohort) run through `validate` -> `validate_transport` -> `report` with no project-specific code
  in the software.

### Promotion gate

G4: gated analyses control type I error at the nominal level where ungated analyses do not; no
loss of power on `supported` states. U3: replication enrichment monotone in status, supported
versus unsupported difference at alpha 0.05 with the state as the unit, gated arm controls type
I error. If U3 fails, the paper is a validation benchmark rather than an inference-gating method.

### Expected artifacts

`06_nvu_hiv_oud/_m/`, `07_inference_validity/_m/`, `G4.md`, `U3.md`, tag v0.3.0.

## Stage 5: generalisation and G5

### Objective

Show the framework holds across systems, species and backends, and freeze the result set the
paper reports.

### Deliverables

- Ortholog `FeatureMap` from real BioMart tables (human -> macaque, human -> mouse) through the
  expression space, with ambiguous and unmapped genes reported.
- Sensitivity analyses driven from the benchmark: thresholds, tentative controls, reduced
  feature sets.
- Optional: cell-level frozen-expression displacement, the within-donor complement of the
  pseudobulk shift, to locate a shifted subpopulation that the pseudobulk averages over and to
  separate identity contamination from a true shift. Built only if Stages 1 to 4 finish on
  schedule.

### Promotion gate

G5 as G2 with feature-retention reporting. U4: advantage in >= 3 systems and >= 3 backends in
>= 75 % of settings. The manuscript release tag (0.4.x) is the version cited at submission.

### Expected artifacts

`08_organoid_scz/_m/`, `09_cross_species/_m/`, `G5.md`, `U4.md`, the manuscript release tag.

## Stage 6: StateTrace-Spatial (0.5.0) — distinguish mapping failure from segmentation failure

### Objective

Core StateTrace asks: **Is this mapped state trustworthy and is it genuinely novel?**
StateTrace-Spatial asks: **Is that conclusion robust to how the spatial cell itself was
reconstructed?** The new diagnostic is boundary stability. CellART is a controlled
preprocessing perturbation and optional adapter, never a correctness oracle or a reason to
turn StateTrace into a segmentation package. The full design is in {doc}`spatial-roadmap`.

### Deliverables

- Segmentation-independent input contract for cell masks, spatial transcript assignments,
  reconstructed counts and `Predictions`, with coordinate and reference provenance.
- Spatial correspondence that preserves one-to-one matches, splits, merges, ambiguous and
  unmatched cells; mask IoU and transcript reassignment summaries.
- Paired novelty, mapped-label, reference-frozen displacement and uncertainty diagnostics,
  with a calibrated segmentation-sensitivity score and donor x state summaries.
- Report interpretations: robust shifted state, segmentation-sensitive, identity-sensitive,
  and robustly reference-unsupported; unavailable evidence remains explicit.
- Optional CellART output adapter after the generic diagnostic passes synthetic checks.
  Cell-level frozen-expression scoring from Stage 5 is a prerequisite for cell-level distance
  deltas; donor x state pseudobulk remains the primary shift evidence.
- A Xenium worked example; real-data runs and corruption experiments live in the benchmark
  repository, while reusable adapters, diagnostics and synthetic tests live here.

### Promotion gate

**G6-Spatial is proposed, not yet registered or passed.** Before the deciding run, the
benchmark repository must fix the dataset and donor split, independent biological-shift
evidence, corruption doses, calibration procedure, endpoint and numerical tolerances.
Compare native, deliberately corrupted and CellART-reconstructed segmentations. The primary
test is improved detection of contamination-induced false novelty over novelty alone and
mapper uncertainty at a matched donor-calibrated false-positive rate, subject to a
prespecified minimum retention of independently supported biological shifts. Report effect
sizes and uncertainty across held-out donors, including match coverage and split/merge cases.
Agreement with CellART cannot satisfy the gate, especially with a shared reference.

### Expected artifacts

Proposed benchmark artifacts: `G6-Spatial.md`, a frozen corruption manifest, spatial
correspondence and paired diagnostic tables, donor x state summaries, calibration and
reference-sensitivity results, an independently validated tissue example, and software,
benchmark, reconstruction and reference provenance. Register paths and gate configuration in
the benchmark repository before implementation; these artifacts do not exist yet.

## Stage 7: usability and stable release (1.0.0)

### Objective

A non-author reaches a transport report from two h5ads and the docs, the paper cites a tagged,
archived, installable release (the Stage 5 manuscript tag), and the stable 1.0.0 closes the core
roadmap after Stage 6.

### Deliverables

- Tutorial: two h5ads to a transport report, under 15 public API calls, minutes on a
  workstation.
- Docs and API fixes from the usability test; every public function documented.
- PyPI release and a conda-forge recipe; CI on every supported Python; Read the Docs build.
- Zenodo DOI; `CITATION.cff` completed; clean-environment reproduction of the frozen result
  set with `env/environment.lock.txt`.
- Final submission tag referenced in the manuscript (0.4.x; Stage 5), archived on Zenodo.
- Stable 1.0.0 tag after Stage 6 has closed; not a submission requirement.

### Promotion gate

U5: a non-author produces and interprets a transport report from the docs alone; then one
outside collaborator does the same on their own data. Release criteria for v1.0 in
{doc}`roadmap`.

## Schedule

Week 1 is 2026-09-29 to 2026-10-05; week 12 ends 2026-12-21. The benchmark repository's
`TIMELINE.md` carries the benchmark and manuscript columns; this table carries the software
column and the decision points.

| Week | Dates | Software | Gate / decision | Stage |
|---|---|---|---|---|
| 1 | Sep 29 - Oct 5 | Expression space (done); identity-loss status; local support; tag v0.1.0 | PI signs thresholds, comparator set, identity-loss design | 1 |
| 2 | Oct 6 - 12 | Novelty decision rule; labels-only path; fixes from the grid; scConform bridge | **G0** on the full grid | 1 |
| 3 | Oct 13 - 19 | Symphony adapter; G1 fixes chosen on the simulator (study-unit null, expression-space reproducibility, kNN score) | Stephenson 2021 PBMC fetched; **G1** evaluated twice, FAIL on detection accepted 2026-10-18; **U1** | 2 |
| 4 | Oct 20 - 26 | v0.2.0 tagged; group-aware calibration | G1 closed as accepted; stages 03/04 open | 2 |
| 5 | Oct 27 - Nov 2 | Offset-assumption test on PBMCBench; study-level holdout on real studies; scANVI per-fold backend; scConform and Azimuth bridges | **G2**, **U2** | 3 |
| 6 | Nov 3 - 9 | Fixes from G2; feature-retention reporting | **G3** | 3 |
| 7 | Nov 10 - 16 | v0.3.0: downstream-validity helpers | Application stage runs (needs its frozen reference) | 4 |
| 8 | Nov 17 - 23 | Decision-impact export | **G4**, **U3** | 4 |
| 9 | Nov 24 - 30 | Ortholog maps on real tables; sensitivity hooks; manuscript release tag | **G5**, **U4**; result set frozen | 5 |
| 10 | Dec 1 - 7 | Tutorial; docs and API fixes from the usability test | **U5** | 7 |
| 11 | Dec 8 - 14 | PyPI, conda recipe, Zenodo; clean-environment reproduction | | 7 |
| 12 | Dec 15 - 21 | Submission tag (0.4.x) cited in the paper; 1.0.0 follows Stage 6 | Submission-ready draft | 7 |

Critical path: the application stage's frozen reference (week 7) gates G4 and U3; if it slips,
Stage 4 slips one-for-one while Stages 2 and 3 finish on the public systems. U1 (week
3) and U2 (week 5) are the real go/no-go points and are decided before the expensive real-data
stages run.

## Milestone log

| Date | Stage | Event | Software SHA | Benchmark SHA | Evidence |
|---|---|---|---|---|---|
| 2026-09-28 | 0 | Renamed to StateTrace and re-scoped; transport, gating, report, frozen format v2 | `b9a63e1` | | test suite, docs build |
| 2026-09-28 | 0 | Cohort-offset removal, standardised novelty, in-reference gating | `8dcf9bb` | | `reports/2026-09-28_gating_sweep_holdout_calibration.md` |
| 2026-09-29 | 1 | scANVI and scPoli adapters; G0 smoke on two seeds; blocker recorded | `76e2cc7` | `d39fa44` | `reports/2026-09-29_g0_comparators_smoke.md` |
| 2026-09-29 | 1 | Frozen expression space; two coordinate systems; simulator corrected | `7f48316` | | 90 tests, docs |
| 2026-09-29 | 1 | Per-donor offset removal; G0 rerun: shift called by the expression space for both mappers, none by their latents | `841f1dc` | `22d01a9` | `reports/2026-09-29_g0_shift_rerun.md` |
| 2026-09-30 | 1 | Local reference support, labels-only path, quantile out-of-reference rule; withheld-state curve (kNN) recommends the held-out 95th percentile | `1e0ea45` | `7e4f152` | `reports/2026-09-30_withheld_state_curve_knn.md` |
| 2026-09-30 | 1 | kNN and joint novelty scores, calibrated support and population thresholds; centroid q95 predeclared by the PI; G0 (b) re-specified as two arms | `cd3c8f1` | `dbbcda3` | `reports/2026-09-30_withheld_state_curve_knn.md` (addendum), `configs/thresholds.yaml` |
| 2026-10-01 | 1 | First full stage-01 run on the local GPU (510 cells): G0 FAIL on easy-cell accuracy (0.3 points), the false rate in mapper latents (0.11 vs 0.05) and the expression arm as predeclared; latent-arm power and truth-category recovery pass | `be57889` | `8352c8a` | `00_gates/_m/G0.md` (first run) |
| 2026-10-01 | 1 | Per-donor latent offsets for query and null (surgery places each query donor with its own displacement); expression arm amended to a discrimination criterion, recorded in `GATES.md`; rerun launched | `f1b2c65` | `833c470` | `GATES.md` Amendments |
| 2026-10-01 | 1 | Held-out null built from assigned labels (as the query is); geometric-median latent offsets; raw and residual displacement reported; per-state AUROC made the primary expression-arm form with the within-run stress test secondary; (a) tolerance 0.03 (both in `GATES.md`, Amendments) | `ef0560a` | `15bc438` | `00_gates/_m/G0_2026-10-01_second_run.md` |
| 2026-10-02 | 1 | Third full stage-01 run (510 cells, 10.7 h local GPU): **G0 PASS**. Pooled false rate 0.062 (scANVI 0.047, scPoli 0.078); latent-arm power margins 0.46 and 0.19; per-state expression AUROC 0.92 and 0.88; truth-category recovery 0.97 and 0.94; accuracy gap 2.3 points. v0.1.0 tagged; stage 02 unlocked | `ef0560a` (tag v0.1.0) | `def81ff` | `00_gates/_m/G0.md`, `configs/gates.yaml` |
| 2026-10-09 | 6 | Added planned StateTrace-Spatial extension: boundary stability, optional CellART adapter, controlled segmentation-error benchmark and proposed G6-Spatial; schedule/version unassigned, no implementation or benchmark claim | pending | not registered | `docs/spatial-roadmap.md` |
| 2026-10-18 | 2 | G1 rerun #2 on `07def17`: detection 0.652 vs 0.722 FAIL, agreement 0.843 and `insufficient` 1.000 pass; support-deficit score added and rejected by its pre-registered rule; failure accepted by the PI as a stated limitation; benchmark stages 03/04 open | `9b6e0a7` | `455959b` | benchmark `00_gates/_m/G1.md`, `reports/2026-10-17_ood_score_rule_support.md` |
| 2026-10-18 | 6, 7 | Stages 6 and 7 swapped: StateTrace-Spatial is Stage 6 at 0.5.0 (unscheduled, not a submission requirement); usability and the stable 1.0.0 release are Stage 7; the submission cites the Stage 5 manuscript tag | | | this page, `docs/roadmap.md`, `docs/spatial-roadmap.md` |
