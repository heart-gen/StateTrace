# StateTrace staged roadmap

Software stages, each closed by a benchmark gate, on the twelve-week schedule that ends with a
submission-ready manuscript on 2026-12-21. The benchmark repository owns the gate definitions
(`GATES.md`, `UTILITY.md`, `configs/gates.yaml`, `configs/utility_gate.yaml`) and the weekly
timeline (`TIMELINE.md`); this page tracks what the *software* must deliver for each gate,
what evidence closes the stage, and which inputs from the PI it waits on. Update the summary
table and the milestone log when a stage changes state.

## Summary

| Stage | Status | Weeks | Deliverables | Promotion gate | Required evidence | Version | Last commit |
|---|---|---|---|---|---|---|---|
| 0 | complete | before 1 | Package, grouped validation (donor, study), reproducibility categories, `Predictions`, kNN baseline, frozen format, transport route and gating, report, feature harmonisation | Fresh install, silent import with torch and scvi blocked, full test suite, docs build | CI log, `tests/test_imports.py`, docs build | 0.1.0 (untagged) | `8dcf9bb` |
| 1 | in progress | 1 - 2 | Two coordinate systems: frozen expression space with per-donor offsets (done); identity-loss status; local support in the tool's latent; novelty decision rule; corrected simulator (done); tag v0.1.0 | **G0** on the full simulation grid: (a) cell accuracy within 2 points of scANVI / scPoli, (b) out-of-reference AUROC >= best comparator - 0.02, (c) >= 90 % planted statuses correct, shift evidence scored separately | `00_gates/_m/G0.md` with both SHAs; grid summary CSV; smoke and rerun reports | 0.1.0 tag, 0.1.1 fixes | `841f1dc` |
| 2 | planned | 3 - 4 | Backend coverage: Symphony adapter (R -> CSV -> `from_symphony`), scConform bridge, scANVI `AnnDataBackend` with per-fold latent refit, Azimuth adapter if signed off; donor-support restriction runs | **G1** semi-synthetic perturbations on the NVU reference; **U1** decided (does the problem exist) | `G1.md`, `U1.md`; withheld-state AUROC table; reproducibility vs donor count; backend status agreement >= 0.8 | 0.2.0 | |
| 3 | planned | 4 - 6 | Real-data robustness: study-level holdout on real studies; test of the per-donor offset assumption under a real platform effect; group-aware calibration tables; feature-retention reporting | **G2** PBMC and HLCA, **U2** decided (beats simple baselines at predicting replication); **G3** brain | `G2.md`, `U2.md`, `G3.md`; scIB reference table; calibration figure | 0.2.x, 0.3.0 | |
| 4 | planned | 7 - 8 | Inference validity: downstream-validity helpers (type I error under permuted null, FDR, power, effect-size bias, replication summaries); gated versus ungated pseudobulk and differential-abundance pipeline | **G4** NVU -> SCORCH application and validity layer; **U3** decided (supported states replicate, gating controls type I error) | `G4.md`, `U3.md`; error-rate and replication tables | 0.3.0 | |
| 5 | planned | 9 | Generalisation: ortholog `FeatureMap` on real BioMart tables; retention in the report; sensitivity analyses (thresholds, reduced features); optional cell-level frozen-expression displacement | **G5** organoid and cross-species student stages; **U4** decided (>= 3 systems, >= 3 backends) | `G5.md`, `U4.md`; frozen result set; manuscript release tag | 0.4.0 | |
| 6 | planned | 10 - 12 | Usability and release: tutorial (two h5ads to a transport report), docs and API fixes from the usability test, PyPI and conda recipe, Zenodo DOI, clean-environment reproduction, final tag cited in the paper | **U5** a non-author produces and interprets a transport report from the docs; then one outside collaborator | `U5.md`; tutorial runtime; `env/environment.lock.txt`; Zenodo record | 1.0.0 | |

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
- Two coordinate systems, always: state identity from the mapping space, state shift from the
  frozen expression space. A diagnostic that needs the query present in the latent it is judged
  in is not a shift diagnostic.
- Freeze before query. Thresholds are predeclared in the benchmark configs and signed off before
  the deciding stage runs.
- No CLI, no pickle, no GitHub remotes until the release stage; commits carry no trailers.

## Current status

Stage 0 is complete. Stage 1 has the expression space in place and validated on two seeds at
the hard grid cell (both mappers absorb a planted state shift in their own latent; the
expression space calls it `shifted` in every run; no false shift calls on stable states). The
full grid has not run. Three software items remain in Stage 1 and one PI decision (the
identity-loss status). Nothing in Stages 2 to 6 has started.

Blocking inputs, by the week they are needed (`INPUTS_NEEDED.md` in the benchmark repository
has the full list):

| Needed by | Input from the PI | Blocks |
|---|---|---|
| week 1 | Sign-off of thresholds, scenarios, comparator set and gate tolerances (`predeclared: false` in `thresholds.yaml`, `utility_gate.yaml`) | G0 evaluation |
| week 1 | Decision on the identity-loss case (new status, or fold the neighbour's contamination evidence into the absent state's row) | Stage 1 vocabulary, manuscript table 1 |
| week 1 | scConform (R dependency) and Azimuth in or out | Stage 2 scope |
| week 3 | NVU control reference (`aim1_reference_annotated.h5ad`); Bridges-2 GPU allocation | G1, U1, and everything after |
| week 3 | PBMCBench accession and donor / protocol split | G2 |
| week 5 | HLCA query studies and annotation level; SCORCH macaque path | G2, G5 |
| week 5 | Replication definitions per public system (which cohort replicates, which association) | U2, U3 |
| week 5 | Student assignments for the organoid and cross-species stages | G5 |
| week 6 | Quest account or a copy of SEA-AD microglia-immune on Bridges-2; reference choice (Tran vs HBCA) and scoring level | G3 |
| week 10 | A non-author tester, then an outside collaborator with two h5ads | U5 |
| week 10 | Author list, affiliations, declarations; fallback venue if U2 fails | Submission |

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
  separation (`simulation.withheld_state`) and the `ood_rule` option are done; the PI chooses
  the rule from the curve and predeclares it in `configs/thresholds.yaml`.
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
- [x] Novelty decision rule from the withheld-state curve (`ood_rule`; curve in `simulation.withheld_state`; rule to be predeclared by the PI)
- [x] Labels-only predictions path (`confidence_source`)
- [ ] Full grid on Bridges-2 (`01_simulation/_h/02c.comparators.sh`, about 20 GPU-hours)
- [ ] `00_gates/_m/G0.md`; tag v0.1.0

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

G0 (a), (b), (c) as predeclared, with shift evidence reported separately from status, on 3
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

- Symphony adapter in the benchmark (R -> CSV -> `Predictions.from_symphony`).
- scConform bridge for the uncertainty arm of G0 (b) and G1.
- scANVI `AnnDataBackend`: the latent is refit inside every held-out fold (`refit_latent=True`),
  lifting the v0.1 caveat that grouped holdout is an optimistic bound on embedding transfer.
- Azimuth adapter, only if the PI includes the practical-workflow arm.
- Donor-support restriction (20 / 10 / 5 / 3 / 1 donors) driven from the benchmark, with
  `insufficient` assigned at <= 3 donors in >= 90 % of restrictions.

### Promotion gate

G1: withheld-state detection >= best comparator - 0.02; `insufficient` at <= 3 donors in
>= 90 % of restrictions; status agreement across the three mapping backends >= 0.8. U1 decided
from S1.5, S2.4 and S2.5: at least two real settings where cell-level F1 >= 0.80 while grouped
holdout says `fragile` or `insufficient`. If U1 fails, the project stops and publishes the
negative result; Stage 3 does not start.

### Expected artifacts

`02_perturbation/_m/`, `00_gates/_m/G1.md`, `00_gates/_m/U1.md`, tag v0.2.0.

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
- The NVU -> SCORCH application run through `validate` -> `validate_transport` -> `report`
  with no project-specific code in the software.

### Promotion gate

G4: gated analyses control type I error at the nominal level where ungated analyses do not; no
loss of power on `supported` states. U3: replication enrichment monotone in status, supported
versus unsupported difference at alpha 0.05 with the state as the unit, gated arm controls type
I error. If U3 fails, the paper is a validation benchmark rather than an inference-gating method.

### Expected artifacts

`06_nvu/_m/`, `07_inference_validity/_m/`, `G4.md`, `U3.md`, tag v0.3.0.

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
>= 75 % of settings.

### Expected artifacts

`08_organoid_scz/_m/`, `09_cross_species/_m/`, `G5.md`, `U4.md`, the manuscript release tag.

## Stage 6: usability and release

### Objective

A non-author reaches a transport report from two h5ads and the docs, and the paper cites a
tagged, archived, installable release.

### Deliverables

- Tutorial: two h5ads to a transport report, under 15 public API calls, minutes on a
  workstation.
- Docs and API fixes from the usability test; every public function documented.
- PyPI release and a conda-forge recipe; CI on every supported Python; Read the Docs build.
- Zenodo DOI; `CITATION.cff` completed; clean-environment reproduction of the frozen result
  set with `env/environment.lock.txt`.
- Final tag referenced in the manuscript.

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
| 3 | Oct 13 - 19 | v0.1.1; Symphony adapter | PBMCBench fetched; **G1** starts when the NVU reference exists; **U1** | 2 |
| 4 | Oct 20 - 26 | v0.2.0: scANVI per-fold backend; group-aware calibration | G1 closes | 2 |
| 5 | Oct 27 - Nov 2 | Offset-assumption test on PBMCBench; study-level holdout on real studies | **G2**, **U2**; students start | 3 |
| 6 | Nov 3 - 9 | Fixes from G2; feature-retention reporting | **G3** | 3 |
| 7 | Nov 10 - 16 | v0.3.0: downstream-validity helpers | NVU application runs (needs the reference) | 4 |
| 8 | Nov 17 - 23 | Decision-impact export | **G4**, **U3** | 4 |
| 9 | Nov 24 - 30 | Ortholog maps on real tables; sensitivity hooks; manuscript release tag | **G5**, **U4**; result set frozen | 5 |
| 10 | Dec 1 - 7 | Tutorial; docs and API fixes from the usability test | **U5** | 6 |
| 11 | Dec 8 - 14 | PyPI, conda recipe, Zenodo; clean-environment reproduction | | 6 |
| 12 | Dec 15 - 21 | Final tag cited in the paper | Submission-ready draft | 6 |

Critical path: the NVU reference (week 3) gates G1 and every utility gate after it; if it slips,
Stage 2 finishes on simulation and the public systems, and Stage 4 slips one-for-one. U1 (week
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
