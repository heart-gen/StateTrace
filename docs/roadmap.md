# Roadmap

Stage-by-stage tracking, schedule and milestone log: {doc}`staged-roadmap`.

Releases are gated by the benchmark repository: StateTrace advances to the next layer only when
it is comparable to or better than the main comparators on the previous layer's task.

| Version | Content | Benchmark gate |
|---|---|---|
| 0.1 | `StateReference`, grouped holdout (donor, study), random-split control, reproducibility categories, frozen format, kNN baseline, `Predictions` adapters (scPoli, scANVI, Symphony, Azimuth, CellTypist), `validate_transport`, shift and novelty diagnostics, inference gating, `compare_backends`, abundance and pseudobulk, transport report, feature harmonisation | G0 simulation parity (passed 2026-10-02) |
| 0.2 | Frozen expression space and per-donor offsets, calibrated novelty stack (kNN out-of-reference score, support and population calibration, study-unit null), expression-space reproducibility, held-out caution calibration, two-sample engines (E-distance; scXMatch when installed) reported next to the status, frozen format 4 | G1 semi-synthetic perturbations on real data (detection failed and was accepted as a stated limitation, 2026-10-18) |
| 0.3.0 | Reporting additions, status rule unchanged: donor-level shift interval (data-driven boundary flag), consensus transport status across backends with reported agreement, feature-retention reporting | G2 public cross-cohort (PBMC, HLCA; stage-03 agreement accepted as a stated limitation, 2026-10-25) and G3 brain (passed 2026-10-26) read on 0.2.1; 0.3.0 released 2026-10-27 for the application stages (tag `v0.3.0`, `31e1856`; published to PyPI as 0.3.1, formatting only) |
| 0.3.x | downstream-validity helpers (type I error / replication summaries, for stage 07), group-aware calibration tables, scANVI `AnnDataBackend` with per-fold refit | G4 application and U3 |
| 0.4 | hierarchical states, plotting, CLI | G4 application |
| 0.5 (Stage 6) | StateTrace-Spatial: boundary stability and segmentation sensitivity; optional CellART output adapter; not a submission requirement | G6-Spatial (proposed; criteria to be registered) |
| 1.0 (Stage 7) | Usability and stable release: tutorial, docs and API fixes from the usability test, PyPI and conda, Zenodo | U5 usability (non-author, then an outside collaborator) |
| later | genetics annotation export and multi-omic references | — |

## Planned: StateTrace-Spatial

Stage 6 (version 0.5.0) adds the question **is the mapped-state conclusion robust to how the
spatial cell was reconstructed?** It compares native, deliberately corrupted and alternative cell
boundaries, measuring transcript reassignment, label changes and changes in novelty,
reference-frozen displacement and uncertainty. CellART is an optional preprocessing
perturbation, not ground truth. See {doc}`spatial-roadmap` for the benchmark design, donor-level
evaluation, reference-sensitivity controls and implementation scope. Its schedule is unassigned;
it is not a submission requirement and precedes the 1.0 stable release (Stage 7) in version number
only. The manuscript cites the Stage 5 release tag.

## Cell-level frozen-expression evidence

The donor x state pseudobulk distance is the primary shift evidence because donors are the
replication unit. Its cell-level complement now exists at the level of the state: a frozen
cell-level space (reference cells only) in which the E-distance and scXMatch's cross-match test
compare each reference state with its query cells, next to the same tests on donor profiles
({mod}`statetrace.twosample`). They are reported, not decisive (see {doc}`concepts`). scXMatch
(Moeller et al., *Bioinformatics* 2026) is the reference two-sample test for that question, so
StateTrace orchestrates it rather than re-implementing it.

Still planned: locating a shifted subpopulation *within* a donor x state that the pseudobulk
averages over (per-cell displacement against a null from held-out reference cells). Together
with mapping-space identity this separates: same identity and same molecular state; same
identity and shifted molecular state; poor identity assignment; a truly out-of-reference
population.

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
