# Roadmap

Stage-by-stage tracking, schedule and milestone log: {doc}`staged-roadmap`.

Releases are gated by the benchmark repository: StateTrace advances to the next layer only when
it is comparable to or better than the main comparators on the previous layer's task.

| Version | Content | Benchmark gate |
|---|---|---|
| 0.1 | `StateReference`, grouped holdout (donor, study), random-split control, reproducibility categories, frozen format, kNN baseline | G0 simulation parity |
| 0.2 | `Predictions` adapters (scPoli, scANVI, Symphony, Azimuth, CellTypist), `validate_transport`, shift diagnostics, two-sample engines (E-distance; scXMatch when installed) reported next to the status, inference gating, `compare_backends`, abundance and pseudobulk, transport report, feature harmonisation | G1 semi-synthetic perturbations on real data |
| 0.3 | scANVI `AnnDataBackend` with per-fold refit, group-aware calibration tables, downstream-validity helpers (type I error / replication summaries) | G2 public cross-cohort (PBMC, HLCA) |
| 0.4 | hierarchical states, plotting, CLI | G3 brain and G4 application |
| TBD (Stage 7) | StateTrace-Spatial: boundary stability and segmentation sensitivity; optional CellART output adapter | G6-Spatial (proposed; criteria to be registered) |
| later | genetics annotation export and multi-omic references | — |

## Planned: StateTrace-Spatial

Stage 7 adds the question **is the mapped-state conclusion robust to how the spatial cell
was reconstructed?** It compares native, deliberately corrupted and alternative cell
boundaries, measuring transcript reassignment, label changes and changes in novelty,
reference-frozen displacement and uncertainty. CellART is an optional preprocessing
perturbation, not ground truth. See {doc}`spatial-roadmap` for the benchmark design, donor-level
evaluation, reference-sensitivity controls and implementation scope. Its schedule and version
are unassigned; it is not a v1.0 release requirement.

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
