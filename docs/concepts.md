# Concepts

## Donors, not cells, are the replication unit

Thousands of cells from two donors are two observations of a state, not thousands. Every
StateTrace metric exists at the level of a *group* (donor by default; individual, study, site
or cohort through `group_key` and `study_key`). Cell-level accuracy is reported alongside,
never instead.

## StateTrace consumes mapping; it does not perform it

Any tool's output becomes a {class}`~statetrace.Predictions` object (labels, optional
probabilities, the tool's native confidence or uncertainty). Grouped validation, transport
validation and backend comparison all work on that object, so the conclusions do not depend on
which mapper you prefer. A built-in kNN backend exists so the workflow runs without any
external tool, and the {class}`~statetrace.AnnDataBackend` protocol lets an external mapper be
refitted inside every holdout fold.

## Seven questions, one route

1. **Grouped validation**: leave-one-donor-out and leave-one-study-out holdouts.
2. **State reproducibility**: does a state survive removal of donors rather than cells
   (`reproducibility` = stable / moderate / fragile / insufficient)?
3. **Cross-cohort transportability**: does a frozen state definition recover in the donors of
   an independent cohort?
4. **Backend robustness**: do scPoli, scANVI, Symphony and the kNN baseline reach the same
   conclusion ({func}`~statetrace.compare_backends`)?
5. **Shift and novelty diagnostics**: absent, novel, or a shifted version of an existing
   state. Per donor x state, pseudobulk distance in the frozen expression space against
   held-out reference donors (decides the status); per cell, reference-distance z-scores in the
   mapping latent; optionally, two-sample tests (E-distance, scXMatch) of each state at cell
   and donor level, reported next to the status (see below).
6. **Inference gating**: `supported`, `supported_with_caution`, `shifted`, `insufficient` or
   `absent` per state; `route.pseudobulk(states="supported")` and
   `route.state_abundance(states="supported")` carry only gated states downstream.
7. **Transport report**: standardised evidence per state that can accompany a publication.

## Two coordinate systems: identity from the mapper, shift from the frozen reference

The latent of scANVI, scPoli or Symphony is fitted with the query present; surgery and
condition embeddings remove what separates the query donors from the reference, including a
displacement confined to one state. StateTrace therefore takes **state identity** from the
mapping space and **state shift** from a frozen biological space built once from the
reference's expression ({mod}`statetrace.expression`): does the molecular phenotype of state S
in the query donors lie within the between-donor variation of state S in the reference? The
null for that question comes from held-out reference donors, not from a theoretical cut-off.
See {doc}`transport`.

## Three layers of shift evidence, never conflated

Whether two sets of profiles come from the same distribution is a two-sample question with
dedicated, principled answers: the graph cross-match test of scXMatch (Moeller et al.,
*Bioinformatics* 2026, [doi:10.1093/bioinformatics/btag706](https://doi.org/10.1093/bioinformatics/btag706))
and the E-distance permutation test (Peidli et al., *Nature Methods* 2024). StateTrace does not
claim that question. It runs those tests as optional engines
({class}`~statetrace.TwoSampleConfig`, {mod}`statetrace.twosample`) and keeps three kinds of
evidence apart:

| Evidence | Unit | Answers | Role |
|---|---|---|---|
| Donor x state pseudobulk distance vs held-out reference donors | donor | is the state outside the reference's between-donor range? | decides `shift_level` and `status` |
| Two-sample tests on donor profiles (E-distance, scXMatch) | donor | do the query donors' profiles of the state differ in distribution from held-out reference donors'? | reported |
| Two-sample tests on cells (E-distance, scXMatch) | cell | do the query cells of the state differ in distribution from the reference cells, including in spread? | reported |
| Reference-distance z-scores in the mapping latent | cell | is a cell outside the reference support (out of reference)? | cell category, `novel_population` |

The cell-level tests are the most sensitive: they see a change in spread that a pseudobulk mean
cannot. They also treat cells of one donor as independent, so a cohort of genuinely new,
unshifted donors already differs from the reference's donors at cell level. A significant
cell-level test is evidence that something differs, not that the state failed to transport;
StateTrace's question is what the evidence together implies (a reproducible shifted state, an
unsupported or novel population, or a state that should not enter downstream inference).
scXMatch's authors name the adjacent question, whether the query was sampled from a subspace
of the reference, as outside a two-sample test; that is the out-of-reference arm here.

## Abstention is a result

`unknown` and `out_of_reference` are valid outcomes. Forcing every query cell into the nearest
state hides novelty and technical shift; StateTrace reports the fraction of uncertain and
out-of-reference cells per state and per donor and uses it in gating.

## Freeze before you look at the query

{meth}`~statetrace.StateReference.freeze` writes an immutable, checksummed bundle of latent
coordinates, labels, group statistics, state geometry, the expression space and validation
tables. A query is evaluated against that bundle and never redefines it.

States are therefore defined independently of every test run on the query. A two-sample test
of clusters inferred from the same data it compares is double dipping and will almost always
find a difference (as scXMatch's authors warn); StateTrace's tests compare a frozen reference
state with the query cells a mapper assigned to it, and the features, scaling and projections
come from the reference alone. {func}`~statetrace.validate_transport` refuses a query that
shares a group (donor) with the reference, warns when the reference is not frozen, and records
both in `route.reference["independence"]`; `TransportConfig(allow_group_overlap=True)` is
reserved for the explicit cell-level control. Mapper assignment can only pull query cells
towards the reference (absorption), which makes the tests conservative, not circular.

## What StateTrace is not

Not an integration method, not a classifier competing with scPoli or scANVI, not a
differential-expression method, not a conformal calibrator (scConform and conformal annotation
already provide cell-level coverage guarantees), and not a distribution-shift test (scXMatch and
the E-distance answer *whether* two distributions differ; StateTrace asks what that implies for
transporting the state). Its calibration question is
group-aware: does a backend's confidence predict recovery in an unseen donor?
