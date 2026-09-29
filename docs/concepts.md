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
5. **Distribution-shift diagnostics**: absent, novel, or a shifted version of an existing state.
   Per cell, reference-distance z-scores in the mapping latent; per donor x state, pseudobulk
   distance in the frozen expression space against held-out reference donors.
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

## Abstention is a result

`unknown` and `out_of_reference` are valid outcomes. Forcing every query cell into the nearest
state hides novelty and technical shift; StateTrace reports the fraction of uncertain and
out-of-reference cells per state and per donor and uses it in gating.

## Freeze before you look at the query

{meth}`~statetrace.StateReference.freeze` writes an immutable, checksummed bundle of latent
coordinates, labels, group statistics, state geometry, the expression space and validation
tables. A query is evaluated against that bundle and never redefines it.

## What StateTrace is not

Not an integration method, not a classifier competing with scPoli or scANVI, not a
differential-expression method, and not a conformal calibrator (scConform and conformal
annotation already provide cell-level coverage guarantees). Its calibration question is
group-aware: does a backend's confidence predict recovery in an unseen donor?
