# Concepts

## Donors, not cells, are the replication unit

Thousands of cells from two donors are two observations of a state, not thousands.
AtlasRoute reports, for every state, how many donors support it, how well each held-out
donor is recovered, and how variable that recovery is. Cell-level accuracy and
donor-level reproducibility are always separate numbers.

## Mapping and validation are separate operations

AtlasRoute does not implement an integration or classification algorithm. Backends
(currently a k-nearest-neighbour classifier in a fixed latent space; scVI/scANVI planned)
sit behind the {class}`~atlasroute.backends.base.MappingBackend` protocol. The
contribution is the validation, calibration, abstention and reporting layer around them,
so the same donor-level diagnostics apply whichever engine is used.

## Abstention is a result

A query cell may lie outside the reference, be equidistant from two states, or be
technically shifted. Forcing it into the nearest state hides that. `unknown` is a valid
assignment; in v0.1 the `abstain_below` option of {class}`~atlasroute.LODOConfig`
exercises this on held-out donors, and v0.2 makes it data-driven for queries.

## Freeze before you look at the query

{meth}`~atlasroute.AtlasReference.freeze` writes an immutable, checksummed bundle of the
latent coordinates, labels, donor statistics, backend configuration and validation
tables. A query is mapped against that bundle; it never redefines the reference unless
you deliberately build a new one. See {doc}`reference-format`.

## Three questions, kept apart

1. **Reference stability**: is a state reproducibly identifiable across donors of the
   reference? (v0.1: {meth}`~atlasroute.AtlasReference.leave_one_donor_out`)
2. **Transportability**: can the state be recovered in an independent query? (v0.2)
3. **Inference readiness**: is the transported state supported well enough for
   donor-level differential, compositional or genetic analyses? (v0.2 report)
