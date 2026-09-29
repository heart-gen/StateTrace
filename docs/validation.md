# Donor-aware validation

## Leave-one-donor-out

For each donor *d*: remove every cell of *d*, fit the backend on the remaining cells in
the fixed latent space, predict the cells of *d*, and record the prediction, its
probability, the second-best state, the margin and the number of nearest reference
neighbours that voted for the prediction. Concatenating the folds gives one row per
reference cell ({attr}`~atlasroute.ValidationResult.cell_predictions`).

### Donor x state table

For each (donor, true state) present in that donor: `n_cells`, `n_predicted`, `recall`,
`precision`, `f1` and `scored` (`n_cells >= LODOConfig.min_donor_cells`). Abstained
cells count as misses for recall and never enter a precision denominator.

### State table

Over scored donors: `n_donors_scored`, `median_donor_recall`, `min_donor_recall`,
`recall_variance` (sample variance, NaN with one donor), `recall_iqr` and
`donor_recovery_rate` (fraction of donors with recall at or above
`StabilityThresholds.recovery_recall`). Over all held-out cells: `pooled_recall`,
`precision`, `f1`. From the metadata: `n_donors_reference` (donors with at least one
cell of the state) and `eligible`.

### `lodo_stability`

A rule-based, descriptive category evaluated in this order with the defaults of
{class}`~atlasroute.StabilityThresholds`:

| Category | Rule |
|---|---|
| `insufficient_donors` | `n_donors_scored < min_donors` (3) |
| `fragile` | median recall < 0.50 **or** recovery rate < 0.50 |
| `stable` | median recall >= 0.70 **and** minimum recall >= 0.50 **and** recovery rate >= 0.80 |
| `moderate` | otherwise |

These are not claims of biological equivalence. Keep the raw columns; the category is a
reading aid. The thresholds are configurable and should be predeclared in a benchmark.

## Abstention in v0.1

`LODOConfig.abstain_below` predicts `unknown` when the top probability is below the threshold.
With the kNN backend that probability is a neighbour vote fraction, so it flags ambiguous cells
between two known states well but out-of-reference cells poorly: a novel state that lies in
empty latent space still gets a unanimous vote from its nearest known state. Use it to explore
ambiguity; rely on the v0.2 novelty statistics for out-of-reference detection.

## Random split

{meth}`~atlasroute.AtlasReference.random_split` runs a stratified K-fold over cells,
ignoring donors. Its accuracy is the conventional, optimistic estimate; AtlasRoute
reports it so the donor effect is visible as a gap rather than hidden.

## Known optimism of v0.1

The latent space (scVI, PCA, ...) was fitted with the held-out donor present. LODO in
v0.1 therefore measures classifier transfer within a fixed embedding, an upper bound on
what a truly unseen donor would achieve. `LODOConfig.refit_latent=True` is reserved for
backends that can refit the embedding per fold (planned scVI backend) and raises
`NotImplementedError` in v0.1. Every result records `latent["refit_latent"] = False`.
