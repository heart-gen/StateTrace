# Grouped validation

## Leave-one-group-out

For each group *g* (a donor by default, a study with `leave_one_study_out()`): remove every cell of *g*, fit the backend on the remaining cells in
the fixed latent space, predict the cells of *g*, and record the prediction, its
probability, the second-best state, the margin and the number of nearest reference
neighbours that voted for the prediction. Concatenating the folds gives one row per
reference cell ({attr}`~statetrace.ValidationResult.cell_predictions`).

### Group x state table

For each (group, true state) present in that group: `n_cells`, `n_predicted`, `recall`,
`precision`, `f1` and `scored` (`n_cells >= LODOConfig.min_group_cells`). Abstained
cells count as misses for recall and never enter a precision denominator.

### State table

Over scored groups: `n_groups_scored`, `median_group_recall`, `min_group_recall`,
`recall_variance` (sample variance, NaN with one group), `recall_iqr` and
`group_recovery_rate` (fraction of groups with recall at or above
`StabilityThresholds.recovery_recall`). Over all held-out cells: `pooled_recall`,
`precision`, `f1`. From the metadata: `n_groups_reference` (groups with at least one
cell of the state) and `eligible`.

### `reproducibility`

A rule-based, descriptive category evaluated in this order with the defaults of
{class}`~statetrace.StabilityThresholds`:

| Category | Rule |
|---|---|
| `insufficient` | `n_groups_scored < min_groups` (3) |
| `fragile` | median recall < 0.50 **or** recovery rate < 0.50 |
| `stable` | median recall >= 0.70 **and** minimum recall >= 0.50 **and** recovery rate >= 0.80 |
| `moderate` | otherwise |

These are not claims of biological equivalence. Keep the raw columns; the category is a
reading aid. The thresholds are configurable and should be predeclared in a benchmark.

## External backends

An {class}`~statetrace.AnnDataBackend` (for example {class}`~statetrace.CallableBackend` around
scANVI, scPoli or a logistic-regression classifier) is refitted on the training groups of every
fold and returns {class}`~statetrace.Predictions` for the held-out cells. If a tool was run in
its own holdout loop, {func}`~statetrace.summarize_holdout` builds the same
{class}`~statetrace.ValidationResult` from the held-out table.

## Abstention

`LODOConfig.abstain_below` predicts `unknown` when the top probability is below the threshold.
With the kNN backend that probability is a neighbour vote fraction, so it flags ambiguous cells
between two known states well but out-of-reference cells poorly: a novel state that lies in
empty latent space still gets a unanimous vote from its nearest known state. Use it to explore
ambiguity; rely on the v0.2 novelty statistics for out-of-reference detection.

### Distance calibration

With the built-in kNN, every fold also records the distance of each held-out cell to the
centroid of its state built from the training groups. The per-state mean, sd and quantiles of
those distances (`ValidationResult.distance_calibration`) are the null distribution for a genuine
cell of the state from an unseen donor; transport uses them to standardise novelty.

## Random split

{meth}`~statetrace.AtlasReference.random_split` runs a stratified K-fold over cells,
ignoring groups. Its accuracy is the conventional, optimistic estimate; StateTrace
reports it so the donor effect is visible as a gap rather than hidden.

## Known optimism of v0.1

With the built-in kNN the latent space (scVI, PCA, ...) was fitted with the held-out group present. Grouped holdout therefore measures classifier transfer within a fixed embedding, an upper bound on
what a truly unseen donor would achieve. `LODOConfig.refit_latent=True` is reserved for
backends that can refit the embedding per fold (planned scVI backend) and raises
`NotImplementedError` in v0.1. Every result records `latent["refit_latent"] = False`.
