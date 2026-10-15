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

### Which space decides it

In the reference's latent (`reproducibility_space="latent"`) the category describes the state
in that latent: the same reference cells can be `stable` in one mapper's latent and `fragile`
in another's. `LODOConfig(reproducibility_space="expression")` repeats the holdout with the kNN on
every reference cell projected into the frozen cell-level expression space (log-normalised
selected genes, standardised, principal components fitted on a seeded per-state sample of
reference cells; see {func}`~statetrace.expression.cell_coordinates`). That space is built
from counts alone, so every mapper's reference gets the same call, which then decides
`reproducibility`. The latent call is kept as `reproducibility_latent` beside the latent metrics,
and the expression holdout adds `reproducibility_expression`, `n_groups_scored_expression`,
`median_group_recall_expression`, `min_group_recall_expression` and
`group_recovery_rate_expression`. It needs the reference AnnData and the cell-level space
(`ExpressionConfig.cell_sample_per_state > 0`). The expression space is fitted with every
group present: the same fixed-space optimism as a fixed latent.

The default, `"auto"`, judges in the expression space whenever the reference has it and falls
back to the latent otherwise (no counts, `expression=False`, or a loaded frozen reference);
`ValidationResult.config.reproducibility_space` records the space applied.

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
ambiguity; rely on the novelty statistics of {func}`~statetrace.validate_transport`
({doc}`transport`) for out-of-reference detection.

### Distance calibration

With the built-in kNN, every fold also records the distance of each held-out cell to the
centroid of its state built from the training groups. The per-state mean, sd and quantiles of
those distances (`ValidationResult.distance_calibration`) are the null distribution for a genuine
cell of the state from an unseen donor; transport uses them to standardise novelty.

`LODOConfig.calibration_unit` sets the holdout behind that null. `"group"` holds out one donor
at a time; `"study"` holds out one study at a time, so the null also carries the
between-study differences a donor offset does not remove, which is what a query from a new
study brings. The default, `"auto"`, uses `"study"` when the reference has a `study_key` with
at least two studies (and an array backend), else `"group"`; states seen in too few
study-held-out donors keep the group null (`distance_calibration["calibration_unit"]`). The
same held-out donors are the null for the caution triggers in transport.

## Random split

{meth}`~statetrace.Reference.random_split` runs a stratified K-fold over cells,
ignoring groups. Its accuracy is the conventional, optimistic estimate; StateTrace
reports it so the donor effect is visible as a gap rather than hidden.

## Known optimism of a fixed latent

With the built-in kNN the latent space (scVI, PCA, ...) was fitted with the held-out group present. Grouped holdout therefore measures classifier transfer within a fixed embedding, an upper bound on
what a truly unseen donor would achieve. `LODOConfig.refit_latent=True` is reserved for
backends that can refit the embedding per fold (the planned scANVI backend) and raises
`NotImplementedError` until one exists. Every result records `latent["refit_latent"] = False`.
