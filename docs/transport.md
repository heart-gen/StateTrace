# Transport validation and inference gating

{func}`~statetrace.validate_transport` evaluates a frozen reference against an independent
query and returns a {class}`~statetrace.TransportRoute`.

## Per cell

`confidence` (from the backend), `margin`, `support` (neighbour vote fraction or mapping
score), `novelty` (standardised distance to the assigned state's reference centroid: how many
reference standard deviations above the typical distance of that state's own cells; needs the
query in the reference latent, after removing the estimated cohort offset) and `category`:
`out_of_reference` when novelty exceeds `ood_z`, else `uncertain` when confidence is below
`uncertain_below`, else `supported`. Labels the reference does not know are `unknown` and
`out_of_reference`.

## Per donor x state

`n_cells`, `fraction_of_group`, `median_confidence`, `uncertain_fraction`, `ood_fraction`,
`observed` (at least `min_cells_per_group` cells) and `recovered` (observed, median confidence
at least `min_confidence`, out-of-reference fraction at most `max_ood_fraction`).

## Per state

| Column | Meaning |
|---|---|
| `ref_groups`, `ref_reproducibility` | donors supporting the state in the reference and its grouped-holdout category (`unvalidated` if no holdout ran) |
| `groups_observed`, `groups_recovered`, `recovery_rate` | query donors observing and recovering the state |
| `median_confidence`, `uncertain_fraction`, `ood_fraction` | over the query cells assigned to the state |
| `shift`, `shift_level` | centroid shift after cohort-offset removal, in per-dimension reference standard deviations: low (< 2) / moderate / high (>= 4) / unknown |
| `n_cells_in_reference`, `groups_recovered_in_reference`, `shift_in_reference` | the same evidence restricted to cells that lie within the reference distribution |
| `novel_population` | an out-of-reference population (above `max_ood_fraction`) sits next to the state |
| `query_recall` | only when query labels are given (benchmarks) |
| `status` | the gating decision |

## Status rules (evaluated in order)

| Status | Rule (defaults of {class}`~statetrace.TransportConfig`) |
|---|---|
| `absent` | fewer than `min_cells_total` (50) query cells or no donor observes it |
| `insufficient` (reference) | reference reproducibility `fragile` or `insufficient` |
| `shifted` | `shift_level` high (at least 4 per-dimension sd) or `ood_fraction` above `max_ood_fraction` (0.3), unless the in-reference cells still recover the state in enough groups with low shift, in which case an out-of-reference population sits beside an intact state -> `supported_with_caution` with `novel_population` |
| `insufficient` (query) | fewer than `min_groups_recovered` (3) donors recover it |
| `supported_with_caution` | reference reproducibility `moderate` or `unvalidated`, recovery rate below 0.8, uncertain fraction above 0.2, or moderate shift |
| `supported` | otherwise |

The rules are descriptive defaults meant to be predeclared in a benchmark and reported with
the raw columns, not tuned to make a state pass.

## Backend robustness

{func}`~statetrace.compare_backends` takes `{name: Predictions or TransportRoute}` for the
same cells and reports the all-backend cell agreement, pairwise agreement, per-state
concordance (cells every backend places in the state over cells any backend places there)
and, for routes, whether the status agrees.

## Downstream

`route.state_abundance(states="supported")`, `route.pseudobulk(states="supported")` and
`route.report(output_dir)`. Both outputs leave out out-of-reference cells by default
(`exclude_out_of_reference=False` to keep them), so a novel population absorbed into a state's
label never enters that state's pseudobulk.

A state that is entirely displaced cannot be told apart from an absent state replaced by a novel
population from geometry alone; both read `shifted`, never `supported`.

## Cohort offset

A study or batch effect that moves every query cell the same way is estimated as the
component-wise median of the per-state centroid displacements and removed before the distance
and shift diagnostics (`TransportConfig.correct_cohort_offset`). Its size is reported once in
`route.reference["cohort_offset_per_dim_sd"]` rather than as shift in every state.

## Held-out novelty calibration

Grouped validation records, per state, the distances of held-out cells to centroids built from
the training donors only (`ValidationResult.distance_calibration`). These include the effect of an
unseen donor, so `validate_transport` standardises query distances against them by default
(`TransportConfig.novelty_calibration="holdout"`), falling back to in-sample statistics for a
reference that was never validated. The choice is recorded in `route.reference["novelty_calibration"]`. Held-out statistics are
wider than in-sample ones, so the trade-off is fewer false out-of-reference calls on genuine
cells from new donors against somewhat lower sensitivity to a novel population that happens to
sit near an existing state; the benchmark's withheld-state layer quantifies both.
