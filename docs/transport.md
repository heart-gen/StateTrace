# Transport validation and inference gating

{func}`~statetrace.validate_transport` evaluates a frozen reference against an independent
query and returns a {class}`~statetrace.TransportRoute`.

## Two coordinate systems

```text
                    QUERY
                      |
          +-----------+-----------+
          |                       |
    Mapping space            Frozen biological
  (scANVI, scPoli, kNN)      space (expression)
          |                       |
          v                       v
     STATE IDENTITY          STATE SHIFT
          |                       |
          +-----------+-----------+
                      v
              StateTrace evidence
```

**State identity** comes from the mapping space: the label and confidence of the tool that
produced the {class}`~statetrace.Predictions`, and, when the query sits in the reference
latent, the per-cell reference distance and `out_of_reference` category. That space was
fitted with the query present. Surgery, condition embeddings and batch correction remove
whatever separates the query donors from the reference, and a displacement confined to one
state is removed the same way as a batch effect, so a mapper's latent cannot be trusted to
show that the state has changed.

**State shift** therefore comes from the frozen biological space of
{mod}`statetrace.expression`, built from the reference's expression at
{meth}`~statetrace.StateReference.fit` and frozen with it. For each state the reference
distribution is the set of reference donor pseudobulk profiles of that state; a query donor's
pseudobulk of the state is standardised gene-wise by the reference between-donor mean and
sd, and its distance to the reference state centroid is compared with the distances of
held-out reference donors (each scored against a space rebuilt without it). The question is
the one that matters for donor-level inference: *is the molecular phenotype of state S in the
query donors within the between-donor variation of state S in the reference?* Whether
individual cells moved some number of latent units is not the question.

The two systems together separate four outcomes:

| identity (mapping space) | shift (expression space) | reading |
|---|---|---|
| assigned, confident | within reference donor range | same identity, same molecular state |
| assigned, confident | beyond reference donor range | same identity, shifted molecular state (`shifted`) |
| uncertain or disagreeing across backends | any | poor identity assignment |
| out of reference | any | out-of-reference population (`novel_population` or `shifted`) |

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
| `shift`, `shift_level`, `shift_source` | the deciding shift, in reference sd units, and the coordinate system it came from (`expression` when the reference carries an expression space and the query has its genes, else `latent`): low (< 2) / moderate / high (>= 4) / unknown |
| `expression_shift`, `expression_shift_level`, `groups_shifted`, `groups_scored_expression` | median over query donors of the donor x state pseudobulk distance standardised against held-out reference donors; donors at or above the high threshold, of donors with enough cells |
| `latent_shift`, `latent_shift_level` | centroid shift in the mapping latent after cohort-offset removal, in per-dimension reference sd (secondary evidence) |
| `n_cells_in_reference`, `groups_recovered_in_reference`, `shift_in_reference`, `expression_shift_in_reference`, `latent_shift_in_reference` | the same evidence restricted to cells that lie within the reference distribution |
| `novel_population` | an out-of-reference population (above `max_ood_fraction`) sits next to the state |
| `query_recall` | only when query labels are given (benchmarks) |
| `status` | the gating decision |

## Status rules (evaluated in order)

| Status | Rule (defaults of {class}`~statetrace.TransportConfig`) |
|---|---|
| `absent` | fewer than `min_cells_total` (50) query cells or no donor observes it |
| `insufficient` (reference) | reference reproducibility `fragile` or `insufficient` |
| `shifted` | `shift_level` high (at least 4 reference sd in the deciding coordinate system) or `ood_fraction` above `max_ood_fraction` (0.3), unless the in-reference cells still recover the state in enough groups with low shift, in which case an out-of-reference population sits beside an intact state -> `supported_with_caution` with `novel_population` |
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

## Per donor x state in expression space

`route.expression` holds one row per query donor x state (twice: over all assigned cells and
over in-reference cells only, `in_reference_only`): `distance` (RMS standardised distance to
the reference state centroid), `distance_pc` and `residual` (its parts inside and outside the
reference PCA subspace), `nearest` (distance to the closest reference donor of the state),
`genes_beyond` (fraction of genes beyond the reference donor range), `shift_z` (distance
standardised against the held-out reference-donor null of the state) and `shift_p` (empirical
upper-tail probability against that null). The `groups` table carries `expression_distance`,
`expression_shift_z`, `expression_shift_p` and `shifted` per donor x state.

Genes are harmonised by name with the expression space's genes, or through a
{class}`~statetrace.FeatureMap` from an ortholog table (`feature_map=`). When genes are
missing the distance is computed on the retained genes and compared with a null built on all
of them; a warning is raised below 90 % retention and the retention is recorded in
`route.reference["expression"]`.

## Cohort offset

A study or batch effect that moves every query cell the same way is estimated and removed
before the distance and shift diagnostics (`TransportConfig.correct_cohort_offset`), in both
coordinate systems. In the latent it is the component-wise median of the per-state centroid
displacements. In expression space the unit is the donor: a donor's genotype, age, sex or
processing batch moves all of its states together, so each donor's gene-wise median
standardised profile over the states it carries (at least
`ExpressionConfig.offset_min_states`, default 3) is subtracted before scoring, for held-out
reference donors when the null is built and for query donors alike (`group_offset`). A
state-specific shift survives the median as long as fewer than half of a donor's states
shift. Donors carrying fewer states receive the cohort-wide median offset. The sizes are
reported once in `route.reference["cohort_offset_per_dim_sd"]` (latent) and
`route.reference["expression"]["cohort_offset_rms"]` (expression, the median donor offset)
rather than as shift in every state. Per-donor centring is what makes the pseudobulk test
sensitive: the between-donor spread of pseudobulk distances is dominated by shared donor
offsets, and removing them narrows the held-out null several-fold. A platform effect whose
size differs by state is indistinguishable from biology by this estimate; the benchmark's
platform system quantifies how much of it survives.

## Held-out novelty calibration

Grouped validation records, per state, the distances of held-out cells to centroids built from
the training donors only (`ValidationResult.distance_calibration`). These include the effect of an
unseen donor, so `validate_transport` standardises query distances against them by default
(`TransportConfig.novelty_calibration="holdout"`), falling back to in-sample statistics for a
reference that was never validated. The choice is recorded in `route.reference["novelty_calibration"]`. Held-out statistics are
wider than in-sample ones, so the trade-off is fewer false out-of-reference calls on genuine
cells from new donors against somewhat lower sensitivity to a novel population that happens to
sit near an existing state; the benchmark's withheld-state layer quantifies both.
