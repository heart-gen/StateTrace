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

`confidence` (from the backend), `margin`, `support`, `distance` and `novelty` (raw and
standardised distance to the assigned state's reference centroid: how many reference
standard deviations above the typical distance of that state's own cells), `knn_distance` and
`knn_novelty` (raw and standardised distance to the k-th nearest reference cell),
`joint_novelty` (the larger of the two standardised distances; all need the query in the
reference latent, after removing the estimated cohort offset) and `category`:
`out_of_reference` by the out-of-reference rule below, else `uncertain` when confidence is
below `uncertain_below` or local support is below the state's support threshold, else
`supported`. Labels the reference does not know are `unknown` and `out_of_reference`.

**Local reference support.** `support` is the fraction of a cell's nearest *reference*
cells, in the tool's latent, that carry the state the tool assigned (the reference's kNN
backend supplies the neighbours when it is one; otherwise a cosine kNN with
`support_neighbors`, 15). A tool that reports its own support (Symphony and Azimuth mapping
scores) keeps it; otherwise StateTrace computes it whenever the query latent is available,
so the judgement does not depend on the mapper abstaining or on how it scores confidence. A
label whose neighbourhood is made of other states has no reference support: the cell is
`uncertain` and does not count towards its donor observing the state, which is how a state
absent from the query still reads `absent` under a mapper that labels every cell. The
threshold is per state: with `support_calibration="holdout"` (default) it is the smaller of
`min_support` (0.5) and the 5th percentile of the support that held-out reference cells of
that state achieved in grouped validation, so a state whose own cells are mixed is not asked
for more than it gives (`support_threshold` in the state table). `route.reference["support"]`
records the source (`backend`, `latent_knn` or `none`).

**Labels-only predictions.** When a tool gives labels but no confidence, local support stands
in for confidence (`confidence_source = "support"`, recorded per state and in
`route.reference`). With neither confidence nor a latent, donor recovery is judged on counts
alone, `confidence_source` is `none`, and every state is at most `supported_with_caution`.

**Out-of-reference rule** (`ood_score`, `ood_rule`). The score is the evidence:
`"knn"` (default; distance to the k-th nearest reference cell, which sees a population that
sits at a normal radius but in a different direction, or inside the long axis of an elongated
neighbouring state), `"centroid"` (distance to the state's reference centroid; the 0.1.0
default) or `"joint"` (the larger of the two standardised distances). The rule
`"z"` (default) flags a cell whose standardised score exceeds `ood_z` (3); `"quantile"` flags
a cell whose score exceeds the `ood_quantile` (0.95 or 0.99) of the held-out reference cells
of its state, stored by grouped validation, so genuine cells of an intact state from a new
donor are flagged at about the nominal rate. The kNN and joint scores and the quantile rule
need that calibration; without it the centroid z rule is applied and
`route.reference["ood_rule"]`, `["ood_score"]` and `["ood_fallback"]` say so. The
benchmark's withheld-state curve (`statetrace_benchmark.simulation.withheld_state`) traces
power and false rate of every score and rule over state separation; the rule used on real
data is predeclared from it. The default score was chosen from its elongated-state arm
(S1.3b): with stretched states and the withheld state between two, the kNN score ranked
withheld cells at mean AUROC 0.86 against 0.68 for the centroid over scANVI, scPoli and
Symphony, with no loss on round states (0.95 both) and a q95 false rate of 0.042 against
0.048.

**Held-out evidence.** Grouped validation stores, per reference cell, the three quantities
measured against the training donors only (`ValidationResult.holdout_evidence`: centroid
distance, kNN distance, support) and summarises them per state in
`ValidationResult.distance_calibration` (mean, sd, 95th and 99th percentiles of both
distances, quantiles of the joint score, 5th and 10th percentiles of support). Both travel
with the frozen reference.

**Population call** (`population_calibration`). A state carries `novel_population`, and
counts as displaced in the status rule, when its out-of-reference fraction exceeds a
threshold that is, per state, the larger of `population_floor` (0.1) and the largest
fraction any held-out reference donor produced under the active rule (`ood_threshold` in the
state table; `"fixed"` uses `max_ood_fraction`, 0.3, which also remains the per-donor
recovery limit). The call therefore means "more than any reference donor showed, and at
least one cell in ten".

## Per donor x state

`n_cells`, `n_supported` (cells at or above `min_support`), `fraction_of_group`,
`median_confidence`, `median_support`, `uncertain_fraction`, `ood_fraction`, `observed` (at
least `min_cells_per_group` supported cells) and `recovered` (observed, median confidence at
least `min_confidence` when a confidence exists, out-of-reference fraction at most
`max_ood_fraction`).

## Per state

| Column | Meaning |
|---|---|
| `ref_groups`, `ref_reproducibility` | donors supporting the state in the reference and its grouped-holdout category (`unvalidated` if no holdout ran) |
| `ref_reproducibility_latent`, `ref_reproducibility_expression` | the grouped-holdout category in the mapping latent and in the frozen expression space (empty when the reference was judged in the latent only); `ref_reproducibility` is the one `reproducibility_space` chose, recorded in `route.reference["reproducibility_space"]` |
| `groups_observed`, `groups_recovered`, `recovery_rate` | query donors observing and recovering the state |
| `n_cells`, `n_cells_supported` | query cells assigned to the state, and those with local reference support |
| `median_confidence`, `median_support`, `uncertain_fraction`, `ood_fraction` | over the query cells assigned to the state |
| `support_threshold`, `ood_threshold` | the per-state support minimum and out-of-reference fraction limit in force (calibrated on held-out reference donors, or the fixed defaults) |
| `confidence_source` | `backend`, `support` (local support stands in for a missing confidence) or `none` |
| `shift`, `shift_level`, `shift_source` | the deciding shift, in reference sd units, and the coordinate system it came from (`expression` when the reference carries an expression space and the query has its genes, else `latent`): low (< 2) / moderate / high (>= 4) / unknown |
| `expression_shift`, `expression_shift_level`, `groups_shifted`, `groups_scored_expression` | median over query donors of the donor x state pseudobulk distance standardised against held-out reference donors; donors at or above the high threshold, of donors with enough cells |
| `latent_shift`, `latent_shift_level` | centroid shift in the mapping latent after donor-offset removal, in per-dimension reference sd (secondary evidence) |
| `latent_shift_raw`, `expression_shift_raw` | the same two shifts before any donor offset was removed; a state whose raw and residual values agree is displaced on its own, one whose raw value is large and residual small moved with its donor |
| `n_cells_in_reference`, `groups_recovered_in_reference`, `shift_in_reference`, `expression_shift_in_reference`, `latent_shift_in_reference` | the same evidence restricted to cells that lie within the reference distribution |
| `novel_population` | an out-of-reference population (above the state's `ood_threshold`) sits next to the state |
| `query_recall` | only when query labels are given (benchmarks) |
| `status` | the gating decision |

## Status rules (evaluated in order)

| Status | Rule (defaults of {class}`~statetrace.TransportConfig`) |
|---|---|
| `absent` | fewer than `min_cells_total` (50) supported query cells or no donor observes it |
| `insufficient` (reference) | reference reproducibility `fragile` or `insufficient` |
| `shifted` | `shift_level` high (at least 4 reference sd in the deciding coordinate system) or `ood_fraction` above the state's `ood_threshold`, unless the in-reference cells still recover the state in enough groups with low shift, in which case an out-of-reference population sits beside an intact state -> `supported_with_caution` with `novel_population` |
| `insufficient` (query) | fewer than `min_groups_recovered` (3) donors recover it |
| `supported_with_caution` | reference reproducibility `moderate` or `unvalidated`, recovery rate below 0.8, uncertain fraction above 0.2 (or, with `caution_calibration="holdout"`, recovery or uncertainty measurably worse than in held-out reference donors; below), moderate shift, or no confidence evidence at all (`confidence_source` `none`) |
| `supported` | otherwise |

The rules are descriptive defaults meant to be predeclared in a benchmark and reported with
the raw columns, not tuned to make a state pass. `caution_reasons` lists the triggers a
cautioned state met (`novel_population`, `reproducibility`, `recovery`, `uncertainty`,
`shift`, `no_confidence`).

### Caution against held-out reference donors

`supported_with_caution` says the state is the reference's population and may enter
downstream analyses, but its labels are less trustworthy in some donors, so a result in that
state could partly reflect labelling. The fixed cut-offs (recovery below 0.8, uncertain
fraction above 0.2) read that off noisy donor counts and ignore how the reference itself
behaves: with 11 donors, 11/11 recovering has a 95% interval of 0.74-1.00, and a state whose
own reference donors are 25% uncertain is cautioned in every query.

`TransportConfig(caution_calibration="holdout")` (the default; `"fixed"` restores the cut-offs) asks instead whether the query's donors do
worse than a new reference donor does. Every held-out donor of grouped validation and every
query donor is measured the same way, from StateTrace's own evidence in the mapper's latent
(local support in place of confidence, the active out-of-reference rule, the per-state
support thresholds), never from the mapper's confidence, which only the query has
(`novelty.group_evidence_table`). A state is cautioned for

- **recovery** when the query recovers it in a lower share of observing donors than the
  held-out reference (one-sided Fisher exact test, `p < caution_alpha`, default 0.05) by more
  than `caution_margin` (0.1): `cmp_recovery_rate`, `ref_recovery_rate`, `recovery_p`;
- **uncertainty** when the per-donor uncertain fractions are higher (one-sided Mann-Whitney U,
  `p < caution_alpha`) with a median more than `caution_margin` above:
  `cmp_uncertain_fraction`, `ref_uncertain_fraction`, `uncertain_p`.

A state seen in fewer than `min_groups_recovered` held-out donors keeps the fixed cut-offs
(`caution_basis`). Sample size enters the test, not the label: a small cohort that behaves
like the reference is `supported`, and caution names a cohort-specific loss of label
reliability. The null should carry the same kinds of difference the query does: a query
from a new study is judged against study-held-out donors with
`LODOConfig(calibration_unit="study")`. On the simulator (a two-study reference, a query
from a third cohort with its own offset; 60 stable states per row), stable states whose
clusters overlap (separation 1) were cautioned for recovery or uncertainty in 43% of cases
under the fixed cut-offs, 13% against the donor-held-out null and 3% against the
study-held-out null; at separation 1.5, 0%, 2% and 2%.

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

## Two-sample engines (reported, not decisive)

`TransportConfig(two_sample=TwoSampleConfig(...))` adds `route.two_sample`
({func}`statetrace.twosample.state_tests`): for every reference state, a two-sample test of the
reference against the query cells assigned to it, by each engine, at two levels and over two
subsets (all assigned cells, and in-reference cells only).

| engine | test | needs |
|---|---|---|
| `edistance` | E-distance permutation test (Peidli et al. 2024; Euclidean, `n^2` within-group means as in pertpy, the larger side subsampled to the smaller) | nothing |
| `scxmatch` | Rosenbaum's cross-match test via scXMatch (Moeller et al. 2026, doi:10.1093/bioinformatics/btag706; kNN graph with `k`, squared Euclidean) | `conda install scxmatch -c conda-forge -c bioconda` |

| level | reference side | query side |
|---|---|---|
| `cell` | up to `ExpressionConfig.cell_sample_per_state` reference cells per state, frozen in the cell-level space (log-normalised over the space's genes, standardised and projected on `cell_components` PCs fitted on those reference cells) | the assigned query cells, projected with the frozen scale and loadings |
| `donor` | the held-out reference donor profiles of the state (each donor standardised against the space rebuilt without it: the donors of the pseudobulk null) | the query donor x state profiles, standardised and offset-corrected the same way |

Each row has `statistic` (the E-distance, or scXMatch's z-score), `p_value`, `z_score`, for
scXMatch `effect_strength_ratio` and `coverage`, the sizes and donors on each side, and a
`status` (`ok`, `unsupported` when the engine or the frozen cell space is missing, `skipped`
when a side is below `min_cells` or `min_groups`, `error`) with a `reason`. The table is
written to `two_sample.csv` and summarised in the report.

These tests never change `status`. Two readings matter. First, a cell-level test treats cells
of one donor as independent: a query of genuinely new, unshifted donors already differs from
the reference's donors at cell level, so a significant cell-level p-value alone is not a
transport failure; the donor-level row is the replicate-aware version of the same test.
Second, a two-sample test asks whether *any* difference exists, while the pseudobulk shift asks
whether the difference exceeds the reference's own between-donor variation, which is what
donor-level inference needs. The cell-level tests are the sensitive evidence for a change in
spread, which a pseudobulk mean is not built to see.

The tests compare frozen reference states with mapper assignments; nothing about the states is
inferred from the query (see {doc}`concepts`, *Freeze before you look at the query*). A query
sharing a donor with the reference raises unless `allow_group_overlap=True` (the explicit
cell-level control); `route.reference["independence"]` records the check.

## Cohort offset

An effect that moves all of a donor's cells the same way is estimated and removed before the
distance and shift diagnostics (`TransportConfig.correct_cohort_offset`), in both coordinate
systems, and the unit is the donor in both. In the latent, each query donor's offset is the
geometric median (the point closest in summed Euclidean distance, so one shifted state
does not leak in) over its states (with at least `min_cells_per_group` cells, and at
least `offset_min_states` such states, default 3) of the displacement of its state means
from the reference centroids (`latent_group_offset`, default on); donors carrying fewer
states receive the cohort-wide median offset. The held-out null is built the same way, each
held-out reference donor's offset removed against the training centroids, so query and null
are measured alike. This matters for mappers that fit the query by surgery: they place each
query donor with its own displacement, which a single cohort offset leaves behind, and on
the simulation grid that raised the false out-of-reference rate in the scANVI and scPoli
latents to two or three times nominal while the kNN latent stayed on it. In expression
space each donor's gene-wise median standardised profile over the states it carries (at
least `ExpressionConfig.offset_min_states`, default 3) is subtracted before scoring, for
held-out reference donors when the null is built and for query donors alike
(`group_offset`). A state-specific shift survives the median as long as fewer than half of a
donor's states shift; a shift that moves every state of a donor by the same vector is
indistinguishable from a batch effect and is removed. The sizes are reported once in
`route.reference["latent_offset"]` (mode, donors with their own offset, mean offset in
per-dimension reference sd), `route.reference["cohort_offset_per_dim_sd"]` (the cohort
median, latent) and `route.reference["expression"]["cohort_offset_rms"]` (expression)
rather than as shift in every state, and per donor in `groups` (`latent_offset_sd`,
`expression_offset_rms`). Both the raw displacement and the residual after offset removal
are kept (`*_shift_raw` per state, `expression_distance_raw` per donor by state), so what
the offset took out is visible next to what remains. Per-donor centring is what makes the pseudobulk test
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
