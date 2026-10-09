# StateTrace-Spatial: boundary stability

Planned Stage 7; no spatial API, CellART integration or benchmark result is claimed here.
See {doc}`staged-roadmap` for dependencies and the proposed G6-Spatial gate.

## 1. Biological Claim

We will test whether controlled cell-boundary errors induce apparent molecular state shifts
within spatial tissue samples, supporting the claim that segmentation stability helps
separate reconstruction-induced novelty from independently supported biological state shifts.

Core StateTrace asks: **Is this mapped state trustworthy and is it genuinely novel?**
StateTrace-Spatial asks: **Is that conclusion robust to how the spatial cell itself was
reconstructed?** Robustness across reconstructions is evidence about sensitivity, not proof
of biological correctness or definitive attribution of a pipeline failure.

| Diagnostic | StateTrace question | Implementation |
|---|---|---|
| State novelty | Is the query displaced from its assigned reference state? | Reference-frozen expression projection and donor x state pseudobulk distance; planned cell-level complement |
| Local support | Do neighboring reference cells support the assigned label? | Existing kNN/support machinery in mapping space; these are reference neighbors, not adjacent tissue cells |
| Boundary stability | Does the result survive alternative cell reconstruction? | New paired comparison of native segmentation and an alternative such as CellART |
| Reference sensitivity | Does annotation change when reference assumptions change? | Alternate or held-out references, crossed with reconstruction where feasible |

CellART jointly uses spatial, image and reference information for segmentation and annotation;
its refined boundaries support reaggregation of transcriptomic information. This makes it a
candidate preprocessing perturbation, while its reference dependence requires explicit
controls. See the [CellART preprint](https://www.biorxiv.org/content/10.64898/2026.09.03.749294v1.full)
and [Xenium tutorial](https://cellart.readthedocs.io/en/latest/tutorials/xenium_crc.html).
StateTrace will consume reconstruction outputs through an optional adapter; it will not
implement segmentation or treat CellART labels as truth.

## 2. Primary Hypothesis

At a matched donor-calibrated false-positive rate, adding boundary-stability evidence will
identify more contamination-induced false novelty than novelty alone or mapper uncertainty,
while retaining a prespecified minimum sensitivity to independently supported true shifts.
Failure to improve detection, or loss of true-shift sensitivity beyond the prespecified
tolerance, fails to support this hypothesis.

## 3. Statistical Unit and Design

Donors are independent biological units; sections, fields, states, cells and corruption
replicates are nested observations. Report every diagnostic at donor level as well as its
cell or correspondence-component level. Use paired donor summaries and donor-blocked
resampling for uncertainty; repeated random corruptions are technical replicates. A dataset
with only one donor supports a technical demonstration, not population-level inference.

Separate calibration donors from evaluation donors. Freeze references, normalization,
projection, state definitions, score scaling and thresholds before evaluation. Every
alternate reference is frozen separately, with its own calibration and provenance.

## 4. Primary Comparison

Start with a well-resolved Xenium dataset with transcript coordinates, native masks, images,
donor metadata and independent evidence for selected biological shifts. Dataset accession,
sample size and validation source remain prerequisites to benchmark registration.

Use a factorial design: unchanged versus planted molecular shifts, crossed with no corruption
versus each boundary corruption at predeclared doses. Plant molecular shifts in a simulation
or semi-synthetic arm with known truth; reserve independently supported tissue shifts for
biological validation. Do not define true shifts by CellART agreement.

Compare three segmentation arms on the same tissue:

1. Vendor/native segmentation, a baseline that may itself contain errors.
2. Deliberately corrupted segmentation: boundary dilation, boundary erosion, transcript
   swapping between adjacent cells, neighboring-cell merging, and partial cell fragmentation.
3. CellART reconstruction of the same underlying images and transcript detections.

For boundary perturbations, recompute assignments from coordinates under a deterministic
overlap rule. Transcript swapping is an explicit assignment perturbation with masks fixed.
Keep molecule identifiers and record gains, losses and unassigned transcripts. Record whether
CellART consumes corrupted masks or reconstructs independently; the latter tests an
alternative reconstruction, not recovery from those corrupted masks.

The primary contrast is classification of artificial novelty versus true shift using paired
segmentation evidence, compared with novelty alone and uncertainty alone. Include simple
IoU-only and reassignment-only baselines to test whether molecular diagnostics add value.

## 5. Covariates and Exclusions

Prespecify transcript detection QC, common tissue extent, feature panel, coordinate units,
registration tolerance, minimum reference support, matching thresholds and minimum donor x
state coverage. Record donor, section, cell density, baseline cell area, baseline transcript
count, panel retention and corruption dose. Use pairing as the primary adjustment; report
stratified performance where density or baseline coverage changes sensitivity.

Do not adjust away reassignment or post-corruption transcript loss in the primary analysis:
they are mechanisms of the perturbation. Report exclusions and their coverage by donor,
state and arm. Low coverage, ambiguous matches and unavailable uncertainty are missing or
insufficient evidence, never zero sensitivity or robust results.

## 6. Expected Effect Direction

Artificial contamination should increase transcript reassignment and paired molecular
displacement, often inducing novelty that changes across reconstructions. A true shifted
state should retain elevated frozen-expression shift across plausible reconstructions.
These are testable expectations: two methods can share a boundary error, and either can
erase a true shift. Stability alone cannot establish correctness.

## 7. Minimal Primary Analysis

For the initial native-versus-CellART workflow:

1. Run the same mapping backend and StateTrace on ordinary 10x segmentation.
2. Reconstruct the same tissue with CellART and retain its boundaries and assignments.
3. Match cells spatially using registered masks and overlap, without using mapped labels or
   novelty to select correspondences. Preserve a correspondence graph: one-to-one, split,
   merge, ambiguous and unmatched cases. Centroid distance may constrain candidates but must
   not force every cell into a one-to-one match.
4. Reaggregate the same QC-filtered detected transcripts over each set of boundaries and rerun
   the same mapping backend and StateTrace with frozen settings. Use measured counts for
   molecular diagnostics; keep any reference-imputed expression out of the primary endpoint.
5. Export paired cell/component metrics and donor x state summaries, then repeat for corrupted
   segmentations. Report the full native/corrupted/reconstructed comparison.

For one-to-one pairs, export mask intersection-over-union (IoU); gained, lost and reassigned
transcript fractions; mapped-label transition (including `unknown`); signed and absolute
novelty change; change in reference-frozen distance; displacement between frozen expression
coordinates; and uncertainty change using the same definition and backend in both arms.
Define transcript reassignment as `1 - |T_native intersect T_alternative| /
|T_native union T_alternative|` for molecule-ID sets in a matched pair; empty unions are
undefined. Include assignment-to-unassigned transitions in the audit.

For splits and merges, additionally compare union masks and transcript sets within each
correspondence component and report internal assignment changes. Union overlap alone can
conceal a merge. Do not fabricate per-cell deltas for these cases; export component summaries,
label composition and coverage separately. Include unmatched cells in donor/state coverage
denominators so the hardest errors cannot disappear from the result.

An exploratory simple score is `S_seg_simple = abs(novelty_native - novelty_CellART)`.
The preferred candidate is a standardized multicomponent score, for example
`S_seg = sqrt(sum_j w_j * z_j**2)`, with nonnegative weights summing to one and components
covering novelty change, transcript reassignment and reference-frozen embedding displacement.
Define `z_j = max(0, (x_j - median_null_j) / scale_null_j)` using a prespecified positive scale
floor. Estimate null centers, scales, weights and decision thresholds from independent
calibration donors under no-error or benign reconstruction perturbations, then freeze them.
Check redundant components and retain the individual terms; the composite is a candidate
requiring validation, not an established score. A missing required component makes it
unavailable, rather than silently changing weights. IoU, label transitions and uncertainty
remain separately interpretable outputs.

Compare molecular distances both to a fixed baseline assigned state and to each arm's
assigned state: a label change otherwise changes the target centroid and confounds distance
deltas. Use the same reference-frozen expression projection in both arms. Cell-level frozen
expression scoring is planned; until implemented, report molecular distance at donor x state
level and mark cell-level distances unavailable. Report state summaries using both fixed
baseline membership and arm-specific membership to expose composition changes.

| Interpretation | Required evidence |
|---|---|
| Robust shifted state | High novelty and elevated frozen-expression shift under both reconstructions, with adequate coverage; biological validation is still required |
| Segmentation-sensitive | Apparent novelty or shift changes materially or disappears after boundary reconstruction; direction and score components are shown |
| Identity-sensitive | Mapped label changes while molecular displacement persists in the common frozen coordinate system |
| Robustly reference-unsupported | Both reconstructions remain poorly supported by the mapped reference state; this does not establish a new biological state |

These are potentially overlapping report interpretations, not new mutually exclusive core
transport statuses. Stable supported cells remain stable supported; inadequate evidence gets
no robustness claim. Formal vocabulary and gating changes require separate implementation.

## 8. Sensitivity Analyses

Cross reconstruction with alternate or held-out references. First hold the StateTrace
reference and mapper fixed while varying boundaries; then hold boundaries fixed while varying
the annotation reference. If feasible, independently vary the reference used by CellART for
reconstruction. Record both references in every arm. A reconstruction change that also
changes reference assumptions is a combined perturbation, not an isolated boundary effect.

Assess corruption dose, matching tolerance, split/merge handling, panel retention, low counts,
donor-offset assumptions and backend choice. Use one primary endpoint; prespecify secondary
families and multiplicity handling for state-specific tests. Report effect sizes and donor
uncertainty, not pooled cell-level significance.

## 9. Orthogonal Validation

Use independent image-based boundary review or held-out membrane/nuclear stains for boundary
plausibility, and independent protein measurements, an assay, or held-out biological evidence
for state shifts. Keep validation features out of reconstruction and threshold selection when
possible. Shared references and transcript-derived labels are not orthogonal validation.
Without independent biological evidence, restrict conclusions to detection of known synthetic
errors and measured reconstruction sensitivity.

## 10. Main Figure-Worthy Result

A tissue example with native, corrupted and reconstructed masks and molecule assignments,
linked to donor-level curves showing that boundary stability flags artificial novelty while
preserving validated biological shifts. Show matched false-positive-rate performance against
novelty, uncertainty and simple geometric baselines, with coverage and split/merge outcomes.
This figure specification is a future benchmark deliverable, not a result.

## 11. Reviewer Objections and Responses

| Objection | Planned response |
|---|---|
| Agreement with CellART is circular, especially with a shared reference | Known corruption truth, independent biological evidence and crossed reference controls; agreement is never the endpoint |
| Counts, density or panel coverage explain the apparent shift | Paired design, baseline strata, measured-count diagnostics and feature-retention reporting |
| Cells or corruption seeds inflate replication | Donor-level estimates and blocked uncertainty; one-donor work is explicitly descriptive |
| Matching removes the difficult errors | Preserve splits, merges and unmatched cases; report coverage and component-level outcomes |
| Thresholds were chosen on the test tissue | Separate calibration/evaluation donors and register all score choices before evaluation |
| A complex score adds nothing over boundary QC | Compare against IoU-only and reassignment-only baselines and report effect sizes |
| Reconstruction erases real biology | Require true-shift retention and orthogonal validation; failure narrows the claim |
| One Xenium dataset supports all spatial platforms | Limit the initial claim to the tested tissue, panel and platform; broader claims require replication |

## 12. Implementation Notes

Implement in sequence: generic spatial input/correspondence contracts and synthetic corruption
fixtures; paired diagnostics and donor aggregation; then the optional CellART adapter and real
benchmark. This preserves the roadmap's one-complexity-layer-at-a-time policy.

Inputs must carry mask geometry, coordinate units/transforms, tissue extent, unique molecule
IDs and gene/coordinate/assignment tables, measured count matrices, donor/section metadata,
`Predictions`, and reference identifiers. Record tool versions, seeds, reconstruction settings,
input checksums, mapper configuration, frozen reference checksums and calibration provenance.
Validate exported CellART formats and dependency/licensing requirements when building the
adapter; do not make CellART a hard dependency or promise an unverified import API.

Candidate results comprise correspondence edges/components, paired diagnostics, donor x state
summaries and an audit manifest. Public result objects, AnnData keys and vocabulary must
follow existing dataclass and `schema.py` conventions. Keep import hygiene and unchanged core
behavior when the spatial feature is unused. Synthetic tests must cover identity
reconstruction, transcript conservation, coordinate mismatch, splits/merges, unmatched cells,
missing uncertainty and frozen-reference invariance; tests must not download real data.

The benchmark repository owns real-data execution, corruption manifests, numeric promotion
criteria and the proposed G6-Spatial report. The manuscript repository owns eventual Methods,
Results and figures. This software plan does not register a gate in either repository or
commit Stage 7 to the core release date.
