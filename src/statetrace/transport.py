"""
Cross-cohort transport validation and inference gating.

:func:`validate_transport` takes a frozen reference, an independent query and either the
built-in kNN mapping or a :class:`~statetrace.predictions.Predictions` object from any
external tool, and answers, per state: is it reproducible in the reference; is it recovered
in enough query groups (donors); is the query version shifted, uncertain or partly out of
reference; and therefore is it **supported** for group-level downstream inference.

Two coordinate systems feed the evidence. **State identity** (which state a cell belongs to,
its confidence, and the per-cell reference distance) comes from the *mapping space*, the
latent of whichever tool produced the predictions. **State shift** (whether the molecular
phenotype of a state in the query donors lies within the reference's between-donor variation)
comes from the *frozen biological space* of :mod:`statetrace.expression`, built from the
reference's expression before any query was seen. Mapping spaces are fitted with the query
present and can absorb a state-specific displacement as if it were a batch effect, so the
shift call prefers the expression space whenever the reference carries one and the query has
the genes; the latent centroid shift is kept as secondary evidence.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import expression as _expr
from . import novelty, schema
from .aggregation import pseudobulk as _pseudobulk
from .aggregation import state_abundance as _state_abundance
from .predictions import Predictions

if TYPE_CHECKING:
    from anndata import AnnData

    from .features import FeatureMap
    from .reference import StateReference

logger = logging.getLogger(__name__)

STATE_COLUMNS = (
    "ref_groups",
    "ref_reproducibility",
    "query_groups",
    "groups_observed",
    "groups_recovered",
    "recovery_rate",
    "groups_recovered_in_reference",
    "n_cells",
    "n_cells_supported",
    "n_cells_in_reference",
    "fraction_of_query",
    "median_confidence",
    "median_support",
    "confidence_source",
    "uncertain_fraction",
    "ood_fraction",
    "shift",
    "shift_level",
    "shift_in_reference",
    "shift_in_reference_level",
    "shift_source",
    "expression_shift",
    "expression_shift_level",
    "expression_shift_in_reference",
    "groups_shifted",
    "groups_scored_expression",
    "latent_shift",
    "latent_shift_level",
    "latent_shift_in_reference",
    "novel_population",
    "query_recall",
    "status",
)


@dataclass(frozen=True)
class TransportConfig:
    """
    Rules that turn per-cell predictions into group support and a per-state transport status.

    Parameters
    ----------
    min_cells_per_group
        A group *observes* a state when at least this many of its cells map to it with local
        reference support (see ``min_support``).
    min_cells_total
        Below this many supported query cells a state is ``absent``.
    min_confidence
        A group *recovers* an observed state when the median confidence of those cells is at
        least this value and their out-of-reference fraction is at most ``max_ood_fraction``.
        When the tool gives no confidence, local support stands in for it
        (``confidence_source = "support"``); with neither, the clause is skipped and every
        state is at most ``supported_with_caution``.
    uncertain_below, ood_z
        Per-cell category cut-offs; ``ood_z`` is a standardised reference distance
        (see :mod:`statetrace.novelty`).
    min_support, support_neighbors
        Local reference support is the fraction of a cell's ``support_neighbors`` nearest
        reference cells, in the tool's latent, that carry the assigned state; it is computed
        by StateTrace whenever the query latent is available and the tool gave no support of
        its own. Cells below ``min_support`` are ``uncertain`` and do not count towards a
        group observing the state, so an absent state that a non-abstaining mapper still
        labels reads ``absent``. ``None`` disables the rule.
    ood_rule, ood_quantile
        ``"z"`` flags a cell as out of reference when its standardised distance exceeds
        ``ood_z``; ``"quantile"`` flags it when its raw distance exceeds the ``ood_quantile``
        (0.95 or 0.99) of the held-out reference distances of its state, which fixes the
        false out-of-reference rate on new donors at about the nominal level (falls back to
        the z rule when the reference carries no held-out calibration).
    min_groups_recovered
        Fewer recovering groups -> ``insufficient``.
    caution_recovery_rate, caution_uncertain_fraction
        Below / above these -> ``supported_with_caution``.
    max_ood_fraction, shift_moderate, shift_high
        Out-of-reference fraction above ``max_ood_fraction`` or a shift at or above
        ``shift_high`` -> ``shifted``; between ``shift_moderate`` and ``shift_high`` -> caution.
        The shift is in reference standard-deviation units in either coordinate system: the
        median over query donors of the donor x state pseudobulk distance standardised against
        held-out reference donors (expression space), or the centroid displacement in
        per-dimension reference sd (mapping latent).
    shift_source
        ``"auto"`` uses the frozen expression space when the reference carries one and the
        query has its genes, else the mapping latent; ``"expression"`` or ``"latent"`` force
        one (an unavailable source leaves the shift ``unknown``).
    expression_layer
        Query ``layers`` key holding raw counts for the expression-space diagnostic (``None``:
        ``query.X``).
    correct_cohort_offset
        Estimate a whole-cohort offset (median per-state displacement, in the latent and in
        standardised expression) and remove it before distance and shift diagnostics; the
        offset is reported once per coordinate system.
    novelty_calibration
        ``"holdout"`` standardises reference distances against the leave-one-group-out
        distance statistics stored with the reference (falls back to in-sample when none
        exist); ``"in_sample"`` always uses the reference's own cells.
    require_reference_validation
        Treat an unvalidated reference state as ``moderate`` (caution) rather than ``stable``.
    """

    min_cells_per_group: int = 20
    min_cells_total: int = 50
    min_confidence: float = 0.6
    uncertain_below: float = 0.6
    ood_z: float = 3.0
    min_support: float | None = 0.5
    support_neighbors: int = 15
    ood_rule: str = "z"
    ood_quantile: float = 0.99
    min_groups_recovered: int = 3
    caution_recovery_rate: float = 0.8
    caution_uncertain_fraction: float = 0.2
    max_ood_fraction: float = 0.3
    shift_moderate: float = 2.0
    shift_high: float = 4.0
    require_reference_validation: bool = True
    correct_cohort_offset: bool = True
    novelty_calibration: str = "holdout"
    shift_source: str = "auto"
    expression_layer: str | None = None

    def __post_init__(self) -> None:
        if self.shift_source not in ("auto", "expression", "latent"):
            raise ValueError("shift_source must be 'auto', 'expression' or 'latent'")
        if self.ood_rule not in novelty.OOD_RULES:
            raise ValueError(f"ood_rule must be one of {novelty.OOD_RULES}")
        if self.ood_rule == "quantile" and float(self.ood_quantile) not in (0.95, 0.99):
            raise ValueError("ood_quantile must be 0.95 or 0.99")
        if self.min_support is not None and not 0 <= self.min_support <= 1:
            raise ValueError("min_support must lie in [0, 1] or be None")
        if self.support_neighbors < 1:
            raise ValueError("support_neighbors must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def transport_status(row: pd.Series | dict, cfg: TransportConfig) -> str:
    """
    Rule-based status for one state, evaluated in order.

    1. ``absent``: too few supported query cells or no observing group.
    2. ``insufficient``: the reference itself does not reproduce the state.
    3. Displaced (high shift in the deciding coordinate system, or out-of-reference fraction
       above the maximum): if the cells that *do* lie within the reference distribution still
       recover the state in enough groups with low shift, an out-of-reference population sits
       next to an otherwise intact state -> ``supported_with_caution`` (``novel_population``);
       otherwise ``shifted``.
    4. ``insufficient``: too few query groups recover the state.
    5. ``supported_with_caution`` on any caution trigger (including a tool that gives neither
       confidence nor a latent to compute support from), else ``supported``.
    """
    n_supported = row.get("n_cells_supported", row["n_cells"])
    if n_supported < cfg.min_cells_total or row["groups_observed"] == 0:
        return "absent"
    rep = row["ref_reproducibility"]
    if rep in ("fragile", "insufficient"):
        return "insufficient"
    displaced = row["shift_level"] == "high" or (
        np.isfinite(row["ood_fraction"]) and row["ood_fraction"] > cfg.max_ood_fraction
    )
    if displaced:
        coherent = row["groups_recovered_in_reference"] >= cfg.min_groups_recovered and row[
            "shift_in_reference_level"
        ] in ("low", "moderate")
        return "supported_with_caution" if coherent else "shifted"
    if row["groups_recovered"] < cfg.min_groups_recovered:
        return "insufficient"
    caution = (
        rep in ("moderate", "unvalidated")
        or row["recovery_rate"] < cfg.caution_recovery_rate
        or (
            np.isfinite(row["uncertain_fraction"])
            and row["uncertain_fraction"] > cfg.caution_uncertain_fraction
        )
        or row["shift_level"] == "moderate"
        or row.get("confidence_source", "backend") == "none"
    )
    return "supported_with_caution" if caution else "supported"


@dataclass
class TransportRoute:
    """
    Result of :func:`validate_transport`.

    Attributes
    ----------
    states
        Index = reference state; the transport evidence table and ``status``.
    groups
        Long table, one row per (query group, state) with counts, confidence and recovery.
    cells
        Per query cell: group, state, confidence, margin, support, distance, novelty,
        category.
    predictions
        The :class:`~statetrace.predictions.Predictions` that were evaluated.
    expression
        Query donor x state pseudobulk shift table from the frozen expression space
        (``None`` when the diagnostic did not run): ``group``, ``state``, ``n_cells``,
        ``distance``, ``distance_pc``, ``residual``, ``nearest``, ``genes_beyond``,
        ``shift_z``, ``shift_p`` and ``in_reference_only``.
    """

    group_key: str
    states: pd.DataFrame
    groups: pd.DataFrame
    cells: pd.DataFrame
    predictions: Predictions
    config: TransportConfig
    reference: dict[str, Any]
    created: str
    statetrace_version: str
    query: Any = field(default=None, repr=False)
    expression: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def supported_states(self) -> list[str]:
        return list(self.states.index[self.states["status"] == "supported"])

    def states_with(self, status: str | list[str]) -> list[str]:
        status = [status] if isinstance(status, str) else list(status)
        return list(self.states.index[self.states["status"].isin(status)])

    def _select(self, states: str | list[str] | None) -> list[str] | None:
        if states is None:
            return None
        if isinstance(states, str):
            if states in schema.TRANSPORT_STATUS:
                return self.states_with(states)
            return [states]
        return [str(s) for s in states]

    def summary(self) -> str:
        counts = self.states["status"].value_counts().reindex(schema.TRANSPORT_STATUS, fill_value=0)
        lines = [
            f"StateTrace transport ({self.predictions.backend}) over {self.group_key!r}: "
            f"{len(self.cells)} query cells, {self.cells['group'].nunique()} groups, {len(self.states)} reference states",
            "  status: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()),
            f"  cells: supported {float((self.cells['category'] == 'supported').mean()):.3f} | "
            f"uncertain {float((self.cells['category'] == 'uncertain').mean()):.3f} | "
            f"out_of_reference {float((self.cells['category'] == 'out_of_reference').mean()):.3f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TransportRoute {self.predictions.backend}: {len(self.states)} states, {self.cells['group'].nunique()} groups>"

    def _gated_cells(self, exclude_out_of_reference: bool) -> pd.DataFrame:
        cells = self.cells
        if exclude_out_of_reference:
            cells = cells.assign(
                state=np.where(
                    cells["category"] == "out_of_reference", schema.UNKNOWN_LABEL, cells["state"]
                )
            )
        return cells

    def state_abundance(
        self,
        *,
        states: str | list[str] | None = None,
        normalize: bool = True,
        exclude_out_of_reference: bool = True,
    ) -> pd.DataFrame:
        """
        Group x state abundance; ``states`` may be a status such as ``"supported"``.
        Out-of-reference cells are counted as ``unknown`` (so fractions keep the full
        denominator) unless ``exclude_out_of_reference=False``.
        """
        return _state_abundance(
            self._gated_cells(exclude_out_of_reference),
            group_col="group",
            state_col="state",
            states=self._select(states),
            normalize=normalize,
        )

    def pseudobulk(
        self,
        adata: AnnData | None = None,
        *,
        states: str | list[str] | None = "supported",
        layer: str | None = None,
        min_cells: int = 10,
        exclude_out_of_reference: bool = True,
    ) -> AnnData:
        """
        Pseudobulk the query for the selected states (default: only ``supported``), leaving
        out out-of-reference cells unless ``exclude_out_of_reference=False``.
        """
        adata = adata if adata is not None else self.query
        if adata is None:
            raise ValueError("pass the query AnnData (the route was built without keeping it)")
        labels = self._gated_cells(exclude_out_of_reference)["state"]
        return _pseudobulk(
            adata,
            group_key=self.group_key,
            state_labels=labels,
            states=self._select(states),
            layer=layer,
            min_cells=min_cells,
        )

    def to_obs(self, adata: AnnData | None = None) -> None:
        """Write per-cell results into ``adata.obs`` using the :mod:`statetrace.schema` keys."""
        adata = adata if adata is not None else self.query
        c = self.cells.reindex(adata.obs_names)
        adata.obs[schema.OBS_STATE] = pd.Categorical(c["state"].astype(str))
        adata.obs[schema.OBS_CONFIDENCE] = c["confidence"].to_numpy()
        adata.obs[schema.OBS_MARGIN] = c["margin"].to_numpy()
        adata.obs[schema.OBS_SUPPORT] = c["support"].to_numpy()
        adata.obs[schema.OBS_NOVELTY] = c["novelty"].to_numpy()
        adata.obs[schema.OBS_CATEGORY] = pd.Categorical(
            c["category"].astype(str), categories=list(schema.CELL_CATEGORIES)
        )
        if self.predictions.uncertainty is not None:
            adata.obs[schema.OBS_UNCERTAINTY] = self.predictions.uncertainty.reindex(
                adata.obs_names
            ).to_numpy()
        adata.uns[schema.UNS_TRANSPORT] = {
            "backend": self.predictions.backend,
            "group_key": self.group_key,
            "status": self.states["status"].to_dict(),
            "created": self.created,
        }

    def report(self, output_dir: str | Path, *, include_cells: bool = True) -> Path:
        """Write the standardised transport report (tables + Markdown) to ``output_dir``."""
        from .reporting import write_report

        return write_report(self, output_dir, include_cells=include_cells)


def _predictions_from_reference(
    reference: StateReference, query: AnnData, latent_key: str | None
) -> tuple[Predictions, np.ndarray | None]:
    key = latent_key or reference.latent_key
    if key is None or key not in query.obsm:
        raise KeyError(
            f"query.obsm[{key!r}] is required to map with the reference's backend; "
            "pass predictions= from an external tool or provide the latent."
        )
    Z = np.asarray(query.obsm[key], dtype=np.float32)
    if Z.shape[1] != reference.latent_.shape[1]:
        raise ValueError(
            f"query latent has {Z.shape[1]} dims, reference has {reference.latent_.shape[1]}"
        )
    b = reference.backend_
    P_local = b.predict_proba(Z)
    states = reference.states_
    P = np.zeros((Z.shape[0], len(states)))
    P[:, np.searchsorted(states, b.classes_)] = P_local
    labels = states[P.argmax(axis=1)]
    preds = Predictions.from_arrays(
        query.obs_names, labels, proba=P, states=list(states), backend=b.name
    )
    return preds, Z


def _expression_shift(
    reference: StateReference,
    query: AnnData,
    group_codes: np.ndarray,
    group_names: np.ndarray,
    state_codes: np.ndarray,
    in_reference: np.ndarray,
    cfg: TransportConfig,
    feature_map: FeatureMap | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Score query donor x state pseudobulks (all assigned cells, and in-reference cells only)."""
    from .features import harmonize_features

    space = reference.expression_
    assert space is not None
    fmap = feature_map or harmonize_features(space.genes, query.var_names)
    mapping = fmap.mapping[fmap.mapping["reference"].isin(space.genes)]
    if len(mapping) == 0:
        warnings.warn(
            "none of the reference expression-space genes are in the query; expression shift "
            "is unknown. Pass feature_map= for cross-species transport.",
            UserWarning,
            stacklevel=3,
        )
        return None, {"available": False, "feature_retention": 0.0}
    gene_pos = pd.Index(space.genes).get_indexer(mapping["reference"].to_numpy())
    order = np.argsort(gene_pos)
    gene_pos = gene_pos[order]
    q_cols = query.var_names.get_indexer(mapping["query"].to_numpy()[order])
    retention = len(gene_pos) / space.n_genes
    if retention < 0.9:
        warnings.warn(
            f"only {retention:.0%} of the reference expression-space genes are in the query; "
            "pseudobulk distances are computed on the retained genes and compared with a null "
            "built on all of them.",
            UserWarning,
            stacklevel=3,
        )
    X = query.layers[cfg.expression_layer] if cfg.expression_layer is not None else query.X
    X = X[:, q_cols]
    tables = []
    for only_in_ref in (False, True):
        codes = np.where(in_reference, state_codes, -1) if only_in_ref else state_codes
        t, prov = _expr.score_query(
            space,
            X,
            group_codes,
            codes,
            gene_index=gene_pos,
            min_cells=cfg.min_cells_per_group,
            correct_cohort_offset=cfg.correct_cohort_offset,
        )
        t["in_reference_only"] = only_in_ref
        tables.append((t, prov))
    table = pd.concat([t for t, _ in tables], ignore_index=True)
    table["group"] = np.asarray(group_names).astype(str)[table["group"].to_numpy()]
    prov = {"available": True, **tables[0][1]}
    return table, prov


def validate_transport(
    reference: StateReference,
    query: AnnData,
    *,
    group_key: str,
    predictions: Predictions | None = None,
    latent_key: str | None = None,
    query_state_key: str | None = None,
    config: TransportConfig | None = None,
    feature_map: FeatureMap | None = None,
    keep_query: bool = True,
) -> TransportRoute:
    """
    Evaluate whether each reference state transports into ``query`` with enough group-level
    support for downstream inference.

    Parameters
    ----------
    reference
        A fitted (ideally validated and frozen) :class:`~statetrace.reference.StateReference`.
    query
        Independent cells x genes with ``obs[group_key]``.
    group_key
        Query replication unit (donor).
    predictions
        Output of an external mapper for the query cells. When omitted the reference's own
        backend maps ``query.obsm[latent_key]``.
    latent_key
        ``obsm`` key of the query embedding in the reference latent space; enables the
        reference-distance and centroid-shift diagnostics. Defaults to the reference's key.
    query_state_key
        Optional query labels for computing state recall (benchmarks only).
    config
        :class:`TransportConfig`.
    feature_map
        A :class:`~statetrace.features.FeatureMap` (for example from an ortholog table)
        aligning query features to the reference's; by default features are intersected by
        name. Only the genes of the frozen expression space are used.
    keep_query
        Keep a handle to ``query`` on the route so ``pseudobulk()`` needs no argument.
    """
    from . import __version__

    cfg = config or TransportConfig()
    if not reference.is_fitted:
        raise RuntimeError("fit() the reference before validating transport")
    if group_key not in query.obs:
        raise KeyError(f"query.obs[{group_key!r}] not found")
    states = np.asarray(reference.states_).astype(str)
    pos = {s: i for i, s in enumerate(states)}

    Z: np.ndarray | None = None
    if predictions is None:
        preds, Z = _predictions_from_reference(reference, query, latent_key)
    else:
        missing = set(query.obs_names.astype(str)) - set(predictions.obs_names)
        if missing:
            raise ValueError(f"predictions do not cover every query cell ({len(missing)} missing)")
        preds = predictions.subset(query.obs_names)
        key = latent_key or reference.latent_key
        if (
            key is not None
            and key in query.obsm
            and np.asarray(query.obsm[key]).shape[1] == reference.latent_.shape[1]
        ):
            Z = np.asarray(query.obsm[key], dtype=np.float32)

    labels = preds.labels.to_numpy().astype(object)
    known = np.isin(labels, states)
    codes = np.array([pos.get(lab, -1) for lab in labels])
    conf = preds.confidence.to_numpy(dtype=float)
    nov = np.full(len(labels), np.nan)
    dist = np.full(len(labels), np.nan)
    ood = np.zeros(len(labels), dtype=bool)
    ood_rule_applied = "z"
    shift = np.full(len(states), np.nan)
    cohort_offset = np.zeros(reference.latent_.shape[1], dtype=np.float32)
    support = preds.support.to_numpy(dtype=float) if preds.support is not None else None
    support_source = "backend" if support is not None else "none"
    if Z is not None:
        if cfg.correct_cohort_offset:
            cohort_offset = novelty.estimate_cohort_offset(
                Z, codes, reference.centroids_, min_cells=cfg.min_cells_per_group
            )
            Z = Z - cohort_offset
        calibration = (
            reference.distance_calibration_ if cfg.novelty_calibration == "holdout" else None
        )
        geometry = novelty.calibrated_geometry(reference.geometry_, calibration, states)
        dist = novelty.raw_distance(Z, codes, reference.centroids_)
        nov = novelty.reference_distance(Z, codes, reference.centroids_, geometry)
        ood, ood_rule_applied = novelty.out_of_reference_mask(
            dist,
            nov,
            codes,
            rule=cfg.ood_rule,
            z=cfg.ood_z,
            quantile=cfg.ood_quantile,
            calibration=calibration,
            states=states,
        )
        shift = novelty.centroid_shift(
            Z, codes, reference.centroids_, reference.geometry_, len(states)
        )
        if support is None:
            # StateTrace's own local support: label agreement of the nearest reference cells
            support = novelty.local_support(
                reference.latent_,
                reference.states_[reference.state_codes_],
                Z,
                codes,
                states,
                n_neighbors=cfg.support_neighbors,
                backend=reference.backend_,
            )
            support_source = "latent_knn"
    if np.isfinite(conf).any():
        confidence_source = "backend"
    elif support is not None and np.isfinite(support).any():
        confidence_source = "support"
        conf = np.array(support, dtype=float)
    else:
        confidence_source = "none"
        warnings.warn(
            f"{preds.backend}: predictions carry no confidence and no query latent is "
            "available to compute local support; group recovery is judged on counts alone "
            "and every state is at most supported_with_caution.",
            UserWarning,
            stacklevel=2,
        )
    category = novelty.cell_category(
        conf,
        nov,
        uncertain_below=cfg.uncertain_below,
        ood_z=cfg.ood_z,
        support=support,
        min_support=cfg.min_support,
        out_of_reference=ood,
    )
    category[~known] = "out_of_reference"
    supported_label = known & (
        np.ones(len(labels), dtype=bool)
        if support is None or cfg.min_support is None
        else np.nan_to_num(support, nan=1.0) >= cfg.min_support
    )
    in_ref = category != "out_of_reference"
    shift_in_ref = np.full(len(states), np.nan)
    if Z is not None:
        shift_in_ref = novelty.centroid_shift(
            Z, np.where(in_ref, codes, -1), reference.centroids_, reference.geometry_, len(states)
        )

    # frozen biological space: donor x state pseudobulk shift
    group_codes_q, group_names_q = pd.factorize(query.obs[group_key].astype(str), sort=True)
    expr_table: pd.DataFrame | None = None
    expr_prov: dict[str, Any] = {"available": False}
    expr_shift = np.full(len(states), np.nan)
    expr_shift_in = np.full(len(states), np.nan)
    groups_shifted = np.zeros(len(states), dtype=int)
    groups_scored = np.zeros(len(states), dtype=int)
    want_expr = cfg.shift_source in ("auto", "expression")
    if want_expr and reference.expression_ is not None and query.n_vars > 0:
        expr_table, expr_prov = _expression_shift(
            reference, query, group_codes_q, group_names_q, codes, in_ref, cfg, feature_map
        )
        if expr_table is not None:
            all_rows = expr_table[~expr_table["in_reference_only"]]
            in_rows = expr_table[expr_table["in_reference_only"]]
            expr_shift, groups_shifted, groups_scored = _expr.summarise_states(
                all_rows, len(states), high=cfg.shift_high
            )
            expr_shift_in, _, _ = _expr.summarise_states(in_rows, len(states), high=cfg.shift_high)
            expr_table["state"] = states[expr_table["state"].to_numpy()]
    elif cfg.shift_source == "expression":
        warnings.warn(
            "shift_source='expression' but the reference carries no expression space or the "
            "query has no genes; shift is unknown.",
            UserWarning,
            stacklevel=2,
        )

    cells = pd.DataFrame(
        {
            "group": query.obs[group_key].astype(str).to_numpy(),
            "state": np.where(known, labels, schema.UNKNOWN_LABEL),
            "confidence": conf,
            "margin": preds.margin().to_numpy(),
            "second_best": preds.second_best().to_numpy(),
            "support": support if support is not None else np.nan,
            "distance": dist,
            "novelty": nov,
            "category": category,
        },
        index=pd.Index(query.obs_names.astype(str), name="cell"),
    )
    cells["_supported_label"] = supported_label
    if query_state_key is not None:
        cells["true_state"] = query.obs[query_state_key].astype(str).to_numpy()

    # group x state
    g = cells.groupby(["group", "state"], sort=True)
    groups = g.agg(
        n_cells=("state", "size"),
        n_supported=("_supported_label", "sum"),
        median_confidence=("confidence", "median"),
        median_support=("support", "median"),
        uncertain_fraction=("category", lambda c: float((c == "uncertain").mean())),
        ood_fraction=("category", lambda c: float((c == "out_of_reference").mean())),
    ).reset_index()
    groups["n_supported"] = groups["n_supported"].astype(int)
    per_group_total = cells.groupby("group").size()
    groups["fraction_of_group"] = (
        groups["n_cells"] / groups["group"].map(per_group_total).to_numpy()
    )
    g_in = (
        cells[in_ref]
        .groupby(["group", "state"], sort=True)
        .agg(
            n_in_reference=("state", "size"),
            median_confidence_in_reference=("confidence", "median"),
        )
        .reset_index()
    )
    groups = groups.merge(g_in, on=["group", "state"], how="left")
    groups["n_in_reference"] = groups["n_in_reference"].fillna(0).astype(int)
    groups["observed"] = groups["n_supported"] >= cfg.min_cells_per_group
    if confidence_source == "none":
        conf_ok = pd.Series(True, index=groups.index)
        conf_ok_in = conf_ok
    else:
        conf_ok = groups["median_confidence"].fillna(0) >= cfg.min_confidence
        conf_ok_in = groups["median_confidence_in_reference"].fillna(0) >= cfg.min_confidence
    groups["recovered"] = (
        groups["observed"] & conf_ok & (groups["ood_fraction"] <= cfg.max_ood_fraction)
    )
    groups["recovered_in_reference"] = (
        groups["n_in_reference"] >= cfg.min_cells_per_group
    ) & conf_ok_in
    groups = groups[groups["state"] != schema.UNKNOWN_LABEL]
    if expr_table is not None:
        e = expr_table[~expr_table["in_reference_only"]][
            ["group", "state", "distance", "shift_z", "shift_p"]
        ].rename(
            columns={
                "distance": "expression_distance",
                "shift_z": "expression_shift_z",
                "shift_p": "expression_shift_p",
            }
        )
        groups = groups.merge(e, on=["group", "state"], how="left")
    else:
        groups["expression_distance"] = np.nan
        groups["expression_shift_z"] = np.nan
        groups["expression_shift_p"] = np.nan
    groups["shifted"] = groups["expression_shift_z"].to_numpy(dtype=float) >= cfg.shift_high

    # state table
    n_groups_query = int(cells["group"].nunique())
    gs = groups.groupby("state")
    st = (
        pd.DataFrame(
            {
                "groups_observed": gs["observed"].sum(),
                "groups_recovered": gs["recovered"].sum(),
                "groups_recovered_in_reference": gs["recovered_in_reference"].sum(),
                "n_cells": gs["n_cells"].sum(),
                "n_cells_supported": gs["n_supported"].sum(),
                "n_cells_in_reference": gs["n_in_reference"].sum(),
            }
        )
        .reindex(states)
        .fillna(0)
        .astype(int)
    )
    st["query_groups"] = n_groups_query
    st["recovery_rate"] = np.where(
        st["groups_observed"] > 0, st["groups_recovered"] / st["groups_observed"].clip(lower=1), 0.0
    )
    st["fraction_of_query"] = st["n_cells"] / len(cells)
    cs = cells[cells["state"] != schema.UNKNOWN_LABEL].groupby("state")
    st["median_confidence"] = cs["confidence"].median().reindex(states)
    st["median_support"] = cs["support"].median().reindex(states)
    st["confidence_source"] = confidence_source
    st["uncertain_fraction"] = (
        cs["category"].apply(lambda c: float((c == "uncertain").mean())).reindex(states)
    )
    st["ood_fraction"] = (
        cs["category"].apply(lambda c: float((c == "out_of_reference").mean())).reindex(states)
    )
    st["latent_shift"] = shift
    st["latent_shift_level"] = novelty.shift_level(
        shift, moderate=cfg.shift_moderate, high=cfg.shift_high
    )
    st["latent_shift_in_reference"] = shift_in_ref
    st["expression_shift"] = expr_shift
    st["expression_shift_level"] = novelty.shift_level(
        expr_shift, moderate=cfg.shift_moderate, high=cfg.shift_high
    )
    st["expression_shift_in_reference"] = expr_shift_in
    st["groups_shifted"] = groups_shifted
    st["groups_scored_expression"] = groups_scored
    use_expr = np.isfinite(expr_shift) & (cfg.shift_source != "latent")
    use_lat = ~use_expr & np.isfinite(shift) & (cfg.shift_source != "expression")
    st["shift_source"] = np.where(use_expr, "expression", np.where(use_lat, "latent", "none"))
    st["shift"] = np.where(use_expr, expr_shift, np.where(use_lat, shift, np.nan))
    st["shift_level"] = novelty.shift_level(
        st["shift"], moderate=cfg.shift_moderate, high=cfg.shift_high
    )
    st["shift_in_reference"] = np.where(
        use_expr, expr_shift_in, np.where(use_lat, shift_in_ref, np.nan)
    )
    st["shift_in_reference_level"] = novelty.shift_level(
        st["shift_in_reference"], moderate=cfg.shift_moderate, high=cfg.shift_high
    )
    st["novel_population"] = st["ood_fraction"].fillna(0) > cfg.max_ood_fraction
    meta = (
        reference.metadata_.state_table.reindex(states) if reference.metadata_ is not None else None
    )
    st["ref_groups"] = meta["n_groups"].fillna(0).astype(int) if meta is not None else 0
    if reference.validation_ is not None:
        st["ref_reproducibility"] = (
            reference.validation_.state_metrics["reproducibility"]
            .reindex(states)
            .fillna("insufficient")
        )
    else:
        st["ref_reproducibility"] = "unvalidated" if cfg.require_reference_validation else "stable"
    if query_state_key is not None:
        true = cells["true_state"].to_numpy()
        pred = cells["state"].to_numpy()
        st["query_recall"] = [
            float(((true == s) & (pred == s)).sum() / max((true == s).sum(), 1))
            if (true == s).any()
            else np.nan
            for s in states
        ]
    else:
        st["query_recall"] = np.nan
    st["status"] = [transport_status(row, cfg) for _, row in st.iterrows()]
    st.index.name = "state"
    st = st[list(STATE_COLUMNS)]
    cells = cells.drop(columns="_supported_label")

    route = TransportRoute(
        group_key=group_key,
        states=st,
        groups=groups.reset_index(drop=True),
        cells=cells,
        predictions=preds,
        config=cfg,
        reference={
            "reference_id": reference.reference_id_,
            "group_key": reference.group_key,
            "state_key": reference.state_key,
            "latent": dict(reference.latent_provenance_),
            "validated": reference.validation_ is not None,
            "latent_available": Z is not None,
            "novelty_calibration": (
                "holdout"
                if Z is not None
                and cfg.novelty_calibration == "holdout"
                and reference.distance_calibration_ is not None
                else ("in_sample" if Z is not None else None)
            ),
            "confidence_source": confidence_source,
            "support": {
                "source": support_source,
                "n_neighbors": cfg.support_neighbors if support_source == "latent_knn" else None,
            },
            "ood_rule": ood_rule_applied if Z is not None else None,
            "cohort_offset_norm": float(np.linalg.norm(cohort_offset)) if Z is not None else None,
            "cohort_offset_per_dim_sd": (
                float(
                    np.linalg.norm(cohort_offset)
                    / np.nanmedian(reference.geometry_[:, 0] / np.sqrt(reference.latent_.shape[1]))
                )
                if Z is not None
                else None
            ),
            "expression": expr_prov,
        },
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        statetrace_version=__version__,
        query=query if keep_query else None,
        expression=expr_table,
    )
    logger.info("transport: %s", route.summary().replace("\n", " | "))
    return route


def route_to_dir(route: TransportRoute, path: str | Path, *, include_cells: bool = True) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    route.states.to_csv(path / "states.csv")
    route.groups.to_csv(path / "groups.csv", index=False)
    if include_cells:
        route.cells.to_csv(path / "cells.csv.gz", index_label="cell")
    (path / "config.json").write_text(
        json.dumps(
            {
                "group_key": route.group_key,
                "backend": route.predictions.backend,
                "config": route.config.to_dict(),
                "reference": route.reference,
                "created": route.created,
                "statetrace_version": route.statetrace_version,
            },
            indent=2,
        )
    )
    return path
