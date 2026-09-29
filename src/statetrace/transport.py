"""
Cross-cohort transport validation and inference gating.

:func:`validate_transport` takes a frozen reference, an independent query and either the
built-in kNN mapping or a :class:`~statetrace.predictions.Predictions` object from any
external tool, and answers, per state: is it reproducible in the reference; is it recovered
in enough query groups (donors); is the query version shifted, uncertain or partly out of
reference; and therefore is it **supported** for group-level downstream inference.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import novelty, schema
from .aggregation import pseudobulk as _pseudobulk
from .aggregation import state_abundance as _state_abundance
from .predictions import Predictions

if TYPE_CHECKING:
    from anndata import AnnData

    from .reference import StateReference

logger = logging.getLogger(__name__)

STATE_COLUMNS = (
    "ref_groups",
    "ref_reproducibility",
    "query_groups",
    "groups_observed",
    "groups_recovered",
    "recovery_rate",
    "n_cells",
    "fraction_of_query",
    "median_confidence",
    "uncertain_fraction",
    "ood_fraction",
    "shift",
    "shift_level",
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
        A group *observes* a state when at least this many of its cells map to it.
    min_cells_total
        Below this many query cells a state is ``absent``.
    min_confidence
        A group *recovers* an observed state when the median confidence of those cells is at
        least this value and their out-of-reference fraction is at most ``max_ood_fraction``.
    uncertain_below, ood_z
        Per-cell category cut-offs (see :mod:`statetrace.novelty`).
    min_groups_recovered
        Fewer recovering groups -> ``insufficient``.
    caution_recovery_rate, caution_uncertain_fraction
        Below / above these -> ``supported_with_caution``.
    max_ood_fraction, shift_moderate, shift_high
        Out-of-reference fraction above ``max_ood_fraction`` or a centroid shift at or above
        ``shift_high`` (in reference-spread units) -> ``shifted``; between ``shift_moderate``
        and ``shift_high`` -> caution.
    require_reference_validation
        Treat an unvalidated reference state as ``moderate`` (caution) rather than ``stable``.
    """

    min_cells_per_group: int = 20
    min_cells_total: int = 50
    min_confidence: float = 0.6
    uncertain_below: float = 0.6
    ood_z: float = 2.5
    min_groups_recovered: int = 3
    caution_recovery_rate: float = 0.8
    caution_uncertain_fraction: float = 0.2
    max_ood_fraction: float = 0.3
    shift_moderate: float = 1.0
    shift_high: float = 2.0
    require_reference_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def transport_status(row: pd.Series | dict, cfg: TransportConfig) -> str:
    """
    Rule-based status for one state, evaluated in order: ``absent`` (too few query cells or
    no observing group); ``insufficient`` when the reference itself does not reproduce the
    state; ``shifted`` (high centroid shift or out-of-reference fraction); ``insufficient``
    when too few query groups recover it; ``supported_with_caution``; else ``supported``.
    """
    if row["n_cells"] < cfg.min_cells_total or row["groups_observed"] == 0:
        return "absent"
    rep = row["ref_reproducibility"]
    if rep in ("fragile", "insufficient"):
        return "insufficient"
    if row["shift_level"] == "high" or (
        np.isfinite(row["ood_fraction"]) and row["ood_fraction"] > cfg.max_ood_fraction
    ):
        return "shifted"
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
        Per query cell: group, state, confidence, margin, support, novelty, category.
    predictions
        The :class:`~statetrace.predictions.Predictions` that were evaluated.
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

    def state_abundance(
        self, *, states: str | list[str] | None = None, normalize: bool = True
    ) -> pd.DataFrame:
        """Group x state abundance; ``states`` may be a status such as ``"supported"``."""
        return _state_abundance(
            self.cells,
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
    ) -> AnnData:
        """Pseudobulk the query for the selected states (default: only ``supported``)."""
        adata = adata if adata is not None else self.query
        if adata is None:
            raise ValueError("pass the query AnnData (the route was built without keeping it)")
        return _pseudobulk(
            adata,
            group_key=self.group_key,
            state_labels=self.cells["state"],
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
    support = None
    if hasattr(b, "neighbor_votes"):
        V = b.neighbor_votes(Z)
        support = V.max(axis=1) / V.sum(axis=1).clip(min=1)
    labels = states[P.argmax(axis=1)]
    preds = Predictions.from_arrays(
        query.obs_names, labels, proba=P, states=list(states), support=support, backend=b.name
    )
    return preds, Z


def validate_transport(
    reference: StateReference,
    query: AnnData,
    *,
    group_key: str,
    predictions: Predictions | None = None,
    latent_key: str | None = None,
    query_state_key: str | None = None,
    config: TransportConfig | None = None,
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
    conf = preds.confidence.to_numpy()
    nov = np.full(len(labels), np.nan)
    shift = np.full(len(states), np.nan)
    if Z is not None:
        nov = novelty.reference_distance(Z, codes, reference.centroids_, reference.spread_)
        shift = novelty.centroid_shift(
            Z, codes, reference.centroids_, reference.spread_, len(states)
        )
    category = novelty.cell_category(
        conf, nov, uncertain_below=cfg.uncertain_below, ood_z=cfg.ood_z
    )
    category[~known] = "out_of_reference"

    cells = pd.DataFrame(
        {
            "group": query.obs[group_key].astype(str).to_numpy(),
            "state": np.where(known, labels, schema.UNKNOWN_LABEL),
            "confidence": conf,
            "margin": preds.margin().to_numpy(),
            "second_best": preds.second_best().to_numpy(),
            "support": preds.support.to_numpy() if preds.support is not None else np.nan,
            "novelty": nov,
            "category": category,
        },
        index=pd.Index(query.obs_names.astype(str), name="cell"),
    )
    if query_state_key is not None:
        cells["true_state"] = query.obs[query_state_key].astype(str).to_numpy()

    # group x state
    g = cells.groupby(["group", "state"], sort=True)
    groups = g.agg(
        n_cells=("state", "size"),
        median_confidence=("confidence", "median"),
        uncertain_fraction=("category", lambda c: float((c == "uncertain").mean())),
        ood_fraction=("category", lambda c: float((c == "out_of_reference").mean())),
    ).reset_index()
    per_group_total = cells.groupby("group").size()
    groups["fraction_of_group"] = (
        groups["n_cells"] / groups["group"].map(per_group_total).to_numpy()
    )
    groups["observed"] = groups["n_cells"] >= cfg.min_cells_per_group
    groups["recovered"] = (
        groups["observed"]
        & (groups["median_confidence"].fillna(0) >= cfg.min_confidence)
        & (groups["ood_fraction"] <= cfg.max_ood_fraction)
    )
    groups = groups[groups["state"] != schema.UNKNOWN_LABEL]

    # state table
    n_groups_query = int(cells["group"].nunique())
    gs = groups.groupby("state")
    st = (
        pd.DataFrame(
            {
                "groups_observed": gs["observed"].sum(),
                "groups_recovered": gs["recovered"].sum(),
                "n_cells": gs["n_cells"].sum(),
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
    st["uncertain_fraction"] = (
        cs["category"].apply(lambda c: float((c == "uncertain").mean())).reindex(states)
    )
    st["ood_fraction"] = (
        cs["category"].apply(lambda c: float((c == "out_of_reference").mean())).reindex(states)
    )
    st["shift"] = shift
    st["shift_level"] = novelty.shift_level(shift, moderate=cfg.shift_moderate, high=cfg.shift_high)
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
        },
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        statetrace_version=__version__,
        query=query if keep_query else None,
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
