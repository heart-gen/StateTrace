"""
Grouped reference validation.

Leave-one-group-out holds out every cell of one *group* (a donor by default; an individual,
study, site or cohort through ``group_key``), fits the mapping backend on the remaining
groups, and predicts the held-out cells. Repeating over groups yields a per-cell prediction
table from which group x state, state-level and confusion summaries are built with
:mod:`statetrace.metrics`. Leave-one-study-out is the same engine keyed on a study column.

``random_split`` (stratified K-fold over cells, ignoring groups) is the conventional
cell-level benchmark; the gap to grouped holdout is the optimism of standard practice.

Any backend can take part: the built-in kNN on a fixed latent, an
:class:`~statetrace.backends.base.AnnDataBackend` refitted per fold, or precomputed held-out
predictions from an external tool through :func:`summarize_holdout`.

Known optimism: with a fixed latent (scVI, PCA, ...) fitted with the held-out group present,
grouped holdout measures classifier transfer within that embedding, an optimistic bound.
``LODOConfig.refit_latent`` is reserved for backends that refit the embedding per fold.
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

from . import metrics, novelty, schema
from .backends.base import AnnDataBackend, MappingBackend

if TYPE_CHECKING:
    from anndata import AnnData


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityThresholds:
    """Cut-offs for the rule-based ``reproducibility`` category (descriptive defaults; predeclare in a benchmark)."""

    min_groups: int = 3
    recovery_recall: float = 0.5
    stable_median_recall: float = 0.70
    stable_min_recall: float = 0.50
    stable_recovery_rate: float = 0.80
    fragile_median_recall: float = 0.50
    fragile_recovery_rate: float = 0.50


@dataclass(frozen=True)
class LODOConfig:
    """
    Options for grouped validation.

    Parameters
    ----------
    min_group_cells
        A (group, state) pair needs this many cells to count toward state-level aggregates.
    abstain_below
        If set, a held-out cell whose top probability is below this value is predicted ``unknown``.
    thresholds
        :class:`StabilityThresholds`.
    per_group_confusion
        Also keep one confusion matrix per held-out group.
    n_jobs
        Threads over folds (array backends only).
    refit_latent
        Reserved; ``True`` raises :class:`NotImplementedError`.
    random_state
        Seed for the random-split fold assignment.
    """

    min_group_cells: int = 20
    abstain_below: float | None = None
    thresholds: StabilityThresholds = field(default_factory=StabilityThresholds)
    per_group_confusion: bool = False
    n_jobs: int = 1
    refit_latent: bool = False
    random_state: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LODOConfig:
        d = dict(d)
        thr = d.pop("thresholds", {})
        return cls(thresholds=StabilityThresholds(**thr), **d)


@dataclass
class ValidationResult:
    """
    Output of a grouped validation run.

    Attributes
    ----------
    method
        ``leave_one_group_out``, ``leave_one_study_out`` or ``random_split``.
    group_key
        The ``obs`` column that defined the groups.
    state_metrics
        Index = state: group-level aggregates, pooled precision/recall/F1, ``reproducibility``.
    group_metrics
        Long table, one row per (group, state) present in that group.
    group_summary
        One row per group.
    confusion_matrix
        True states x predicted states (plus ``unknown`` when abstention happened).
    cell_predictions
        One row per reference cell (index = ``obs_names``).
    """

    method: str
    group_key: str
    state_metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    group_summary: pd.DataFrame
    confusion_matrix: pd.DataFrame
    cell_predictions: pd.DataFrame
    per_group_confusion: dict[str, pd.DataFrame] | None
    overall: dict[str, float]
    config: LODOConfig
    backend: dict[str, Any]
    latent: dict[str, Any]
    created: str
    statetrace_version: str
    distance_calibration: pd.DataFrame | None = None
    holdout_evidence: pd.DataFrame | None = None

    # donor-flavoured aliases for the common case
    @property
    def donor_metrics(self) -> pd.DataFrame:
        return self.group_metrics

    @property
    def donor_summary(self) -> pd.DataFrame:
        return self.group_summary

    def summary(self) -> str:
        o = self.overall
        counts = (
            self.state_metrics["reproducibility"]
            .value_counts()
            .reindex(schema.STABILITY_CATEGORIES, fill_value=0)
        )
        lines = [
            f"StateTrace {self.method} over {self.group_key!r}: {o['n_cells']} cells, {o['n_groups']} groups, "
            f"{o['n_states']} states, {o['n_folds']} folds",
            f"  accuracy {o['accuracy']:.3f} | balanced accuracy {o['balanced_accuracy']:.3f} | "
            f"macro F1 {o['macro_f1']:.3f} | abstention {o['abstention_rate']:.3f}",
            f"  median group balanced accuracy {o['median_group_balanced_accuracy']:.3f}",
            "  states: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()),
            f"  backend {self.backend.get('name')} {self.backend.get('params')}",
        ]
        if not self.latent.get("refit_latent", False) and self.latent.get("source") not in (
            None,
            "external",
        ):
            lines.append(
                "  note: latent space was fixed across folds (fitted with every group present); "
                "this is an optimistic bound on embedding transfer."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ValidationResult {self.method}: {self.overall['n_groups']} groups, {self.overall['n_states']} states>"

    def to_obs(self, adata: AnnData, *, prefix: str = schema.OBS_HOLDOUT_STATE) -> None:
        """Write held-out predictions back into ``adata.obs`` (aligned on ``obs_names``)."""
        cp = self.cell_predictions.reindex(adata.obs_names)
        adata.obs[prefix] = pd.Categorical(cp["predicted_state"].astype(str))
        adata.obs[f"{prefix}_probability"] = cp["probability"].to_numpy()
        adata.obs[f"{prefix}_margin"] = cp["margin"].to_numpy()
        adata.obs[f"{prefix}_abstained"] = cp["abstained"].astype(bool).to_numpy()
        adata.uns[schema.UNS_VALIDATION] = {
            "method": self.method,
            "group_key": self.group_key,
            "overall": dict(self.overall),
            "created": self.created,
            "statetrace_version": self.statetrace_version,
        }

    def to_dir(self, path: str | Path, *, include_cell_predictions: bool = True) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "method": self.method,
                    "group_key": self.group_key,
                    "config": self.config.to_dict(),
                    "backend": self.backend,
                    "latent": self.latent,
                    "created": self.created,
                    "statetrace_version": self.statetrace_version,
                },
                indent=2,
            )
        )
        (path / "overall.json").write_text(json.dumps(self.overall, indent=2))
        self.state_metrics.to_csv(path / "state_metrics.csv")
        self.group_metrics.to_csv(path / "group_metrics.csv", index=False)
        self.group_summary.to_csv(path / "group_summary.csv", index=False)
        self.confusion_matrix.to_csv(path / "confusion_matrix.csv")
        if self.distance_calibration is not None:
            self.distance_calibration.to_csv(path / "distance_calibration.csv")
        if self.holdout_evidence is not None:
            self.holdout_evidence.to_csv(path / "holdout_evidence.csv.gz", index_label="cell")
        if include_cell_predictions:
            self.cell_predictions.to_csv(path / "cell_predictions.csv.gz", index_label="cell")
        if self.per_group_confusion:
            pdc = path / "per_group_confusion"
            pdc.mkdir(exist_ok=True)
            for g, cm in self.per_group_confusion.items():
                cm.to_csv(pdc / f"{g}.csv")
        return path

    @classmethod
    def from_dir(cls, path: str | Path) -> ValidationResult:
        path = Path(path)
        meta = json.loads((path / "config.json").read_text())
        overall = json.loads((path / "overall.json").read_text())
        state_metrics = pd.read_csv(path / "state_metrics.csv", index_col="state")
        state_metrics.index = state_metrics.index.astype(str)
        group_metrics = pd.read_csv(path / "group_metrics.csv", dtype={"group": str, "state": str})
        group_summary = pd.read_csv(path / "group_summary.csv", dtype={"group": str})
        cm = pd.read_csv(path / "confusion_matrix.csv", index_col="true_state")
        cm.index = cm.index.astype(str)
        cm.columns.name = "predicted_state"
        cp_file = path / "cell_predictions.csv.gz"
        if cp_file.exists():
            cell_predictions = pd.read_csv(
                cp_file,
                index_col="cell",
                dtype={"group": str, "true_state": str, "predicted_state": str, "second_best": str},
            )
            cell_predictions.index = cell_predictions.index.astype(str)
        else:
            cell_predictions = pd.DataFrame(columns=list(metrics.CELL_COLUMNS))
        cal_file = path / "distance_calibration.csv"
        calibration = None
        if cal_file.exists():
            calibration = pd.read_csv(cal_file, index_col="state")
            calibration.index = calibration.index.astype(str)
        ev_file = path / "holdout_evidence.csv.gz"
        evidence = None
        if ev_file.exists():
            evidence = pd.read_csv(
                ev_file, index_col="cell", dtype={"group": str, "true_state": str}
            )
            evidence.index = evidence.index.astype(str)
        per_group: dict[str, pd.DataFrame] | None = None
        pdc = path / "per_group_confusion"
        if pdc.is_dir():
            per_group = {}
            for f in sorted(pdc.glob("*.csv")):
                d = pd.read_csv(f, index_col="true_state")
                d.index = d.index.astype(str)
                per_group[f.stem] = d
        return cls(
            method=meta["method"],
            group_key=meta.get("group_key", "group"),
            state_metrics=state_metrics,
            group_metrics=group_metrics,
            group_summary=group_summary,
            confusion_matrix=cm,
            cell_predictions=cell_predictions,
            per_group_confusion=per_group,
            overall=overall,
            config=LODOConfig.from_dict(meta["config"]),
            backend=meta["backend"],
            latent=meta["latent"],
            created=meta["created"],
            statetrace_version=meta["statetrace_version"],
            distance_calibration=calibration,
            holdout_evidence=evidence,
        )


# --------------------------------------------------------------------------- folds
def logo_folds(group_labels: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """One (fold name, test mask) per group label, in sorted order."""
    g = np.asarray(group_labels).astype(str)
    return [(str(u), g == u) for u in np.unique(g)]


def random_split_folds(
    state_codes: np.ndarray, *, n_splits: int, random_state: int
) -> list[tuple[str, np.ndarray]]:
    """Stratified K-fold over cells by state, ignoring group membership."""
    from sklearn.model_selection import StratifiedKFold

    n_splits = int(n_splits)
    min_class = int(np.bincount(state_codes).min())
    if n_splits > min_class:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest state size ({min_class}); lower n_splits."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for i, (_, test_idx) in enumerate(skf.split(np.zeros(len(state_codes)), state_codes)):
        mask = np.zeros(len(state_codes), dtype=bool)
        mask[test_idx] = True
        folds.append((f"split{i}", mask))
    return folds


# ------------------------------------------------------------------------ engine
def _cell_table_from_proba(
    P: np.ndarray,
    votes: np.ndarray | None,
    *,
    states: np.ndarray,
    groups: np.ndarray,
    true_states: np.ndarray,
    fold_name: str,
    abstain_below: float | None,
) -> pd.DataFrame:
    order = np.argsort(-P, axis=1, kind="stable")
    best = order[:, 0]
    second = order[:, 1] if P.shape[1] > 1 else best
    rows = np.arange(len(best))
    p1 = P[rows, best]
    p2 = P[rows, second] if P.shape[1] > 1 else np.zeros_like(p1)
    predicted = states[best].astype(object)
    abstained = np.zeros(len(best), dtype=bool)
    if abstain_below is not None:
        abstained = p1 < float(abstain_below)
        predicted[abstained] = schema.UNKNOWN_LABEL
    support = votes[rows, best] / votes.sum(axis=1).clip(min=1) if votes is not None else np.nan
    return pd.DataFrame(
        {
            "group": groups,
            "true_state": true_states,
            "predicted_state": predicted,
            "probability": p1,
            "second_best": states[second],
            "margin": p1 - p2,
            "support": support,
            "abstained": abstained,
            "fold": fold_name,
        }
    )


def _run_fold_array(backend: MappingBackend, X, y, groups, test, states, fold_name, abstain_below):
    train = ~test
    b = backend.clone().fit(X[train], y[train])
    P_local = b.predict_proba(X[test])
    P = np.zeros((int(test.sum()), len(states)), dtype=np.float64)
    col = np.searchsorted(states, b.classes_)
    P[:, col] = P_local
    votes = None
    if hasattr(b, "neighbor_votes"):
        V = np.zeros_like(P, dtype=np.int32)
        V[:, col] = b.neighbor_votes(X[test])
        votes = V
    return _cell_table_from_proba(
        P,
        votes,
        states=states,
        groups=groups[test],
        true_states=y[test],
        fold_name=fold_name,
        abstain_below=abstain_below,
    )


def _run_fold_anndata(
    backend: AnnDataBackend, adata, state_key, y, groups, test, states, fold_name, abstain_below
):
    train = ~test
    b = backend.clone().fit(adata[train].copy(), state_key=state_key)
    preds = b.predict(adata[test].copy())
    P = preds.aligned_proba(states)
    votes = None
    if preds.support is not None:
        # translate support into a pseudo-vote matrix on the predicted column so downstream columns agree
        votes = np.zeros_like(P)
        pos = {s: i for i, s in enumerate(states)}
        for i, (lab, sup) in enumerate(
            zip(preds.labels.to_numpy(), preds.support.fillna(0).to_numpy(), strict=True)
        ):
            if lab in pos:
                votes[i, pos[lab]] = sup
                votes[i, :] = np.where(votes[i, :] > 0, votes[i, :], 0)
        votes = np.where(
            votes.sum(axis=1, keepdims=True) > 0,
            votes / votes.sum(axis=1, keepdims=True).clip(min=1e-9),
            votes,
        )
        votes = votes * 1.0
    table = _cell_table_from_proba(
        P,
        None,
        states=states,
        groups=groups[test],
        true_states=y[test],
        fold_name=fold_name,
        abstain_below=abstain_below,
    )
    if preds.support is not None:
        table["support"] = preds.support.to_numpy()
    # when the tool gives labels but no probabilities, honour its labels directly
    if preds.proba is None:
        lab = preds.labels.to_numpy().astype(object)
        known = np.isin(lab, states)
        table.loc[~known, "predicted_state"] = schema.UNKNOWN_LABEL
        table.loc[~known, "abstained"] = True
        table["probability"] = preds.confidence.to_numpy()
    return table


def summarize_holdout(
    cell_predictions: pd.DataFrame,
    *,
    states: list[str] | np.ndarray,
    group_key: str = "group",
    method: str = "leave_one_group_out",
    config: LODOConfig | None = None,
    backend: str = "external",
    metadata_state_table: pd.DataFrame | None = None,
    latent: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Build a :class:`ValidationResult` from held-out predictions produced elsewhere.

    Use this when an external tool (scPoli, scANVI, Symphony, ...) was run in a
    leave-one-group-out loop outside StateTrace. ``cell_predictions`` needs at least the
    columns ``group``, ``true_state``, ``predicted_state``; ``probability``, ``second_best``,
    ``margin``, ``support``, ``abstained`` and ``fold`` are filled with defaults when absent.
    """
    from . import __version__

    config = config or LODOConfig()
    cp = cell_predictions.copy()
    for col, default in (
        ("probability", np.nan),
        ("second_best", pd.NA),
        ("margin", np.nan),
        ("support", np.nan),
        ("abstained", False),
    ):
        if col not in cp:
            cp[col] = default
    if "fold" not in cp:
        cp["fold"] = cp["group"].astype(str)
    cp["abstained"] = cp["abstained"].astype(bool) | (
        cp["predicted_state"].astype(str) == schema.UNKNOWN_LABEL
    )
    cp = cp[list(metrics.CELL_COLUMNS)]
    states = np.asarray([str(s) for s in states])
    group_metrics = metrics.group_state_table(cp, min_group_cells=config.min_group_cells)
    gsum = metrics.group_summary(cp)
    state_metrics = metrics.state_table(
        group_metrics,
        cp,
        states=states,
        thresholds=config.thresholds,
        metadata_state_table=metadata_state_table,
    )
    cm = metrics.confusion_matrix(cp, states=states)
    per_group = None
    if config.per_group_confusion:
        per_group = {
            g: metrics.confusion_matrix(cp[cp["group"].astype(str) == g], states=states)
            for g in np.unique(cp["group"].astype(str))
        }
    overall = metrics.overall_metrics(cp, gsum)
    return ValidationResult(
        method=method,
        group_key=group_key,
        state_metrics=state_metrics,
        group_metrics=group_metrics,
        group_summary=gsum,
        confusion_matrix=cm,
        cell_predictions=cp,
        per_group_confusion=per_group,
        overall=overall,
        config=config,
        backend={"name": backend, "params": {}} if isinstance(backend, str) else backend,
        latent={
            **(latent or {"source": "external"}),
            "refit_latent": bool((latent or {}).get("refit_latent", False)),
        },
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        statetrace_version=__version__,
    )


def run_validation(
    *,
    method: str,
    group_key: str,
    X: np.ndarray | None,
    state_labels: np.ndarray,
    group_labels: np.ndarray,
    obs_names: np.ndarray,
    states: np.ndarray,
    backend: MappingBackend | AnnDataBackend,
    config: LODOConfig,
    latent_provenance: dict[str, Any],
    metadata_state_table: pd.DataFrame | None = None,
    n_splits: int = 5,
    adata: AnnData | None = None,
    state_key: str | None = None,
) -> ValidationResult:
    """
    Run grouped or random-split validation with an array backend (``X`` required) or an
    AnnData backend (``adata`` and ``state_key`` required), and assemble the result.
    """
    if method not in schema.VALIDATION_METHODS:
        raise ValueError(f"method must be one of {schema.VALIDATION_METHODS}, got {method!r}")
    if config.refit_latent:
        raise NotImplementedError(
            "refit_latent=True is reserved for backends that refit the latent per fold; "
            "the built-in kNN treats the latent as fixed."
        )
    states = np.asarray([str(s) for s in states])
    y = np.asarray([str(s) for s in state_labels], dtype=object)
    groups = np.asarray([str(g) for g in group_labels], dtype=object)
    state_codes = np.searchsorted(states, y)
    if method == "random_split":
        folds = random_split_folds(state_codes, n_splits=n_splits, random_state=config.random_state)
    else:
        folds = logo_folds(groups)
    logger.info(
        "running %s over %r with %d folds on %d cells", method, group_key, len(folds), len(y)
    )

    is_array = isinstance(backend, MappingBackend) and not isinstance(backend, AnnDataBackend)
    if is_array:
        if X is None:
            raise ValueError("an array backend needs the latent matrix X")
        X = np.asarray(X, dtype=np.float32)
        if config.n_jobs and config.n_jobs != 1:
            from joblib import Parallel, delayed

            parts = Parallel(n_jobs=config.n_jobs, prefer="threads")(
                delayed(_run_fold_array)(
                    backend, X, y, groups, m, states, name, config.abstain_below
                )
                for name, m in folds
            )
        else:
            parts = [
                _run_fold_array(backend, X, y, groups, m, states, name, config.abstain_below)
                for name, m in folds
            ]
    else:
        if adata is None or state_key is None:
            raise ValueError("an AnnData backend needs adata and state_key")
        parts = [
            _run_fold_anndata(
                backend, adata, state_key, y, groups, m, states, name, config.abstain_below
            )
            for name, m in folds
        ]
    idx = np.concatenate([np.asarray(obs_names)[m] for _, m in folds])
    cp = pd.concat(parts, ignore_index=True)
    cp.index = pd.Index(idx.astype(str), name="cell")
    cp = cp.reindex(np.asarray(obs_names).astype(str))
    calibration = None
    evidence = None
    if X is not None and method != "random_split":
        predicted = cp["predicted_state"].to_numpy().astype(str)
        assigned = pd.Index(states).get_indexer(predicted)  # -1 for unknown / abstained
        evidence = novelty.holdout_evidence(
            X,
            state_codes,
            groups,
            folds,
            states,
            index_template=backend if is_array else None,
            min_cells=config.min_group_cells,
            assigned_codes=assigned,
        )
        evidence.index = pd.Index(idx.astype(str), name="cell")
        evidence = evidence.reindex(np.asarray(obs_names).astype(str))
        _, geometry = novelty.state_geometry(X, state_codes, len(states))
        calibration = novelty.holdout_distance_calibration(evidence, geometry, states)
    result = summarize_holdout(
        cp,
        states=states,
        group_key=group_key,
        method=method,
        config=config,
        backend={
            "name": getattr(backend, "name", type(backend).__name__),
            "params": backend.get_params(),
        },
        metadata_state_table=metadata_state_table,
        latent={**latent_provenance, "refit_latent": False},
    )
    result.distance_calibration = calibration
    result.holdout_evidence = evidence
    return result
