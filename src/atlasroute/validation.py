"""
Donor-aware reference validation.

Leave-one-donor-out (LODO) holds out every cell of one donor, fits the mapping backend
on the remaining donors in the fixed latent space, and predicts the held-out cells.
Repeating over donors yields a per-cell prediction table from which donor x state,
state-level and confusion summaries are built with :mod:`atlasroute.metrics`.

``random_split`` (stratified K-fold over cells, ignoring donors) is the conventional
cell-level benchmark. Its accuracy is an optimistic ceiling; the gap to LODO is the
donor effect the reference must survive.

Known optimism in v0.1: the latent space (scVI, PCA, ...) was fitted with the held-out
donor present. LODO therefore measures classifier transfer within a fixed embedding, not
de-novo embedding transfer. ``LODOConfig.refit_latent`` is reserved for backends that can
refit per fold.
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

from . import metrics, schema

if TYPE_CHECKING:
    from anndata import AnnData

    from .backends.base import MappingBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Cut-offs for the rule-based ``lodo_stability`` category. Descriptive defaults; the
    benchmark repository predeclares the values used in the manuscript.
    """

    min_donors: int = 3
    recovery_recall: float = 0.5
    stable_median_recall: float = 0.70
    stable_min_recall: float = 0.50
    stable_recovery_rate: float = 0.80
    fragile_median_recall: float = 0.50
    fragile_recovery_rate: float = 0.50


@dataclass(frozen=True)
class LODOConfig:
    """
    Options for :meth:`AtlasReference.leave_one_donor_out` and :meth:`AtlasReference.random_split`.

    Parameters
    ----------
    min_donor_cells
        A (donor, state) pair needs this many cells to count toward state-level aggregates.
    abstain_below
        If set, a held-out cell whose top probability is below this value is predicted as
        ``unknown`` (abstention) instead of its best state.
    thresholds
        :class:`StabilityThresholds`.
    per_donor_confusion
        Also keep one confusion matrix per held-out donor.
    n_jobs
        Threads over folds (joblib, ``prefer="threads"``).
    refit_latent
        Reserved. ``True`` raises :class:`NotImplementedError` in v0.1.
    random_state
        Seed for the random-split fold assignment.
    """

    min_donor_cells: int = 20
    abstain_below: float | None = None
    thresholds: StabilityThresholds = field(default_factory=StabilityThresholds)
    per_donor_confusion: bool = False
    n_jobs: int = 1
    refit_latent: bool = False
    random_state: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LODOConfig:
        d = dict(d)
        thr = d.pop("thresholds", {})
        return cls(thresholds=StabilityThresholds(**thr), **d)


@dataclass
class ValidationResult:
    """
    Output of a donor-aware validation run.

    Attributes
    ----------
    method
        ``"leave_one_donor_out"`` or ``"random_split"``.
    state_metrics
        Index = state. Donor-level aggregates, pooled precision/recall/F1, ``lodo_stability``.
    donor_metrics
        Long table, one row per (donor, state) present in that donor.
    donor_summary
        One row per donor.
    confusion_matrix
        True states x predicted states (plus ``unknown`` when abstention happened).
    cell_predictions
        One row per reference cell (index = ``obs_names``).
    per_donor_confusion
        ``{donor: confusion}`` if requested.
    overall
        Pooled summary numbers.
    """

    method: str
    state_metrics: pd.DataFrame
    donor_metrics: pd.DataFrame
    donor_summary: pd.DataFrame
    confusion_matrix: pd.DataFrame
    cell_predictions: pd.DataFrame
    per_donor_confusion: dict[str, pd.DataFrame] | None
    overall: dict[str, float]
    config: LODOConfig
    backend: dict[str, Any]
    latent: dict[str, Any]
    created: str
    atlasroute_version: str

    # ----------------------------------------------------------------- display
    def summary(self) -> str:
        """Short human-readable summary."""
        o = self.overall
        counts = (
            self.state_metrics["lodo_stability"]
            .value_counts()
            .reindex(schema.STABILITY_CATEGORIES, fill_value=0)
        )
        lines = [
            f"AtlasRoute {self.method}: {o['n_cells']} cells, {o['n_donors']} donors, "
            f"{o['n_states']} states, {o['n_folds']} folds",
            f"  accuracy {o['accuracy']:.3f} | balanced accuracy {o['balanced_accuracy']:.3f} | "
            f"macro F1 {o['macro_f1']:.3f} | abstention {o['abstention_rate']:.3f}",
            f"  median donor balanced accuracy {o['median_donor_balanced_accuracy']:.3f}",
            "  states: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()),
            f"  backend {self.backend.get('name')} {self.backend.get('params')}",
        ]
        if not self.latent.get("refit_latent", False):
            lines.append(
                "  note: latent space was fixed across folds (fitted with every donor present); "
                "this is an optimistic bound on embedding transfer."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ValidationResult {self.method}: {self.overall['n_donors']} donors, {self.overall['n_states']} states>"

    # --------------------------------------------------------------- AnnData io
    def to_obs(self, adata: AnnData, *, prefix: str = schema.OBS_LODO_STATE) -> None:
        """Write held-out predictions back into ``adata.obs`` (aligned on ``obs_names``)."""
        cp = self.cell_predictions.reindex(adata.obs_names)
        adata.obs[prefix] = pd.Categorical(cp["predicted_state"].astype(str))
        adata.obs[f"{prefix}_probability"] = cp["probability"].to_numpy()
        adata.obs[f"{prefix}_margin"] = cp["margin"].to_numpy()
        adata.obs[f"{prefix}_abstained"] = cp["abstained"].astype(bool).to_numpy()
        adata.uns[schema.UNS_VALIDATION] = {
            "method": self.method,
            "overall": dict(self.overall),
            "created": self.created,
            "atlasroute_version": self.atlasroute_version,
        }

    # ------------------------------------------------------------- directory io
    def to_dir(self, path: str | Path, *, include_cell_predictions: bool = True) -> Path:
        """Write tables as CSV and configuration as JSON into ``path``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "method": self.method,
                    "config": self.config.to_dict(),
                    "backend": self.backend,
                    "latent": self.latent,
                    "created": self.created,
                    "atlasroute_version": self.atlasroute_version,
                },
                indent=2,
            )
        )
        (path / "overall.json").write_text(json.dumps(self.overall, indent=2))
        self.state_metrics.to_csv(path / "state_metrics.csv")
        self.donor_metrics.to_csv(path / "donor_metrics.csv", index=False)
        self.donor_summary.to_csv(path / "donor_summary.csv", index=False)
        self.confusion_matrix.to_csv(path / "confusion_matrix.csv")
        if include_cell_predictions:
            self.cell_predictions.to_csv(path / "cell_predictions.csv.gz", index_label="cell")
        if self.per_donor_confusion:
            pdc = path / "per_donor_confusion"
            pdc.mkdir(exist_ok=True)
            for donor, cm in self.per_donor_confusion.items():
                cm.to_csv(pdc / f"{donor}.csv")
        return path

    @classmethod
    def from_dir(cls, path: str | Path) -> ValidationResult:
        """Read a result written by :meth:`to_dir`."""
        path = Path(path)
        meta = json.loads((path / "config.json").read_text())
        overall = json.loads((path / "overall.json").read_text())
        state_metrics = pd.read_csv(path / "state_metrics.csv", index_col="state")
        state_metrics.index = state_metrics.index.astype(str)
        donor_metrics = pd.read_csv(path / "donor_metrics.csv", dtype={"donor": str, "state": str})
        donor_summary = pd.read_csv(path / "donor_summary.csv", dtype={"donor": str})
        cm = pd.read_csv(path / "confusion_matrix.csv", index_col="true_state")
        cm.index = cm.index.astype(str)
        cm.columns.name = "predicted_state"
        cp_file = path / "cell_predictions.csv.gz"
        if cp_file.exists():
            cell_predictions = pd.read_csv(
                cp_file,
                index_col="cell",
                dtype={"donor": str, "true_state": str, "predicted_state": str, "second_best": str},
            )
            cell_predictions.index = cell_predictions.index.astype(str)
        else:
            cell_predictions = pd.DataFrame(columns=list(metrics.CELL_COLUMNS))
        per_donor: dict[str, pd.DataFrame] | None = None
        pdc = path / "per_donor_confusion"
        if pdc.is_dir():
            per_donor = {}
            for f in sorted(pdc.glob("*.csv")):
                d = pd.read_csv(f, index_col="true_state")
                d.index = d.index.astype(str)
                per_donor[f.stem] = d
        return cls(
            method=meta["method"],
            state_metrics=state_metrics,
            donor_metrics=donor_metrics,
            donor_summary=donor_summary,
            confusion_matrix=cm,
            cell_predictions=cell_predictions,
            per_donor_confusion=per_donor,
            overall=overall,
            config=LODOConfig.from_dict(meta["config"]),
            backend=meta["backend"],
            latent=meta["latent"],
            created=meta["created"],
            atlasroute_version=meta["atlasroute_version"],
        )


# --------------------------------------------------------------------------- folds
def lodo_folds(donor_codes: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """One (fold name, test mask) per donor code, in sorted code order."""
    return [(str(d), donor_codes == d) for d in np.unique(donor_codes)]


def random_split_folds(
    state_codes: np.ndarray, *, n_splits: int, random_state: int
) -> list[tuple[str, np.ndarray]]:
    """Stratified K-fold over cells by state, ignoring donor membership."""
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


def _run_fold(
    backend: MappingBackend,
    X: np.ndarray,
    y: np.ndarray,
    donor_codes: np.ndarray,
    test: np.ndarray,
    states: np.ndarray,
    fold_name: str,
    abstain_below: float | None,
) -> pd.DataFrame:
    train = ~test
    b = backend.clone().fit(X[train], y[train])
    P_local = b.predict_proba(X[test])
    # re-align to the global state list; states absent from training get zero columns
    P = np.zeros((int(test.sum()), len(states)), dtype=np.float64)
    col = np.searchsorted(states, b.classes_)
    P[:, col] = P_local
    votes = None
    if hasattr(b, "neighbor_votes"):
        V = np.zeros_like(P, dtype=np.int32)
        V[:, col] = b.neighbor_votes(X[test])
        votes = V
    order = np.argsort(-P, axis=1, kind="stable")
    best = order[:, 0]
    second = order[:, 1] if P.shape[1] > 1 else best
    p1 = P[np.arange(len(best)), best]
    p2 = P[np.arange(len(best)), second] if P.shape[1] > 1 else np.zeros_like(p1)
    predicted = states[best].astype(object)
    abstained = np.zeros(len(best), dtype=bool)
    if abstain_below is not None:
        abstained = p1 < float(abstain_below)
        predicted[abstained] = schema.UNKNOWN_LABEL
    return pd.DataFrame(
        {
            "donor": donor_codes[test],
            "true_state": y[test],
            "predicted_state": predicted,
            "probability": p1,
            "second_best": states[second],
            "margin": p1 - p2,
            "knn_support": votes[np.arange(len(best)), best] if votes is not None else np.nan,
            "abstained": abstained,
            "fold": fold_name,
        }
    )


def run_validation(
    *,
    method: str,
    X: np.ndarray,
    state_labels: np.ndarray,
    donor_labels: np.ndarray,
    obs_names: np.ndarray,
    states: np.ndarray,
    backend: MappingBackend,
    config: LODOConfig,
    latent_provenance: dict[str, Any],
    metadata_state_table: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> ValidationResult:
    """
    Run LODO or random-split validation and assemble a :class:`ValidationResult`.

    This is the engine behind :meth:`AtlasReference.validate`; call it directly only to
    validate arrays that are not wrapped in an :class:`AtlasReference`.
    """
    if method not in schema.VALIDATION_METHODS:
        raise ValueError(f"method must be one of {schema.VALIDATION_METHODS}, got {method!r}")
    if config.refit_latent:
        raise NotImplementedError(
            "refit_latent=True is reserved for backends that can refit the latent per fold "
            "(planned scvi backend); v0.1 treats the latent as fixed."
        )
    from . import __version__

    states = np.asarray([str(s) for s in states])
    y = np.asarray([str(s) for s in state_labels], dtype=object)
    donors = np.asarray([str(d) for d in donor_labels], dtype=object)
    donor_codes = np.searchsorted(np.unique(donors), donors)
    state_codes = np.searchsorted(states, y)
    X = np.asarray(X, dtype=np.float32)

    if method == "leave_one_donor_out":
        folds = [(np.unique(donors)[int(k)], m) for k, m in lodo_folds(donor_codes)]
    else:
        folds = random_split_folds(state_codes, n_splits=n_splits, random_state=config.random_state)
    logger.info("running %s with %d folds on %d cells", method, len(folds), X.shape[0])

    if config.n_jobs and config.n_jobs != 1:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=config.n_jobs, prefer="threads")(
            delayed(_run_fold)(backend, X, y, donors, m, states, name, config.abstain_below)
            for name, m in folds
        )
    else:
        parts = [
            _run_fold(backend, X, y, donors, m, states, name, config.abstain_below)
            for name, m in folds
        ]
    idx = np.concatenate([np.asarray(obs_names)[m] for _, m in folds])
    cell_predictions = pd.concat(parts, ignore_index=True)
    cell_predictions.index = pd.Index(idx.astype(str), name="cell")
    cell_predictions = cell_predictions.reindex(np.asarray(obs_names).astype(str))

    donor_metrics = metrics.donor_state_table(
        cell_predictions, min_donor_cells=config.min_donor_cells
    )
    donor_sum = metrics.donor_summary(cell_predictions)
    state_metrics = metrics.state_table(
        donor_metrics,
        cell_predictions,
        states=states,
        thresholds=config.thresholds,
        metadata_state_table=metadata_state_table,
    )
    cm = metrics.confusion_matrix(cell_predictions, states=states)
    per_donor = None
    if config.per_donor_confusion:
        per_donor = {
            d: metrics.confusion_matrix(
                cell_predictions[cell_predictions["donor"] == d], states=states
            )
            for d in np.unique(donors)
        }
    overall = metrics.overall_metrics(cell_predictions, donor_sum)
    return ValidationResult(
        method=method,
        state_metrics=state_metrics,
        donor_metrics=donor_metrics,
        donor_summary=donor_sum,
        confusion_matrix=cm,
        cell_predictions=cell_predictions,
        per_donor_confusion=per_donor,
        overall=overall,
        config=config,
        backend={
            "name": getattr(backend, "name", type(backend).__name__),
            "params": backend.get_params(),
        },
        latent={**latent_provenance, "refit_latent": False},
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        atlasroute_version=__version__,
    )
