"""
Pure functions that turn a per-cell prediction table into donor- and state-level
metrics. They take and return pandas objects only, so the benchmark repository can
apply them to transport results as well as to validation folds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import schema

if TYPE_CHECKING:
    from .validation import StabilityThresholds

CELL_COLUMNS = (
    "donor",
    "true_state",
    "predicted_state",
    "probability",
    "second_best",
    "margin",
    "knn_support",
    "abstained",
    "fold",
)


def _f1(precision: pd.Series | float, recall: pd.Series | float):
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, 2 * precision * recall / denom, 0.0)


def donor_state_table(cell_predictions: pd.DataFrame, *, min_donor_cells: int = 20) -> pd.DataFrame:
    """
    One row per (donor, true state) with recall, precision and F1 inside that donor.

    Abstained cells count as misses for recall and never enter a precision denominator
    (their predicted label is :data:`~atlasroute.schema.UNKNOWN_LABEL`).

    Parameters
    ----------
    cell_predictions
        Table with at least ``donor``, ``true_state``, ``predicted_state``.
    min_donor_cells
        A (donor, state) pair with fewer cells is reported but ``scored=False`` and is
        excluded from state-level aggregates.
    """
    df = cell_predictions
    correct = (df["true_state"].astype(str) == df["predicted_state"].astype(str)).to_numpy()
    rec = (
        pd.DataFrame(
            {
                "donor": df["donor"].astype(str).to_numpy(),
                "state": df["true_state"].astype(str).to_numpy(),
                "correct": correct,
            }
        )
        .groupby(["donor", "state"], sort=True)
        .agg(n_cells=("correct", "size"), tp=("correct", "sum"))
    )
    pred = (
        pd.DataFrame(
            {
                "donor": df["donor"].astype(str).to_numpy(),
                "state": df["predicted_state"].astype(str).to_numpy(),
            }
        )
        .groupby(["donor", "state"], sort=True)
        .size()
        .rename("n_predicted")
    )
    out = rec.join(pred, how="left").fillna({"n_predicted": 0})
    out["n_predicted"] = out["n_predicted"].astype(int)
    out["n_cells"] = out["n_cells"].astype(int)
    out["tp"] = out["tp"].astype(int)
    out["recall"] = out["tp"] / out["n_cells"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["precision"] = np.where(out["n_predicted"] > 0, out["tp"] / out["n_predicted"], np.nan)
    out["f1"] = np.where(
        out["precision"].isna(), np.nan, _f1(out["precision"].fillna(0), out["recall"])
    )
    out["scored"] = out["n_cells"] >= int(min_donor_cells)
    out = out.drop(columns="tp").reset_index()
    return out[["donor", "state", "n_cells", "n_predicted", "recall", "precision", "f1", "scored"]]


def donor_summary(cell_predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per donor: accuracy, balanced accuracy over the donor's states, abstention rate."""
    df = pd.DataFrame(
        {
            "donor": cell_predictions["donor"].astype(str).to_numpy(),
            "state": cell_predictions["true_state"].astype(str).to_numpy(),
            "correct": (
                cell_predictions["true_state"].astype(str)
                == cell_predictions["predicted_state"].astype(str)
            ).to_numpy(),
            "abstained": cell_predictions["abstained"].astype(bool).to_numpy()
            if "abstained" in cell_predictions
            else False,
        }
    )
    per_state = df.groupby(["donor", "state"], sort=True)["correct"].mean()
    out = df.groupby("donor", sort=True).agg(
        n_cells=("correct", "size"),
        n_states=("state", "nunique"),
        accuracy=("correct", "mean"),
        abstention_rate=("abstained", "mean"),
    )
    out["balanced_accuracy"] = per_state.groupby(level="donor").mean()
    out = out.reset_index()
    return out[["donor", "n_cells", "n_states", "accuracy", "balanced_accuracy", "abstention_rate"]]


def stability_category(
    row: pd.Series | dict,
    thresholds: StabilityThresholds,
) -> str:
    """
    Rule-based LODO stability label for one state.

    Evaluated in order: ``insufficient_donors`` (too few scored donors), ``fragile``
    (median donor recall or donor recovery rate below the fragile floor), ``stable``
    (median, minimum and recovery all above the stable floors), otherwise ``moderate``.
    The thresholds are descriptive defaults, not claims of biological equivalence.
    """
    n = row["n_donors_scored"]
    med = row["median_donor_recall"]
    mn = row["min_donor_recall"]
    rr = row["donor_recovery_rate"]
    if pd.isna(n) or n < thresholds.min_donors:
        return "insufficient_donors"
    if med < thresholds.fragile_median_recall or rr < thresholds.fragile_recovery_rate:
        return "fragile"
    if (
        med >= thresholds.stable_median_recall
        and mn >= thresholds.stable_min_recall
        and rr >= thresholds.stable_recovery_rate
    ):
        return "stable"
    return "moderate"


def state_table(
    donor_metrics: pd.DataFrame,
    cell_predictions: pd.DataFrame,
    *,
    states: list[str] | np.ndarray,
    thresholds: StabilityThresholds,
    metadata_state_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    State-level summary over donors (index = state).

    Donor aggregates use only ``scored`` rows of ``donor_metrics``; pooled recall,
    precision and F1 use every held-out cell. ``n_donors_reference`` and ``eligible``
    come from ``metadata_state_table`` when given (donors with >= 1 cell of the state).
    """
    states = [str(s) for s in states]
    scored = donor_metrics[donor_metrics["scored"]]
    g = scored.groupby("state")["recall"]
    agg = pd.DataFrame(
        {
            "n_donors_scored": g.size(),
            "median_donor_recall": g.median(),
            "min_donor_recall": g.min(),
            "recall_variance": g.var(ddof=1),
            "recall_iqr": g.quantile(0.75) - g.quantile(0.25),
            "donor_recovery_rate": g.apply(
                lambda r: float((r >= thresholds.recovery_recall).mean())
            ),
        }
    ).reindex(states)
    agg["n_donors_scored"] = agg["n_donors_scored"].fillna(0).astype(int)

    true = cell_predictions["true_state"].astype(str).to_numpy()
    pred = cell_predictions["predicted_state"].astype(str).to_numpy()
    tp = pd.Series({s: int(((true == s) & (pred == s)).sum()) for s in states})
    n_true = pd.Series({s: int((true == s).sum()) for s in states})
    n_pred = pd.Series({s: int((pred == s).sum()) for s in states})
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["pooled_recall"] = np.where(n_true > 0, tp / n_true, np.nan)
        agg["precision"] = np.where(n_pred > 0, tp / n_pred, np.nan)
    agg["f1"] = np.where(
        agg["precision"].isna() | agg["pooled_recall"].isna(),
        np.nan,
        _f1(agg["precision"].fillna(0), agg["pooled_recall"].fillna(0)),
    )
    agg["n_cells_reference"] = n_true
    if metadata_state_table is not None:
        meta = metadata_state_table.reindex(states)
        agg["n_donors_reference"] = meta["n_donors"].fillna(0).astype(int)
        agg["eligible"] = meta["eligible"].fillna(False).astype(bool)
    else:
        nd = donor_metrics.groupby("state")["donor"].nunique()
        agg["n_donors_reference"] = nd.reindex(states).fillna(0).astype(int)
        agg["eligible"] = True
    agg["lodo_stability"] = [stability_category(row, thresholds) for _, row in agg.iterrows()]
    agg.index.name = "state"
    cols = [
        "n_donors_reference",
        "n_donors_scored",
        "n_cells_reference",
        "median_donor_recall",
        "min_donor_recall",
        "recall_variance",
        "recall_iqr",
        "donor_recovery_rate",
        "pooled_recall",
        "precision",
        "f1",
        "lodo_stability",
        "eligible",
    ]
    return agg[cols]


def confusion_matrix(
    cell_predictions: pd.DataFrame, *, states: list[str] | np.ndarray
) -> pd.DataFrame:
    """True states (rows) x predicted states (columns); an ``unknown`` column is added when any cell abstained."""
    states = [str(s) for s in states]
    true = cell_predictions["true_state"].astype(str)
    pred = cell_predictions["predicted_state"].astype(str)
    cols = list(states)
    if (pred == schema.UNKNOWN_LABEL).any():
        cols.append(schema.UNKNOWN_LABEL)
    cm = pd.crosstab(true, pred).reindex(index=states, columns=cols, fill_value=0)
    cm.index.name = "true_state"
    cm.columns.name = "predicted_state"
    return cm.astype(int)


def overall_metrics(
    cell_predictions: pd.DataFrame, donor_summary_table: pd.DataFrame
) -> dict[str, float]:
    """Pooled accuracy, balanced accuracy, macro F1, abstention rate and sizes."""
    true = cell_predictions["true_state"].astype(str).to_numpy()
    pred = cell_predictions["predicted_state"].astype(str).to_numpy()
    states = np.unique(true)
    correct = true == pred
    recalls = np.array([correct[true == s].mean() for s in states])
    f1s = []
    for s in states:
        tp = float(((true == s) & (pred == s)).sum())
        n_pred = float((pred == s).sum())
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec = tp / float((true == s).sum())
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    abst = (
        float(cell_predictions["abstained"].astype(bool).mean())
        if "abstained" in cell_predictions
        else 0.0
    )
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(np.mean(f1s)),
        "abstention_rate": abst,
        "median_donor_balanced_accuracy": float(donor_summary_table["balanced_accuracy"].median()),
        "n_cells": int(len(true)),
        "n_donors": int(cell_predictions["donor"].nunique()),
        "n_states": int(len(states)),
        "n_folds": int(cell_predictions["fold"].nunique()) if "fold" in cell_predictions else 0,
    }
