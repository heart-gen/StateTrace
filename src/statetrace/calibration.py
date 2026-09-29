"""
Group-aware calibration of mapping confidence.

Conformal and probabilistic calibration of *cell-level* confidence exists elsewhere
(scConform, conformal annotation). StateTrace asks a different question: does a backend's
confidence predict whether a state is recovered in an unseen *group*? :func:`calibration_table`
bins held-out confidence and reports both cell accuracy and group-level recovery per bin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .validation import ValidationResult


def calibration_table(
    result: ValidationResult, *, bins: int = 10, recovery_recall: float = 0.5
) -> pd.DataFrame:
    """
    Per confidence bin: number of cells, mean confidence, cell accuracy, and the fraction of
    (group, state) pairs whose median confidence falls in the bin that were recovered
    (recall >= ``recovery_recall``).
    """
    cp = result.cell_predictions.dropna(subset=["probability"])
    edges = np.linspace(0, 1, bins + 1)
    edges[-1] = 1.0 + 1e-9
    cell_bin = np.clip(np.digitize(cp["probability"].to_numpy(), edges) - 1, 0, bins - 1)
    correct = (cp["true_state"].astype(str) == cp["predicted_state"].astype(str)).to_numpy()
    cells = (
        pd.DataFrame({"bin": cell_bin, "conf": cp["probability"].to_numpy(), "correct": correct})
        .groupby("bin")
        .agg(
            n_cells=("correct", "size"),
            mean_confidence=("conf", "mean"),
            cell_accuracy=("correct", "mean"),
        )
    )
    gs = (
        cp.assign(correct=correct)
        .groupby(["group", "true_state"])
        .agg(median_confidence=("probability", "median"), recall=("correct", "mean"))
    )
    gbin = np.clip(np.digitize(gs["median_confidence"].to_numpy(), edges) - 1, 0, bins - 1)
    groups = (
        pd.DataFrame({"bin": gbin, "recovered": (gs["recall"] >= recovery_recall).to_numpy()})
        .groupby("bin")
        .agg(n_group_states=("recovered", "size"), group_recovery=("recovered", "mean"))
    )
    out = cells.join(groups, how="outer").reindex(range(bins))
    out.insert(0, "bin_low", edges[:-1])
    out.insert(1, "bin_high", np.minimum(edges[1:], 1.0))
    out.index.name = "bin"
    return out


def expected_calibration_error(table: pd.DataFrame, *, column: str = "cell_accuracy") -> float:
    """Weighted mean absolute difference between confidence and accuracy over bins (cells or group-states as weights)."""
    w = table["n_cells"] if column == "cell_accuracy" else table["n_group_states"]
    t = table.dropna(subset=[column, "mean_confidence"])
    w = w.reindex(t.index).fillna(0)
    if w.sum() == 0:
        return float("nan")
    return float((w * (t["mean_confidence"] - t[column]).abs()).sum() / w.sum())


class IsotonicCalibrator:
    """Map raw confidence to empirical held-out accuracy with isotonic regression."""

    def __init__(self) -> None:
        self._model = None

    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
        )
        return self

    @classmethod
    def from_result(cls, result: ValidationResult) -> IsotonicCalibrator:
        cp = result.cell_predictions.dropna(subset=["probability"])
        correct = (cp["true_state"].astype(str) == cp["predicted_state"].astype(str)).to_numpy()
        return cls().fit(cp["probability"].to_numpy(), correct)

    def transform(self, confidence: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.predict(np.asarray(confidence, dtype=float))
