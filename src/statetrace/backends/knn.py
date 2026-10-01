"""Nearest-neighbour classification in a frozen latent space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

_METRICS = ("cosine", "euclidean")


@dataclass
class KNNBackend:
    """
    k-nearest-neighbour state assignment on a fixed embedding.

    ``metric="cosine"`` is implemented by L2-normalising rows and running a Euclidean
    search: on unit vectors ``||a - b||^2 = 2 (1 - cos)``, so the neighbour ranking is
    identical while tree-based search stays available.

    Parameters
    ----------
    n_neighbors
        Neighbours per query cell (capped at the training-set size).
    metric
        ``"cosine"`` (default; matches cosine neighbour graphs on scVI latents) or ``"euclidean"``.
    weights
        ``"uniform"`` or ``"distance"``; passed to scikit-learn.
    n_jobs
        Threads for the neighbour search.
    """

    n_neighbors: int = 15
    metric: str = "cosine"
    weights: str = "uniform"
    n_jobs: int | None = None
    name: ClassVar[str] = "knn"

    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _y_codes: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

    # ------------------------------------------------------------------ helpers
    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return X

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("KNNBackend is not fitted; call fit(X, y) first.")

    # ------------------------------------------------------------- MappingBackend
    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNBackend:
        from sklearn.neighbors import KNeighborsClassifier

        y = np.asarray(y)
        if len(y) == 0:
            raise ValueError("cannot fit on an empty training set")
        k = int(min(self.n_neighbors, len(y)))
        self._model = KNeighborsClassifier(
            n_neighbors=k, weights=self.weights, metric="euclidean", n_jobs=self.n_jobs
        ).fit(self._prep(X), y)
        self._y_codes = np.searchsorted(self._model.classes_, y)
        return self

    @property
    def classes_(self) -> np.ndarray:
        self._check_fitted()
        return self._model.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._model.predict_proba(self._prep(X)), dtype=np.float64)

    def kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Distances and training-row indices of the k nearest training cells, both shaped
        (n, k). Distances are in the backend's metric space (Euclidean on unit vectors for
        ``metric="cosine"``).
        """
        self._check_fitted()
        dist, idx = self._model.kneighbors(self._prep(X), return_distance=True)
        return np.asarray(dist, dtype=np.float32), np.asarray(idx)

    def neighbor_codes(self, idx: np.ndarray) -> np.ndarray:
        """Class codes (positions in ``classes_``) of training rows ``idx``."""
        self._check_fitted()
        return self._y_codes[idx]

    def neighbor_votes(self, X: np.ndarray) -> np.ndarray:
        """Integer count of the k nearest training cells per class, shape (n, n_classes)."""
        self._check_fitted()
        _, idx = self.kneighbors(X)
        votes = np.zeros((idx.shape[0], len(self.classes_)), dtype=np.int32)
        rows = np.arange(idx.shape[0])
        for j in range(idx.shape[1]):
            np.add.at(votes, (rows, self._y_codes[idx[:, j]]), 1)
        return votes

    def get_params(self) -> dict[str, Any]:
        return {
            "n_neighbors": int(self.n_neighbors),
            "metric": self.metric,
            "weights": self.weights,
            "n_jobs": self.n_jobs,
        }

    def clone(self) -> KNNBackend:
        return KNNBackend(**self.get_params())
