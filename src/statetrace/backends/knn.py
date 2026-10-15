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
    _classes: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
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
        """
        Fit the neighbour index on ``X`` (cells x dimensions; rows L2-normalised for
        ``metric='cosine'``) and labels ``y``; ``n_neighbors`` is capped at the number of cells.

        Returns
        -------
        KNNBackend
            ``self``.

        Raises
        ------
        ValueError
            If ``y`` is empty.
        """
        from sklearn.neighbors import KNeighborsClassifier

        y = np.asarray(y)
        if len(y) == 0:
            raise ValueError("cannot fit on an empty training set")
        k = int(min(self.n_neighbors, len(y)))
        # fit on integer codes: one pass over the labels (they are often strings) instead of
        # scikit-learn's own unique plus a lookup; classes_ keeps the sorted labels
        self._classes, self._y_codes = np.unique(y, return_inverse=True)
        self._model = KNeighborsClassifier(
            n_neighbors=k, weights=self.weights, metric="euclidean", n_jobs=self.n_jobs
        ).fit(self._prep(X), self._y_codes)
        return self

    @property
    def classes_(self) -> np.ndarray:
        """The sorted state labels, in column order of :meth:`predict_proba`."""
        self._check_fitted()
        return self._classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Neighbour-vote probabilities (cells x states), columns in the order of :attr:`classes_`.
        """
        self._check_fitted()
        return np.asarray(self._model.predict_proba(self._prep(X)), dtype=np.float64)

    def predict_proba_votes(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :meth:`predict_proba` and :meth:`neighbor_votes` from one neighbour search. With
        uniform weights the probabilities are the vote fractions; distance weights fall back
        to scikit-learn's weighting.
        """
        self._check_fitted()
        if self.weights != "uniform":
            return self.predict_proba(X), self.neighbor_votes(X)
        _, idx = self.kneighbors(X)
        votes = self._votes(idx)
        return votes / votes.sum(axis=1, keepdims=True).astype(np.float64), votes

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
        return self._votes(self.kneighbors(X)[1])

    def _votes(self, idx: np.ndarray) -> np.ndarray:
        n, c = idx.shape[0], len(self.classes_)
        flat = (np.arange(n)[:, None] * c + self._y_codes[idx]).ravel()
        return np.bincount(flat, minlength=n * c).reshape(n, c).astype(np.int32)

    def get_params(self) -> dict[str, Any]:
        """``n_neighbors``, ``metric``, ``weights`` and ``n_jobs``."""
        return {
            "n_neighbors": int(self.n_neighbors),
            "metric": self.metric,
            "weights": self.weights,
            "n_jobs": self.n_jobs,
        }

    def clone(self) -> KNNBackend:
        """An unfitted copy with the same parameters."""
        return KNNBackend(**self.get_params())
