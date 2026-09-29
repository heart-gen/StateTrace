"""
Protocols for mapping backends.

Two flavours exist. :class:`MappingBackend` works on arrays in a fixed latent space (the
built-in kNN). :class:`AnnDataBackend` wraps tools that need the full AnnData (scANVI/scArches,
scPoli, CellTypist, ...) and returns :class:`~statetrace.predictions.Predictions`; it is the
hook through which external mappers take part in grouped validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

    from ..predictions import Predictions


@runtime_checkable
class MappingBackend(Protocol):
    """
    A classifier over a fixed latent space.

    ``predict_proba`` returns one row per cell with columns aligned to ``classes_`` summing to
    one. ``get_params`` must be JSON-serialisable; ``clone`` returns an unfitted copy.
    """

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> MappingBackend: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def classes_(self) -> np.ndarray: ...

    def get_params(self) -> dict[str, Any]: ...

    def clone(self) -> MappingBackend: ...


@runtime_checkable
class SupportsNeighborSupport(Protocol):
    """Backends that can report how many reference neighbours voted for each class."""

    def neighbor_votes(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AnnDataBackend(Protocol):
    """
    A mapper that trains on an AnnData with state labels and predicts another AnnData.

    ``fit`` receives the training cells and the ``obs`` key holding the state labels;
    ``predict`` returns :class:`~statetrace.predictions.Predictions` for the query cells.
    """

    name: str

    def fit(self, adata: AnnData, *, state_key: str) -> AnnDataBackend: ...

    def predict(self, adata: AnnData) -> Predictions: ...

    def get_params(self) -> dict[str, Any]: ...

    def clone(self) -> AnnDataBackend: ...


class CallableBackend:
    """
    Wrap a function ``fn(train_adata, test_adata, state_key) -> Predictions`` as an
    :class:`AnnDataBackend`, the quickest way to run a custom mapper inside grouped validation.
    """

    def __init__(self, fn: Any, *, name: str = "callable", **params: Any) -> None:
        self.fn = fn
        self.name = name
        self.params = params
        self._train: AnnData | None = None
        self._state_key: str | None = None

    def fit(self, adata: AnnData, *, state_key: str) -> CallableBackend:
        self._train = adata
        self._state_key = state_key
        return self

    def predict(self, adata: AnnData) -> Predictions:
        if self._train is None:
            raise RuntimeError("call fit() first")
        p = self.fn(self._train, adata, self._state_key, **self.params)
        if p.backend == "custom":
            p.backend = self.name
        return p

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}

    def clone(self) -> CallableBackend:
        return CallableBackend(self.fn, name=self.name, **self.params)
