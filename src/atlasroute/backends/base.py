"""Protocols every mapping backend must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MappingBackend(Protocol):
    """
    A classifier over a fixed latent space.

    ``predict_proba`` must return one row per cell with columns aligned to
    ``classes_`` and summing to one. ``get_params`` must be JSON-serialisable so the
    backend configuration can be stored in a frozen reference; ``clone`` returns an
    unfitted copy with the same parameters (used once per validation fold).
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
