"""
Mapping backends: interchangeable engines that assign reference states to cells.

StateTrace does not implement an integration method. The built-in kNN backend works on any
fixed latent space; external tools (scPoli, scANVI/scArches, Symphony, Azimuth, CellTypist)
take part either through :class:`~statetrace.predictions.Predictions` (their finished output)
or through the :class:`~statetrace.backends.base.AnnDataBackend` protocol (refit per fold).
"""

from __future__ import annotations

from typing import Any

from .base import AnnDataBackend, CallableBackend, MappingBackend, SupportsNeighborSupport

__all__ = [
    "BACKENDS",
    "AnnDataBackend",
    "CallableBackend",
    "MappingBackend",
    "SupportsNeighborSupport",
    "get_backend",
]

# name -> (module, class); resolved lazily so optional dependencies are not imported eagerly
BACKENDS: dict[str, tuple[str, str]] = {
    "knn": ("statetrace.backends.knn", "KNNBackend"),
    "scanvi": ("statetrace.backends.scvi", "ScanviBackend"),
}


def get_backend(name: str, **params: Any) -> Any:
    """Instantiate a registered backend by name (``"knn"``; ``"scanvi"`` needs the ``scvi`` extra)."""
    try:
        mod_name, cls_name = BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}; known backends: {sorted(BACKENDS)}") from None
    import importlib

    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(**params)
