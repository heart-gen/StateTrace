"""
Mapping backends: interchangeable engines that assign reference states to cells.

AtlasRoute does not implement an integration method; it wraps established mapping
engines behind :class:`~atlasroute.backends.base.MappingBackend` so the validation,
calibration and reporting layers stay engine-agnostic.
"""

from __future__ import annotations

from typing import Any

from .base import MappingBackend, SupportsNeighborSupport

__all__ = ["BACKENDS", "MappingBackend", "SupportsNeighborSupport", "get_backend"]

# name -> (module, class); resolved lazily so optional dependencies are not imported eagerly
BACKENDS: dict[str, tuple[str, str]] = {
    "knn": ("atlasroute.backends.knn", "KNNBackend"),
    "scvi": ("atlasroute.backends.scvi", "ScviBackend"),
}


def get_backend(name: str, **params: Any) -> MappingBackend:
    """
    Instantiate a registered backend by name.

    Parameters
    ----------
    name
        One of ``BACKENDS`` (currently ``"knn"``; ``"scvi"`` is reserved).
    **params
        Passed to the backend constructor.
    """
    try:
        mod_name, cls_name = BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}; known backends: {sorted(BACKENDS)}") from None
    import importlib

    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(**params)
