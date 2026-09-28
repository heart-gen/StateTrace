"""
AtlasRoute: donor-aware building, freezing, validation and transport of
cellular-state references across single-cell and single-nucleus datasets.

Donors, not cells, are the replication unit. Mapping engines are pluggable
backends; AtlasRoute is the validation, calibration, abstention and reporting
layer around them.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from typing import TYPE_CHECKING

try:
    __version__ = _v("atlasroute")  # distribution name
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

# Public API
__all__ = [
    # reference construction and validation
    "AtlasReference",
    "Reference",
    "ValidationResult",
    "LODOConfig",
    "StabilityThresholds",
    # backends
    "MappingBackend",
    "KNNBackend",
    "get_backend",
    # utilities and data
    "MetadataReport",
    "validate_metadata",
    "resolve_latent",
    "simulate_reference",
    "harmonize_features",
    # query transport (v0.2)
    "transport",
    "TransportResult",
    # io
    "FrozenReferenceError",
    "read_frozen",
    "write_frozen",
]

# Map public names for lazy loading: keeps `import atlasroute` light.
_lazy = {
    "AtlasReference": (".reference", "AtlasReference"),
    "Reference": (".reference", "Reference"),
    "ValidationResult": (".validation", "ValidationResult"),
    "LODOConfig": (".validation", "LODOConfig"),
    "StabilityThresholds": (".validation", "StabilityThresholds"),
    "MappingBackend": (".backends.base", "MappingBackend"),
    "KNNBackend": (".backends.knn", "KNNBackend"),
    "get_backend": (".backends", "get_backend"),
    "MetadataReport": (".utils", "MetadataReport"),
    "validate_metadata": (".utils", "validate_metadata"),
    "resolve_latent": (".utils", "resolve_latent"),
    "simulate_reference": (".datasets.synthetic", "simulate_reference"),
    "harmonize_features": (".features", "harmonize_features"),
    "transport": (".transport", "transport"),
    "TransportResult": (".transport", "TransportResult"),
    "FrozenReferenceError": (".io", "FrozenReferenceError"),
    "read_frozen": (".io", "read_frozen"),
    "write_frozen": (".io", "write_frozen"),
}


def __getattr__(name: str):
    """Lazy attribute loader to keep import-time light."""
    if name in _lazy:
        import importlib

        mod_name, attr_name = _lazy[name]
        mod = importlib.import_module(mod_name, __name__)
        obj = getattr(mod, attr_name)
        globals()[name] = obj  # cache for future access
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # help() and tab-complete show the public API
    return sorted(set(globals().keys()) | set(__all__))


# Make type checkers happy without importing heavy deps at runtime
if TYPE_CHECKING:
    from .backends import get_backend
    from .backends.base import MappingBackend
    from .backends.knn import KNNBackend
    from .datasets.synthetic import simulate_reference
    from .features import harmonize_features
    from .io import FrozenReferenceError, read_frozen, write_frozen
    from .reference import AtlasReference, Reference
    from .transport import TransportResult, transport
    from .utils import MetadataReport, resolve_latent, validate_metadata
    from .validation import LODOConfig, StabilityThresholds, ValidationResult
