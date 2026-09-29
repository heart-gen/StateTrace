"""
StateTrace: a backend-independent framework for donor- and cohort-aware validation of
cellular-state transportability and downstream inference.

StateTrace does not map cells. It takes the output of any mapping tool (scPoli, scANVI/scArches,
Symphony, Azimuth, CellTypist, a user classifier, or its own kNN baseline) and determines
whether each cellular state defined in one cohort is reproducible across the reference's
donors, recovers in an independent cohort, is robust to the choice of backend, is not
shifted or out of reference, and therefore is **supported** for group-level inference.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from typing import TYPE_CHECKING

try:
    __version__ = _v("statetrace")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    # reference construction and grouped validation
    "StateReference",
    "Reference",
    "ValidationResult",
    "LODOConfig",
    "StabilityThresholds",
    "summarize_holdout",
    # predictions from any backend
    "Predictions",
    "MappingBackend",
    "AnnDataBackend",
    "CallableBackend",
    "KNNBackend",
    "get_backend",
    # cross-cohort transport and inference gating
    "validate_transport",
    "TransportRoute",
    "TransportConfig",
    "compare_backends",
    "BackendComparison",
    # outputs
    "state_abundance",
    "pseudobulk",
    "harmonize_features",
    "FeatureMap",
    "calibration_table",
    # utilities and data
    "MetadataReport",
    "validate_metadata",
    "resolve_latent",
    "simulate_reference",
    "simulate_transport",
    # io
    "FrozenReferenceError",
    "read_frozen",
    "write_frozen",
]

_lazy = {
    "StateReference": (".reference", "StateReference"),
    "Reference": (".reference", "Reference"),
    "ValidationResult": (".validation", "ValidationResult"),
    "LODOConfig": (".validation", "LODOConfig"),
    "StabilityThresholds": (".validation", "StabilityThresholds"),
    "summarize_holdout": (".validation", "summarize_holdout"),
    "Predictions": (".predictions", "Predictions"),
    "MappingBackend": (".backends.base", "MappingBackend"),
    "AnnDataBackend": (".backends.base", "AnnDataBackend"),
    "CallableBackend": (".backends.base", "CallableBackend"),
    "KNNBackend": (".backends.knn", "KNNBackend"),
    "get_backend": (".backends", "get_backend"),
    "validate_transport": (".transport", "validate_transport"),
    "TransportRoute": (".transport", "TransportRoute"),
    "TransportConfig": (".transport", "TransportConfig"),
    "compare_backends": (".robustness", "compare_backends"),
    "BackendComparison": (".robustness", "BackendComparison"),
    "state_abundance": (".aggregation", "state_abundance"),
    "pseudobulk": (".aggregation", "pseudobulk"),
    "harmonize_features": (".features", "harmonize_features"),
    "FeatureMap": (".features", "FeatureMap"),
    "calibration_table": (".calibration", "calibration_table"),
    "MetadataReport": (".utils", "MetadataReport"),
    "validate_metadata": (".utils", "validate_metadata"),
    "resolve_latent": (".utils", "resolve_latent"),
    "simulate_reference": (".datasets.synthetic", "simulate_reference"),
    "simulate_transport": (".datasets.synthetic", "simulate_transport"),
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
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:
    from .aggregation import pseudobulk, state_abundance
    from .backends import get_backend
    from .backends.base import AnnDataBackend, CallableBackend, MappingBackend
    from .backends.knn import KNNBackend
    from .calibration import calibration_table
    from .datasets.synthetic import simulate_reference, simulate_transport
    from .features import FeatureMap, harmonize_features
    from .io import FrozenReferenceError, read_frozen, write_frozen
    from .predictions import Predictions
    from .reference import Reference, StateReference
    from .robustness import BackendComparison, compare_backends
    from .transport import TransportConfig, TransportRoute, validate_transport
    from .utils import MetadataReport, resolve_latent, validate_metadata
    from .validation import LODOConfig, StabilityThresholds, ValidationResult, summarize_holdout
