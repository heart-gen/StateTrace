"""
Query transport: map an independent dataset onto a frozen reference (planned for v0.2).

The v0.2 API will be::

    result = atlasroute.transport(reference_or_path, query_adata, donor_key="donor_id")
    result.obs[schema.OBS_STATE], result.obs[schema.OBS_CONFIDENCE], result.obs[schema.OBS_UNKNOWN]

and ``TransportResult`` will keep the predicted state, probability, second-best state,
margin, reference-support statistic, novelty statistic and donor identifier per cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MSG = "atlasroute.transport is planned for v0.2; v0.1 covers reference validation only."


@dataclass
class TransportResult:
    """Placeholder for the v0.2 transport result."""

    def __post_init__(self) -> None:
        raise NotImplementedError(_MSG)


def transport(reference: Any, query: Any, *, donor_key: str, **kwargs: Any) -> TransportResult:
    """Map ``query`` onto a frozen ``reference`` (v0.2)."""
    raise NotImplementedError(_MSG)
