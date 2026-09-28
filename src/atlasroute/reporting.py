"""
Standardised transport report (planned for v0.2).

``TransportReport`` will write a results directory with reference QC, query QC and
per-state transport QC (reference donors, query donors, LODO recall, median confidence,
unknown fraction, reference-distance shift, transport status).
"""

from __future__ import annotations

from typing import Any


class TransportReport:
    """Placeholder for the v0.2 report object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("atlasroute.reporting.TransportReport is planned for v0.2.")
