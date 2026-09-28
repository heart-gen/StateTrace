"""
Confidence calibration (planned for v0.2).

Reference holdouts establish the empirical relationship between classifier confidence
and the probability of a correct transport. The planned ``Calibrator`` will support
isotonic regression, Platt scaling and empirical quantile thresholds.
"""

from __future__ import annotations

from typing import Any


class Calibrator:
    """Placeholder for the v0.2 calibrator."""

    def __init__(self, *, method: str = "isotonic", **kwargs: Any) -> None:
        raise NotImplementedError("atlasroute.calibration.Calibrator is planned for v0.2.")
