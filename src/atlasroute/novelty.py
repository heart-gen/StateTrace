"""
Out-of-reference detection (planned for v0.2).

Planned quantities per query cell: distance to the assigned state's reference
distribution in latent space, local reference support, and an out-of-distribution
statistic relative to the donor-holdout empirical distribution. Cells are then labelled
``supported`` / ``uncertain`` / ``out_of_reference``.
"""

from __future__ import annotations

from typing import Any


def novelty_scores(*args: Any, **kwargs: Any) -> Any:
    """Compute novelty statistics for query cells (v0.2)."""
    raise NotImplementedError("atlasroute.novelty is planned for v0.2.")
