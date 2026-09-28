"""
Donor-level biological outputs (planned for v0.2).

``state_abundance`` returns a donor x state matrix; ``pseudobulk`` returns
donor x state x gene sums ready for edgeR, DESeq2, limma/voom or dream.
"""

from __future__ import annotations

from typing import Any


def state_abundance(*args: Any, **kwargs: Any) -> Any:
    """Donor x state abundance table (v0.2)."""
    raise NotImplementedError("atlasroute.aggregation.state_abundance is planned for v0.2.")


def pseudobulk(*args: Any, **kwargs: Any) -> Any:
    """Donor x state pseudobulk expression (v0.2)."""
    raise NotImplementedError("atlasroute.aggregation.pseudobulk is planned for v0.2.")
