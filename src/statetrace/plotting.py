"""
Plotting helpers (planned); requires the ``plot`` extra. matplotlib is imported inside each
function so ``import statetrace`` stays light.
"""

from __future__ import annotations

import importlib.util
from typing import Any

INSTALL_HINT = "pip install 'statetrace[plot]'"


def _require_matplotlib() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            f"statetrace.plotting requires matplotlib. Install it with: {INSTALL_HINT}."
        )


def plot_state_reproducibility(*args: Any, **kwargs: Any) -> Any:
    """Group recall distributions per state (planned)."""
    _require_matplotlib()
    raise NotImplementedError("statetrace.plotting is planned for a later release.")
