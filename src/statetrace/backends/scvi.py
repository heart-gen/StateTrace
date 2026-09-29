"""
scANVI backend (``AnnDataBackend``) through scvi-tools; requires the ``scvi`` extra.

The class is instantiable only when scvi-tools is importable; the implementation lands with
the benchmark's gate for external backends.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar

INSTALL_HINT = "pip install 'statetrace[scvi]'"


class ScanviBackend:
    """scANVI / scArches refit-per-fold backend (planned)."""

    name: ClassVar[str] = "scanvi"

    def __init__(self, **params: Any) -> None:
        if importlib.util.find_spec("scvi") is None:
            raise ImportError(
                f"The scanvi backend requires scvi-tools. Install it with: {INSTALL_HINT}."
            )
        raise NotImplementedError("The scanvi backend is planned for a later release.")
