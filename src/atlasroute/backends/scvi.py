"""
scVI/scANVI backend (reserved for v0.2).

The class exists so the registry, documentation and error messages are stable; it
raises at construction time until the backend is implemented.
"""

from __future__ import annotations

import importlib.util
from typing import Any, ClassVar

INSTALL_HINT = "pip install 'atlasroute[scvi]'"


class ScviBackend:
    """Placeholder for the scvi-tools backend (planned for v0.2)."""

    name: ClassVar[str] = "scvi"

    def __init__(self, **params: Any) -> None:
        if importlib.util.find_spec("scvi") is None:
            raise ImportError(
                "The scvi backend requires scvi-tools. Install it with: "
                f"{INSTALL_HINT} (the backend itself is planned for AtlasRoute v0.2)."
            )
        raise NotImplementedError("The scvi backend is planned for AtlasRoute v0.2.")
