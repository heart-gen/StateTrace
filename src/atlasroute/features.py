"""
Feature harmonisation between a reference and a query (planned for v0.2).

Same-species transport intersects ``var_names``; cross-species transport (for example
human -> macaque or human -> mouse) maps query genes through a one-to-one ortholog
table and reports ambiguous and unmapped genes.
"""

from __future__ import annotations

from typing import Any


def harmonize_features(
    reference_var: Any, query_var: Any, *, ortholog_table: Any = None, **kwargs: Any
) -> Any:
    """Align query features to the reference feature set (v0.2)."""
    raise NotImplementedError("atlasroute.features.harmonize_features is planned for v0.2.")
