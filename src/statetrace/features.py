"""
Feature harmonisation between a reference and a query.

Same-species transport intersects ``var_names``; cross-species transport maps query genes
through a one-to-one ortholog table (for example Ensembl BioMart human -> macaque) and reports
ambiguous and unmapped genes so feature loss is visible in the transport report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


@dataclass
class FeatureMap:
    """Alignment of query features to reference features."""

    mapping: pd.DataFrame  # columns: reference, query
    n_reference: int
    n_query: int
    unmapped_reference: list[str]
    unmapped_query: list[str]
    ambiguous: list[str]

    @property
    def n_retained(self) -> int:
        """Number of reference features mapped to a query feature."""
        return len(self.mapping)

    @property
    def retention(self) -> float:
        """Fraction of reference features retained (NaN for an empty reference)."""
        return self.n_retained / self.n_reference if self.n_reference else float("nan")

    def summary(self) -> str:
        """
        A one-line summary of retained, unmapped and ambiguous features.

        Returns
        -------
        str
        """
        return (
            f"{self.n_retained}/{self.n_reference} reference features retained ({self.retention:.1%}); "
            f"{len(self.unmapped_reference)} reference and {len(self.unmapped_query)} query features unmapped; "
            f"{len(self.ambiguous)} ambiguous (dropped)"
        )

    def apply(self, query: AnnData) -> AnnData:
        """Subset ``query`` to mapped features and rename them to the reference names."""
        q = query[:, self.mapping["query"].to_numpy()].copy()
        q.var["query_feature"] = q.var_names.to_numpy()
        q.var_names = pd.Index(self.mapping["reference"].to_numpy(), name=query.var_names.name)
        return q


def harmonize_features(
    reference_var: pd.Index | list[str] | np.ndarray,
    query_var: pd.Index | list[str] | np.ndarray,
    *,
    ortholog_table: pd.DataFrame | None = None,
    reference_col: str = "reference",
    query_col: str = "query",
) -> FeatureMap:
    """
    Align query features to the reference feature set.

    Parameters
    ----------
    reference_var, query_var
        Feature names of the reference and query.
    ortholog_table
        Optional two-column table mapping reference genes to query genes. Rows whose
        reference or query gene appears more than once (one-to-many) are dropped as
        ``ambiguous``. Without a table, features are intersected by name.
    """
    ref = pd.Index(np.asarray(reference_var).astype(str))
    qry = pd.Index(np.asarray(query_var).astype(str))
    if ref.has_duplicates or qry.has_duplicates:
        raise ValueError("feature names must be unique in both reference and query")
    if ortholog_table is None:
        common = ref.intersection(qry)
        mapping = pd.DataFrame({"reference": common, "query": common})
        ambiguous: list[str] = []
    else:
        t = ortholog_table[[reference_col, query_col]].dropna().astype(str).drop_duplicates()
        t = t[t[reference_col].isin(ref) & t[query_col].isin(qry)]
        dup_ref = t[reference_col].duplicated(keep=False)
        dup_qry = t[query_col].duplicated(keep=False)
        ambiguous = sorted(set(t.loc[dup_ref | dup_qry, reference_col]))
        t = t[~(dup_ref | dup_qry)]
        mapping = t.rename(columns={reference_col: "reference", query_col: "query"}).reset_index(
            drop=True
        )
    mapping = mapping.sort_values("reference").reset_index(drop=True)
    return FeatureMap(
        mapping=mapping,
        n_reference=len(ref),
        n_query=len(qry),
        unmapped_reference=sorted(set(ref) - set(mapping["reference"])),
        unmapped_query=sorted(set(qry) - set(mapping["query"])),
        ambiguous=ambiguous,
    )
