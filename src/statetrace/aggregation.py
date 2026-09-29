"""
Group-level biological outputs: state abundance and pseudobulk expression.

Both accept a list of states (or a transport status) so only states that passed inference
gating enter a downstream differential-abundance or differential-expression model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def state_abundance(
    cells: pd.DataFrame,
    *,
    group_col: str = "group",
    state_col: str = "state",
    states: list[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Group x state table of cell counts (or fractions of each group's cells).

    ``states`` restricts the columns; fractions are still computed over all of a group's cells
    so restricting states does not renormalise the denominator.
    """
    tab = pd.crosstab(cells[group_col].astype(str), cells[state_col].astype(str))
    if normalize:
        tab = tab.div(tab.sum(axis=1), axis=0)
    if states is not None:
        tab = tab.reindex(columns=[str(s) for s in states], fill_value=0.0)
    tab.index.name = "group"
    tab.columns.name = "state"
    return tab


def pseudobulk(
    adata: AnnData,
    *,
    group_key: str,
    state_labels: pd.Series | str,
    states: list[str] | None = None,
    layer: str | None = None,
    min_cells: int = 10,
) -> AnnData:
    """
    Sum counts per (group, state) into a pseudobulk AnnData ready for edgeR / DESeq2 / limma.

    Parameters
    ----------
    adata
        Query cells x genes; ``X`` or ``layers[layer]`` should hold raw counts.
    group_key
        ``obs`` column with the replication unit (donor).
    state_labels
        A Series aligned to ``adata.obs_names`` or the name of an ``obs`` column.
    states
        Keep only these states (for example ``route.supported_states``).
    layer
        Layer to sum instead of ``X``.
    min_cells
        Drop (group, state) pairs with fewer cells.

    Returns
    -------
    AnnData
        One observation per (group, state) with ``obs`` columns ``group``, ``state``, ``n_cells``.
    """
    import anndata as ad
    from scipy import sparse

    labels = (
        adata.obs[state_labels]
        if isinstance(state_labels, str)
        else state_labels.reindex(adata.obs_names)
    )
    labels = labels.astype(str)
    groups = adata.obs[group_key].astype(str)
    key = groups + "||" + labels
    if states is not None:
        keep = labels.isin([str(s) for s in states]).to_numpy()
    else:
        keep = np.ones(adata.n_obs, dtype=bool)
    X = adata.layers[layer] if layer is not None else adata.X
    codes, uniques = pd.factorize(key[keep], sort=True)
    n_cells = np.bincount(codes, minlength=len(uniques))
    ind = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniques), int(keep.sum()))
    )
    Xk = X[keep]
    summed = ind @ Xk
    summed = np.asarray(summed.todense()) if sparse.issparse(summed) else np.asarray(summed)
    obs = pd.DataFrame(
        {
            "group": [u.split("||")[0] for u in uniques],
            "state": [u.split("||")[1] for u in uniques],
            "n_cells": n_cells,
        },
        index=pd.Index(uniques, name="pseudobulk"),
    )
    ok = obs["n_cells"] >= int(min_cells)
    pb = ad.AnnData(X=summed[ok.to_numpy()], obs=obs[ok].copy(), var=adata.var[[]].copy())
    pb.uns["statetrace_pseudobulk"] = {
        "group_key": group_key,
        "states": states,
        "layer": layer,
        "min_cells": min_cells,
    }
    return pb
