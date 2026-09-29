"""Metadata validation and latent-space resolution shared by the reference and transport paths."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import schema

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataReport:
    """
    Result of :func:`validate_metadata`.

    Attributes
    ----------
    state_table
        One row per state (index) with ``n_cells``, ``n_groups``, ``n_studies`` (NaN when no
        study key), ``eligible`` and ``reason``.
    n_groups, n_cells, n_states, n_studies
        Reference sizes.
    warnings
        Warnings that were also emitted through :mod:`warnings`.
    """

    state_table: pd.DataFrame
    n_groups: int
    n_cells: int
    n_states: int
    n_studies: int | None
    warnings: tuple[str, ...]


def _as_categorical(adata: AnnData, key: str) -> pd.Series:
    col = adata.obs[key]
    if col.isna().any():
        raise ValueError(
            f"obs[{key!r}] contains missing values; drop or relabel those cells first."
        )
    if isinstance(col.dtype, pd.CategoricalDtype):
        col = col.cat.remove_unused_categories()
    else:
        col = col.astype(str).astype("category")
    adata.obs[key] = col
    return col


def validate_metadata(
    adata: AnnData,
    *,
    group_key: str,
    state_key: str,
    celltype_key: str | None = None,
    study_key: str | None = None,
    min_groups_per_state: int = 3,
    min_cells_per_state: int = 100,
) -> MetadataReport:
    """
    Check the group (donor) and state annotations of a reference and tabulate group support.

    Label columns are coerced to categorical in place. States with fewer than
    ``min_groups_per_state`` groups or ``min_cells_per_state`` cells are flagged
    ``eligible=False`` with a warning; they are **not** removed.

    Raises
    ------
    KeyError
        If a requested column is absent from ``obs``.
    ValueError
        If a label column has missing values or fewer than two groups are present.
    """
    keys = [k for k in (group_key, state_key, celltype_key, study_key) if k is not None]
    for key in keys:
        if key not in adata.obs.columns:
            available = ", ".join(map(str, list(adata.obs.columns)[:20]))
            raise KeyError(f"obs[{key!r}] not found. Available columns include: {available}")
    for key in keys:
        _as_categorical(adata, key)

    groups = adata.obs[group_key]
    states = adata.obs[state_key]
    if groups.nunique() < 2:
        raise ValueError("At least two groups (donors) are required for grouped validation.")

    tab = pd.crosstab(states, groups)
    n_cells = tab.sum(axis=1).astype(int)
    n_groups = (tab > 0).sum(axis=1).astype(int)
    few_groups = n_groups < min_groups_per_state
    few_cells = n_cells < min_cells_per_state
    reason = np.where(
        few_groups & few_cells,
        "few_groups,few_cells",
        np.where(few_groups, "few_groups", np.where(few_cells, "few_cells", "")),
    )
    state_table = pd.DataFrame(
        {
            "n_cells": n_cells,
            "n_groups": n_groups,
            "eligible": ~(few_groups | few_cells),
            "reason": reason,
        },
        index=pd.Index(tab.index.astype(str), name="state"),
    )
    n_studies: int | None = None
    if study_key is not None:
        st_tab = pd.crosstab(states, adata.obs[study_key])
        state_table.insert(
            2, "n_studies", (st_tab > 0).sum(axis=1).reindex(tab.index).astype(int).to_numpy()
        )
        n_studies = int(adata.obs[study_key].nunique())
    else:
        state_table.insert(2, "n_studies", np.nan)

    msgs: list[str] = []
    for state, row in state_table[~state_table["eligible"]].iterrows():
        msg = (
            f"state {state!r} has {row.n_groups} groups and {row.n_cells} cells "
            f"(thresholds: {min_groups_per_state} groups, {min_cells_per_state} cells); "
            "it stays in the reference but is flagged eligible=False."
        )
        msgs.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    logger.info(
        "validated metadata: %d cells, %d groups, %d states (%d eligible)",
        adata.n_obs,
        groups.nunique(),
        len(state_table),
        int(state_table["eligible"].sum()),
    )
    return MetadataReport(
        state_table=state_table,
        n_groups=int(groups.nunique()),
        n_cells=int(adata.n_obs),
        n_states=int(len(state_table)),
        n_studies=n_studies,
        warnings=tuple(msgs),
    )


def _looks_like_counts(X: Any, n_check: int = 2000) -> bool:
    block = X[:n_check]
    if hasattr(block, "toarray"):
        block = block.toarray()
    block = np.asarray(block)
    if block.size == 0:
        return False
    return bool((block >= 0).all() and np.allclose(block, np.round(block)))


def resolve_latent(
    adata: AnnData,
    *,
    latent_key: str | None,
    n_pcs: int = 50,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return the fixed latent representation StateTrace works in.

    With ``latent_key`` the matrix in ``obsm[latent_key]`` is used as is. Without it a PCA is
    computed on a lightweight copy (``normalize_total`` + ``log1p`` + 2,000 highly variable
    genes when ``n_vars > 2000``) and stored in ``obsm[schema.OBSM_PCA]``; this branch is the
    only place scanpy is imported.
    """
    if latent_key is not None:
        if latent_key not in adata.obsm:
            raise KeyError(
                f"obsm[{latent_key!r}] not found. Available: {list(adata.obsm.keys())}. "
                "Pass latent_key=None to compute a PCA fallback."
            )
        X = np.asarray(adata.obsm[latent_key], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"obsm[{latent_key!r}] must be 2-D, got shape {X.shape}.")
        return X.copy(), {
            "source": "obsm",
            "key": latent_key,
            "n_dims": int(X.shape[1]),
            "n_pcs": None,
            "hvg": None,
        }

    import anndata as ad
    import scanpy as sc

    if adata.n_vars < 2:
        raise ValueError("PCA fallback needs at least two genes; pass latent_key instead.")
    if not _looks_like_counts(adata.X):
        warnings.warn(
            "adata.X does not look like raw counts; the PCA fallback applies normalize_total "
            "and log1p, which assumes counts. Pass latent_key= to use a precomputed embedding.",
            UserWarning,
            stacklevel=2,
        )
    tmp = ad.AnnData(X=adata.X.copy(), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    sc.pp.normalize_total(tmp)
    sc.pp.log1p(tmp)
    hvg: int | None = None
    if tmp.n_vars > 2000:
        sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
        tmp = tmp[:, tmp.var["highly_variable"]].copy()
        hvg = 2000
    n_comps = int(min(n_pcs, tmp.n_vars - 1, tmp.n_obs - 1))
    sc.pp.pca(tmp, n_comps=n_comps, random_state=random_state)
    X = np.asarray(tmp.obsm["X_pca"], dtype=np.float32)
    adata.obsm[schema.OBSM_PCA] = X
    logger.info("computed PCA fallback with %d components (hvg=%s)", n_comps, hvg)
    return X.copy(), {
        "source": "pca",
        "key": schema.OBSM_PCA,
        "n_dims": n_comps,
        "n_pcs": n_comps,
        "hvg": hvg,
    }
