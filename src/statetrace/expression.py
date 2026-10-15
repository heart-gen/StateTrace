"""
The frozen biological space: donor x state pseudobulk shift diagnostics in expression space.

StateTrace evaluates a query in two coordinate systems. The *mapping space* (the latent of
scANVI, scPoli, Symphony or the built-in kNN) answers which reference state a query cell
belongs to. That space is fitted with the query present, so surgery, condition embeddings or
batch correction can absorb a state-specific displacement the same way they absorb a batch
effect. The *frozen biological space* answers a different question: is the molecular phenotype
of state ``s`` in the query donors within the between-donor variation of state ``s`` in the
reference? It is built once from the reference's harmonised expression, frozen with the
reference and never refitted on a query. Donors, not cells, are the unit throughout.

For every state ``s`` the procedure is:

1. donor x state pseudobulk profiles in gene space (summed counts, library-size normalised,
   ``log1p``);
2. the reference distribution of state ``s`` from reference donors only: gene-wise mean
   ``mu_gs`` and between-donor sd ``sigma_gs`` (shrunk towards the state's median gene sd);
3. standardisation ``z_gds = (x_gds - mu_gs) / sigma_gs``;
4. a PCA fitted on the standardised reference donor x state matrix (never refitted);
5. projection of query donor-state profiles;
6. the standardised distance of each query donor-state profile to the reference state
   centroid, decomposed into the part inside the reference PCA subspace and the residual
   outside it, plus the distance to the nearest reference donor and the fraction of genes
   beyond the reference donor range.

The null distribution of every metric comes from held-out reference donors: each donor is
scored against a space rebuilt without it (:func:`holdout_null`). Query donor-state distances
are reported as z-scores and empirical p-values against that null, not against a theoretical
cut-off.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRICS = ("distance", "distance_pc", "residual", "nearest", "genes_beyond")
NULL_COLUMNS = ("group", "state", "n_cells", *METRICS)


@dataclass(frozen=True)
class ExpressionConfig:
    """
    How the frozen expression space is built from a reference.

    Parameters
    ----------
    layer
        ``layers`` key holding raw counts; ``None`` uses ``X``.
    n_genes
        Genes kept, ranked by the ratio of between-state to between-donor variance of the
        reference pseudobulk profiles; all genes when the reference has fewer.
    n_components
        Upper bound on the PCA components (also bounded by the number of donor x state rows).
    min_cells
        A donor x state pseudobulk needs at least this many cells.
    min_groups
        A state needs at least this many reference donors to define a distribution.
    shrinkage
        Prior degrees of freedom pulling each gene's between-donor sd towards the state's
        median gene sd (guards against near-zero sds with few donors).
    target_sum
        Library-size normalisation target before ``log1p``.
    beyond_z
        ``|z|`` above which a gene counts as beyond the reference donor range.
    group_offset
        Remove each donor's own offset (the gene-wise median of its standardised profiles over
        the states it carries, when it carries at least ``offset_min_states``) before scoring.
        A donor's genotype, age, sex or processing batch moves all of its states together;
        a state-specific shift survives the median. Applied identically to held-out reference
        donors (the null) and to query donors.
    offset_min_states
        Minimum states a donor must carry for its own offset to be estimated; donors with
        fewer states receive the cohort-wide median offset instead.
    cell_sample_per_state, cell_components, cell_seed
        The frozen *cell-level* space for the two-sample tests of :mod:`statetrace.twosample`:
        at most ``cell_sample_per_state`` reference cells per state (seeded with
        ``cell_seed``) are log-normalised over the selected genes, standardised gene-wise and
        projected onto ``cell_components`` principal components fitted on those reference
        cells only. Only the projected coordinates, the gene-wise centre and scale and the
        loadings are frozen; ``cell_sample_per_state=0`` disables the cell-level space.
    n_jobs
        Worker processes over the held-out donors of :func:`holdout_null` (``-1`` uses every
        core), each with single-threaded native libraries. Execution only: the space does not
        depend on it.
    """

    layer: str | None = None
    n_genes: int = 2000
    n_components: int = 20
    min_cells: int = 20
    min_groups: int = 3
    shrinkage: float = 2.0
    target_sum: float = 1e4
    beyond_z: float = 2.0
    group_offset: bool = True
    offset_min_states: int = 3
    cell_sample_per_state: int = 500
    cell_components: int = 50
    cell_seed: int = 0
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)


@dataclass
class ExpressionSpace:
    """
    A frozen expression space built by :func:`build_expression_space`.

    Attributes
    ----------
    genes, states
        Feature names (a subset of the reference ``var_names``) and state names.
    profiles, rows
        Reference donor x state pseudobulk profiles (log-normalised, ``(n_rows, n_genes)``)
        and their ``group``, ``state``, ``n_cells``.
    mu, sigma, n_groups
        Per-state gene-wise mean and shrunk between-donor sd (``(n_states, n_genes)``) and the
        number of reference donors behind them.
    components
        PCA loadings ``(n_components, n_genes)`` fitted on the standardised reference rows.
    null
        Held-out reference donor metrics (one row per reference donor x state).
    null_stats
        Per-state ``n_holdout``, ``mean``, ``sd``, ``q95``, ``q99`` of the held-out distance.
    params
        The :class:`ExpressionConfig` used plus the input kind.
    null_profiles
        Held-out standardised profiles of the reference donors, aligned with ``null`` (each
        donor scored against the space rebuilt without it, offsets removed); ``None`` when
        read from a format-3 bundle.
    cell_center, cell_scale, cell_components, cell_coords, cell_rows
        The frozen cell-level space (``None`` when disabled or read from a format-3 bundle):
        gene-wise centre and scale of the sampled reference cells' log-normalised expression,
        PCA loadings ``(n_cell_components, n_genes)``, the sampled reference cells'
        coordinates ``(n_sampled, n_cell_components)`` and their ``group`` and ``state`` codes.
    """

    genes: np.ndarray
    states: np.ndarray
    profiles: np.ndarray
    rows: pd.DataFrame
    mu: np.ndarray
    sigma: np.ndarray
    n_groups: np.ndarray
    components: np.ndarray
    null: pd.DataFrame
    null_stats: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    null_profiles: np.ndarray | None = None
    cell_center: np.ndarray | None = None
    cell_scale: np.ndarray | None = None
    cell_components: np.ndarray | None = None
    cell_coords: np.ndarray | None = None
    cell_rows: pd.DataFrame | None = None

    @property
    def has_cell_space(self) -> bool:
        """Whether the frozen cell-level space for the two-sample tests is present."""
        return self.cell_coords is not None and self.cell_components is not None

    @property
    def n_genes(self) -> int:
        return int(len(self.genes))

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def summary(self) -> str:
        ok = int(np.isfinite(self.null_stats["sd"]).sum())
        return (
            f"expression space: {self.n_genes} genes, {len(self.rows)} reference donor x state "
            f"profiles, {self.n_components} components, held-out null for {ok}/{len(self.states)} states"
        )


# ------------------------------------------------------------------------- profiles
def counts_like(X: Any, *, n_check: int = 2000) -> bool:
    """``True`` when a sample of the matrix holds non-negative integers."""
    from scipy import sparse

    if X.shape[0] == 0 or X.shape[1] == 0:
        return True
    sub = X[: min(n_check, X.shape[0])]
    vals = sub.data if sparse.issparse(sub) else np.asarray(sub).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return True
    return bool((vals >= 0).all() and np.allclose(vals, np.round(vals)))


def pseudobulk_profiles(
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    *,
    min_cells: int,
    counts: bool = True,
    target_sum: float = 1e4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Sum (or average, for non-count input) cells per (group, state) and log-normalise.

    Cells with a negative state code are ignored. Returns the profiles ``(n_rows, n_genes)``
    and a table with ``group``, ``state`` (codes) and ``n_cells``.
    """
    from scipy import sparse

    g = np.asarray(group_codes)
    s = np.asarray(state_codes)
    keep = (s >= 0) & (g >= 0)
    key = g[keep].astype(np.int64) * (int(s.max()) + 1 if s.size else 1) + s[keep]
    codes, uniques = pd.factorize(key, sort=True)
    n_cells = np.bincount(codes, minlength=len(uniques))
    ind = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniques), int(keep.sum()))
    )
    summed = ind @ X[keep]
    summed = np.asarray(summed.todense() if sparse.issparse(summed) else summed, dtype=np.float64)
    ns = int(s.max()) + 1 if s.size else 1
    rows = pd.DataFrame(
        {
            "group": (uniques // ns).astype(int),
            "state": (uniques % ns).astype(int),
            "n_cells": n_cells,
        }
    )
    ok = rows["n_cells"].to_numpy() >= int(min_cells)
    summed, rows = summed[ok], rows[ok].reset_index(drop=True)
    if counts:
        lib = summed.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            summed = np.where(lib > 0, summed / lib * float(target_sum), 0.0)
        profiles = np.log1p(summed)
    else:
        profiles = summed / rows["n_cells"].to_numpy()[:, None].clip(min=1)
    return profiles, rows


# ------------------------------------------------------------------------------ fit
def _state_stats(
    profiles: np.ndarray, rows: pd.DataFrame, n_states: int, *, min_groups: int, shrinkage: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    G = profiles.shape[1]
    mu = np.full((n_states, G), np.nan)
    sigma = np.full((n_states, G), np.nan)
    n = np.zeros(n_states, dtype=int)
    st = rows["state"].to_numpy()
    for k in range(n_states):
        sel = st == k
        n[k] = int(sel.sum())
        if n[k] < max(2, min_groups):
            continue
        P = profiles[sel]
        mu[k] = P.mean(axis=0)
        sd = P.std(axis=0, ddof=1)
        s0 = float(np.median(sd[sd > 0])) if (sd > 0).any() else 1.0
        df = n[k] - 1
        sigma[k] = np.sqrt((df * sd**2 + shrinkage * s0**2) / (df + shrinkage))
        sigma[k] = np.maximum(sigma[k], 1e-3)
    return mu, sigma, n


def _standardise(profiles: np.ndarray, rows: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray):
    st = rows["state"].to_numpy()
    Z = (profiles - mu[st]) / sigma[st]
    ok = np.isfinite(Z).all(axis=1)
    return Z, ok


def _pca(Z: np.ndarray, n_components: int) -> np.ndarray:
    if Z.shape[0] < 2:
        return np.zeros((0, Z.shape[1]))
    Zc = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    k = int(min(n_components, Z.shape[0] - 1, Z.shape[1]))
    return vt[:k]


def _fit(
    profiles: np.ndarray, rows: pd.DataFrame, n_states: int, cfg: ExpressionConfig
) -> dict[str, Any]:
    mu, sigma, n = _state_stats(
        profiles, rows, n_states, min_groups=cfg.min_groups, shrinkage=cfg.shrinkage
    )
    Z, ok = _standardise(profiles, rows, mu, sigma)
    comps = _pca(Z[ok], cfg.n_components)
    return {
        "mu": mu,
        "sigma": sigma,
        "n": n,
        "Z": Z[ok],
        "rows": rows[ok].reset_index(drop=True),
        "components": comps,
    }


def score_profiles(
    Z: np.ndarray,
    state_codes: np.ndarray,
    *,
    components: np.ndarray | None,
    reference_Z: np.ndarray,
    reference_states: np.ndarray,
    beyond_z: float = 2.0,
) -> pd.DataFrame:
    """
    Metrics of standardised profiles ``Z`` (rows x genes) against a fitted space.

    ``distance`` is the RMS standardised distance to the reference state centroid (zero by
    construction); ``distance_pc`` and ``residual`` split it into the reference PCA subspace
    and its complement (``distance^2 = distance_pc^2 + residual^2``); ``nearest`` is the RMS
    distance to the closest reference donor of the same state; ``genes_beyond`` the fraction
    of genes with ``|z|`` above ``beyond_z``.
    """
    G = Z.shape[1]
    out = pd.DataFrame(index=range(Z.shape[0]), columns=list(METRICS), dtype=float)
    out["distance"] = np.sqrt((Z**2).mean(axis=1))
    if components is not None and components.shape[0] > 0:
        S = Z @ components.T
        out["distance_pc"] = np.sqrt((S**2).sum(axis=1) / G)
        out["residual"] = np.sqrt(np.clip(out["distance"] ** 2 - out["distance_pc"] ** 2, 0, None))
    out["genes_beyond"] = (np.abs(Z) > beyond_z).mean(axis=1)
    ref_states = np.asarray(reference_states)
    for i, k in enumerate(np.asarray(state_codes)):
        R = reference_Z[ref_states == k]
        if R.shape[0]:
            out.loc[i, "nearest"] = float(np.sqrt(((R - Z[i]) ** 2).mean(axis=1)).min())
    return out


def remove_group_offsets(
    Z: np.ndarray,
    group_codes: np.ndarray,
    *,
    min_states: int = 3,
    cohort_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[int, np.ndarray]]:
    """
    Subtract each donor's gene-wise median standardised profile over the states it carries.

    Donors with fewer than ``min_states`` profiles receive ``cohort_offset`` (the median of
    the estimated donor offsets, or the given vector) instead. Returns the centred matrix,
    the cohort offset actually used (``None`` when nothing could be estimated) and the
    offset applied to each donor (own or cohort).
    """
    g = np.asarray(group_codes)
    Z = np.array(Z, dtype=np.float64, copy=True)
    offsets: dict[int, np.ndarray] = {}
    for d in np.unique(g):
        sel = g == d
        if sel.sum() >= min_states:
            offsets[int(d)] = np.median(Z[sel], axis=0)
    if cohort_offset is None and offsets:
        cohort_offset = np.median(np.vstack(list(offsets.values())), axis=0)
    applied: dict[int, np.ndarray] = {}
    for d in np.unique(g):
        sel = g == d
        off = offsets.get(int(d), cohort_offset)
        if off is not None:
            Z[sel] -= off
            applied[int(d)] = off
    return Z, cohort_offset, applied


def _holdout_donor(
    profiles: np.ndarray, rows: pd.DataFrame, d: int, n_states: int, cfg: ExpressionConfig
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Donor ``d``'s profiles scored against a space rebuilt without it (see :func:`holdout_null`)."""
    held = rows["group"].to_numpy() == d
    fit = _fit(profiles[~held], rows[~held].reset_index(drop=True), n_states, cfg)
    sub = rows[held].reset_index(drop=True)
    st = sub["state"].to_numpy()
    Z = (profiles[held] - fit["mu"][st]) / fit["sigma"][st]
    ok = np.isfinite(Z).all(axis=1)
    if not ok.any():
        return None
    Z, st, sub = Z[ok], st[ok], sub[ok].reset_index(drop=True)
    ref_Z, ref_st = fit["Z"], fit["rows"]["state"].to_numpy()
    if cfg.group_offset:
        # the held-out donor is centred on itself; a single donor has no cohort fallback
        Z, _, _ = remove_group_offsets(
            Z, np.zeros(len(st), dtype=int), min_states=cfg.offset_min_states
        )
        ref_Z, _, _ = remove_group_offsets(
            ref_Z, fit["rows"]["group"].to_numpy(), min_states=cfg.offset_min_states
        )
    m = score_profiles(
        Z,
        st,
        components=fit["components"],
        reference_Z=ref_Z,
        reference_states=ref_st,
        beyond_z=cfg.beyond_z,
    )
    m.insert(0, "n_cells", sub["n_cells"].to_numpy())
    m.insert(0, "state", st)
    m.insert(0, "group", int(d))
    return m, Z


def holdout_null(
    profiles: np.ndarray,
    rows: pd.DataFrame,
    n_states: int,
    cfg: ExpressionConfig,
    *,
    return_profiles: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Score every reference donor x state profile against a space rebuilt without that donor.

    With ``return_profiles`` also return the held-out standardised profiles (offsets removed
    exactly as for scoring), aligned with the rows of the table: reference donors seen the
    way a query donor is seen, the reference side of the donor-level two-sample tests.
    """
    donors = np.unique(rows["group"].to_numpy())
    if cfg.n_jobs and cfg.n_jobs != 1:
        from joblib import Parallel, delayed

        scored = Parallel(n_jobs=cfg.n_jobs)(
            delayed(_holdout_donor)(profiles, rows, d, n_states, cfg) for d in donors
        )
    else:
        scored = [_holdout_donor(profiles, rows, d, n_states, cfg) for d in donors]
    records = [m for m, _ in filter(None, scored)]
    held_Z = [Z for _, Z in filter(None, scored)]
    if not records:
        empty = pd.DataFrame(columns=list(NULL_COLUMNS))
        return (empty, np.zeros((0, profiles.shape[1]))) if return_profiles else empty
    table = pd.concat(records, ignore_index=True)[list(NULL_COLUMNS)]
    return (table, np.vstack(held_Z)) if return_profiles else table


def null_statistics(null: pd.DataFrame, n_states: int, states: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(n_states):
        d = null.loc[null["state"] == k, "distance"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        rows.append(
            {
                "state": str(states[k]),
                "n_holdout": int(d.size),
                "mean": float(d.mean()) if d.size else np.nan,
                "sd": float(d.std(ddof=1)) if d.size > 2 else np.nan,
                "q95": float(np.quantile(d, 0.95)) if d.size else np.nan,
                "q99": float(np.quantile(d, 0.99)) if d.size else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("state")


def select_genes(
    profiles: np.ndarray, rows: pd.DataFrame, n_states: int, *, n_genes: int
) -> np.ndarray:
    """
    Indices of the ``n_genes`` genes with the highest ratio of between-state variance of the
    donor-mean profile to the mean within-state between-donor variance (all genes when fewer).
    """
    G = profiles.shape[1]
    if n_genes <= 0 or G <= n_genes:
        return np.arange(G)
    st = rows["state"].to_numpy()
    means, within = [], []
    for k in range(n_states):
        sel = st == k
        if sel.sum() >= 2:
            means.append(profiles[sel].mean(axis=0))
            within.append(profiles[sel].var(axis=0, ddof=1))
    if len(means) < 2:
        score = profiles.var(axis=0)
    else:
        between = np.var(np.vstack(means), axis=0)
        score = between / (np.mean(np.vstack(within), axis=0) + 1e-6)
    return np.sort(np.argsort(score)[::-1][:n_genes])


def build_expression_space(
    X: Any,
    var_names: np.ndarray | pd.Index,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    states: np.ndarray,
    *,
    config: ExpressionConfig | None = None,
) -> ExpressionSpace:
    """
    Build the frozen expression space of a reference from its cells x genes matrix.

    Parameters
    ----------
    X
        Reference counts (``adata.X`` or a layer). Non-count input is averaged per donor x
        state without normalisation and recorded as ``input="non_count"``.
    var_names
        Feature names aligned to the columns of ``X``.
    group_codes, state_codes
        Integer codes per cell; ``states`` gives the names for the state codes.
    config
        :class:`ExpressionConfig`.
    """
    cfg = config or ExpressionConfig()
    genes_all = np.asarray(var_names).astype(str)
    n_states = len(states)
    counts = counts_like(X)
    if not counts:
        warnings.warn(
            "reference expression does not look like raw counts; donor x state profiles are "
            "averaged without normalisation. Pass counts through ExpressionConfig(layer=...) "
            "for the intended pseudobulk model.",
            UserWarning,
            stacklevel=2,
        )
    profiles, rows = pseudobulk_profiles(
        X,
        group_codes,
        state_codes,
        min_cells=cfg.min_cells,
        counts=counts,
        target_sum=cfg.target_sum,
    )
    if len(rows) < 2:
        raise ValueError("fewer than two donor x state pseudobulk profiles; lower min_cells")
    idx = select_genes(profiles, rows, n_states, n_genes=cfg.n_genes)
    profiles = profiles[:, idx]
    genes = genes_all[idx]
    fit = _fit(profiles, rows, n_states, cfg)
    null, null_profiles = holdout_null(profiles, rows, n_states, cfg, return_profiles=True)
    stats = null_statistics(null, n_states, np.asarray(states).astype(str))
    cell = build_cell_space(
        X,
        group_codes,
        state_codes,
        gene_index=idx,
        n_states=n_states,
        counts=counts,
        config=cfg,
    )
    space = ExpressionSpace(
        genes=genes,
        states=np.asarray(states).astype(str),
        profiles=profiles.astype(np.float32),
        rows=rows,
        mu=fit["mu"].astype(np.float32),
        sigma=fit["sigma"].astype(np.float32),
        n_groups=fit["n"],
        components=fit["components"].astype(np.float32),
        null=null,
        null_stats=stats,
        params={
            **cfg.to_dict(),
            "input": "counts" if counts else "non_count",
            "n_genes_input": int(len(genes_all)),
        },
        null_profiles=null_profiles.astype(np.float32),
        **cell,
    )
    logger.info(space.summary())
    return space


# ----------------------------------------------------------------------- cell space
CELL_CLIP = 10.0  # standardised cell values are clipped to +/- this (scanpy's scale convention)


def _cell_values(X: Any, *, counts: bool, target_sum: float) -> np.ndarray:
    """Dense log-normalised values of a cells x genes block (library over the given genes)."""
    from scipy import sparse

    V = np.asarray(X.todense() if sparse.issparse(X) else X, dtype=np.float64)
    if not counts:
        return V
    lib = V.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(lib > 0, V / lib * float(target_sum), 0.0)
    return np.log1p(V)


def _standardise_cells(V: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((V - center) / scale, -CELL_CLIP, CELL_CLIP)


def build_cell_space(
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    *,
    gene_index: np.ndarray,
    n_states: int,
    counts: bool,
    config: ExpressionConfig,
) -> dict[str, Any]:
    """
    The frozen cell-level space: a seeded sample of at most ``cell_sample_per_state``
    reference cells per state, log-normalised over the selected genes, standardised gene-wise
    and projected onto principal components fitted on those reference cells only.

    Returns the keyword arguments of :class:`ExpressionSpace` that hold it (all ``None`` when
    the space is disabled or fewer than two cells are available).
    """
    empty = {
        "cell_center": None,
        "cell_scale": None,
        "cell_components": None,
        "cell_coords": None,
        "cell_rows": None,
    }
    if config.cell_sample_per_state <= 0 or config.cell_components <= 0:
        return empty
    rng = np.random.default_rng(config.cell_seed)
    g = np.asarray(group_codes)
    s = np.asarray(state_codes)
    picked = []
    for k in range(n_states):
        pool = np.flatnonzero((s == k) & (g >= 0))
        if pool.size > config.cell_sample_per_state:
            pool = np.sort(rng.choice(pool, size=config.cell_sample_per_state, replace=False))
        picked.append(pool)
    cells = np.concatenate(picked) if picked else np.zeros(0, dtype=int)
    if cells.size < 2:
        return empty
    V = _cell_values(X[cells][:, gene_index], counts=counts, target_sum=config.target_sum)
    center = V.mean(axis=0)
    scale = np.maximum(V.std(axis=0, ddof=1), 1e-2)
    Z = _standardise_cells(V, center, scale)
    comps = _pca(Z, config.cell_components)
    coords = Z @ comps.T  # uncentred, exactly as project_cells() places query cells
    return {
        "cell_center": center.astype(np.float32),
        "cell_scale": scale.astype(np.float32),
        "cell_components": comps.astype(np.float32),
        "cell_coords": coords.astype(np.float32),
        "cell_rows": pd.DataFrame({"group": g[cells].astype(int), "state": s[cells].astype(int)}),
    }


def project_cells(
    space: ExpressionSpace,
    X: Any,
    *,
    gene_index: np.ndarray | None = None,
    counts: bool | None = None,
) -> np.ndarray:
    """
    Coordinates of query cells in the frozen cell-level space.

    ``X`` holds the query cells x genes aligned to ``space.genes[gene_index]``. Genes of the
    space that the query lacks are set to the reference centre (standardised value 0), which
    pulls the query towards the reference; the retention is reported by the caller.
    """
    if not space.has_cell_space:
        raise ValueError("the expression space carries no cell-level space")
    idx = np.arange(space.n_genes) if gene_index is None else np.asarray(gene_index)
    input_counts = counts_like(X) if counts is None else bool(counts)
    V = _cell_values(X, counts=input_counts, target_sum=float(space.params.get("target_sum", 1e4)))
    center = np.asarray(space.cell_center, dtype=np.float64)
    scale = np.asarray(space.cell_scale, dtype=np.float64)
    Z = np.zeros((V.shape[0], space.n_genes))
    Z[:, idx] = _standardise_cells(V, center[idx], scale[idx])
    comps = np.asarray(space.cell_components, dtype=np.float64)
    return Z @ comps.T


def cell_coordinates(
    space: ExpressionSpace,
    X: Any,
    var_names: np.ndarray | pd.Index,
    *,
    chunk_size: int = 20000,
) -> np.ndarray:
    """
    Coordinates of every cell of ``X`` (cells x ``var_names``) in the frozen cell-level space,
    projected ``chunk_size`` cells at a time. Every gene of the space must be present, as it
    is for the reference the space was built from.
    """
    if not space.has_cell_space:
        raise ValueError("the expression space carries no cell-level space")
    col = pd.Index(np.asarray(var_names).astype(str)).get_indexer(space.genes)
    if (col < 0).any():
        raise ValueError(f"{int((col < 0).sum())} genes of the expression space are missing")
    counts = space.params.get("input", "counts") == "counts"
    n = X.shape[0]
    out = np.empty((n, space.cell_components.shape[0]), dtype=np.float32)
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        out[start:stop] = project_cells(space, X[start:stop][:, col], counts=counts)
    return out


# ---------------------------------------------------------------------------- query
def _subset(space: ExpressionSpace, idx: np.ndarray) -> dict[str, Any]:
    full = len(idx) == space.n_genes and np.array_equal(idx, np.arange(space.n_genes))
    return {
        "mu": space.mu[:, idx].astype(np.float64),
        "sigma": space.sigma[:, idx].astype(np.float64),
        "profiles": space.profiles[:, idx].astype(np.float64),
        "components": space.components.astype(np.float64) if full else None,
    }


def estimate_cohort_offset(
    Z: np.ndarray, state_codes: np.ndarray, *, min_states: int = 3
) -> np.ndarray | None:
    """
    Gene-wise median over states of the mean standardised displacement of the query donors:
    a shared platform or cohort effect. ``None`` when fewer than ``min_states`` states carry
    query profiles (the median of few states would remove state-specific shift).
    """
    st = np.asarray(state_codes)
    per_state = [Z[st == k].mean(axis=0) for k in np.unique(st)]
    if len(per_state) < min_states:
        return None
    return np.median(np.vstack(per_state), axis=0)


def query_profiles(
    space: ExpressionSpace,
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    *,
    gene_index: np.ndarray | None = None,
    min_cells: int = 20,
    correct_cohort_offset: bool = True,
    counts: bool | None = None,
) -> dict[str, Any]:
    """
    Standardised query and reference donor x state profiles in the frozen space, with the
    offsets of :func:`score_query` removed; the shared first half of :func:`score_query`
    (also consumed by the donor-level two-sample tests of :mod:`statetrace.twosample`).

    Returns a dict with ``Z`` / ``Z_raw`` (query rows with and without offset removal),
    ``rows`` and ``states`` (their group and state codes), ``ref_Z`` / ``ref_Z_raw``,
    ``ref_rows`` and ``ref_states`` (the reference rows, offsets removed the same way), and
    the offsets applied.
    """
    idx = np.arange(space.n_genes) if gene_index is None else np.asarray(gene_index)
    sub = _subset(space, idx)
    input_counts = counts_like(X) if counts is None else bool(counts)
    profiles, rows = pseudobulk_profiles(
        X,
        group_codes,
        state_codes,
        min_cells=min_cells,
        counts=input_counts,
        target_sum=float(space.params.get("target_sum", 1e4)),
    )
    st = rows["state"].to_numpy()
    Z = (profiles - sub["mu"][st]) / sub["sigma"][st]
    ok = np.isfinite(Z).all(axis=1)
    Z, rows, st = Z[ok], rows[ok].reset_index(drop=True), st[ok]
    offset = None
    per_group = bool(space.params.get("group_offset", False))
    min_states = int(space.params.get("offset_min_states", 3))
    ref_rows = space.rows
    ref_st = ref_rows["state"].to_numpy()
    ref_Z = (sub["profiles"] - sub["mu"][ref_st]) / sub["sigma"][ref_st]
    ref_ok = np.isfinite(ref_Z).all(axis=1)
    ref_Z = ref_Z[ref_ok]
    ref_Z_raw = ref_Z
    Z_raw = Z
    applied: dict[int, np.ndarray] = {}
    if correct_cohort_offset and len(rows):
        if per_group:
            Z, offset, applied = remove_group_offsets(
                Z, rows["group"].to_numpy(), min_states=min_states
            )
        else:
            offset = estimate_cohort_offset(Z, st, min_states=min_states)
            if offset is not None:
                Z = Z - offset
                applied = {int(d): offset for d in np.unique(rows["group"].to_numpy())}
    if per_group:
        ref_Z, _, _ = remove_group_offsets(
            ref_Z, ref_rows.loc[ref_ok, "group"].to_numpy(), min_states=min_states
        )
    return {
        "idx": idx,
        "sub": sub,
        "input_counts": input_counts,
        "Z": Z,
        "Z_raw": Z_raw,
        "rows": rows,
        "states": st,
        "ref_Z": ref_Z,
        "ref_Z_raw": ref_Z_raw,
        "ref_rows": ref_rows[ref_ok].reset_index(drop=True),
        "ref_states": ref_st[ref_ok],
        "offset": offset,
        "applied": applied,
        "per_group": per_group,
    }


def score_query(
    space: ExpressionSpace,
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    *,
    gene_index: np.ndarray | None = None,
    min_cells: int = 20,
    correct_cohort_offset: bool = True,
    counts: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Score query donor x state pseudobulk profiles against a frozen expression space.

    Parameters
    ----------
    space
        A frozen :class:`ExpressionSpace`.
    X
        Query cells x genes, with columns aligned to ``space.genes`` (or to
        ``space.genes[gene_index]`` when some reference genes are missing from the query).
    group_codes, state_codes
        Query donor codes and assigned state codes (indexing ``space.states``); a negative
        state code excludes the cell.
    gene_index
        Indices into ``space.genes`` of the columns of ``X`` (``None``: all, in order).
    min_cells
        Minimum cells per donor x state profile.
    correct_cohort_offset
        Remove offsets before scoring: each query donor's own offset when the space was built
        with ``group_offset`` (the default), else the gene-wise median displacement across
        states shared by the whole cohort.
    counts
        Whether ``X`` holds raw counts; detected when ``None``.

    Returns
    -------
    table, provenance
        One row per query donor x state with ``group``, ``state`` (codes), ``n_cells``, the
        metrics of :func:`score_profiles`, ``shift_z`` (distance standardised against the
        held-out null of that state) and ``shift_p`` (empirical upper-tail probability);
        provenance records the cohort offset size and feature retention.
    """
    q = query_profiles(
        space,
        X,
        group_codes,
        state_codes,
        gene_index=gene_index,
        min_cells=min_cells,
        correct_cohort_offset=correct_cohort_offset,
        counts=counts,
    )
    idx, sub, input_counts = q["idx"], q["sub"], q["input_counts"]
    Z, Z_raw, rows, st = q["Z"], q["Z_raw"], q["rows"], q["states"]
    ref_Z, ref_Z_raw, ref_st = q["ref_Z"], q["ref_Z_raw"], q["ref_states"]
    offset, applied, per_group = q["offset"], q["applied"], q["per_group"]
    beyond = float(space.params.get("beyond_z", 2.0))
    if len(rows):
        m = score_profiles(
            Z,
            st,
            components=sub["components"],
            reference_Z=ref_Z,
            reference_states=ref_st,
            beyond_z=beyond,
        )
        # the same profiles before any offset removal, against the uncentred reference rows:
        # what the donor offset took out, reported next to the residual
        m_raw = score_profiles(
            Z_raw,
            st,
            components=sub["components"],
            reference_Z=ref_Z_raw,
            reference_states=ref_st,
            beyond_z=beyond,
        )
        m["distance_raw"] = m_raw["distance"].to_numpy()
        m["offset_rms"] = [
            float(np.sqrt((applied[int(d)] ** 2).mean())) if int(d) in applied else np.nan
            for d in rows["group"].to_numpy()
        ]
    else:
        m = pd.DataFrame(columns=[*METRICS, "distance_raw", "offset_rms"], dtype=float)
    table = pd.concat([rows, m], axis=1)
    stats = space.null_stats.reset_index(drop=True)
    mean = stats["mean"].to_numpy(dtype=float)
    sd = stats["sd"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["shift_z"] = (table["distance"].to_numpy(dtype=float) - mean[st]) / sd[st]
        table["shift_z_raw"] = (table["distance_raw"].to_numpy(dtype=float) - mean[st]) / sd[st]
    pvals = np.full(len(table), np.nan)
    for i, (k, d) in enumerate(zip(st, table["distance"].to_numpy(dtype=float), strict=True)):
        nd = space.null.loc[space.null["state"] == k, "distance"].to_numpy(dtype=float)
        nd = nd[np.isfinite(nd)]
        if nd.size and np.isfinite(d):
            pvals[i] = (1.0 + float((nd >= d).sum())) / (1.0 + nd.size)
    table["shift_p"] = pvals
    prov = {
        "feature_retention": float(len(idx) / space.n_genes) if space.n_genes else np.nan,
        "n_genes_scored": int(len(idx)),
        "cohort_offset_rms": float(np.sqrt((offset**2).mean())) if offset is not None else None,
        "offset": "group"
        if (correct_cohort_offset and per_group)
        else ("cohort" if correct_cohort_offset else "none"),
        "input": "counts" if input_counts else "non_count",
        "pc_decomposition": sub["components"] is not None,
    }
    return table, prov


def summarise_states(
    table: pd.DataFrame, n_states: int, *, high: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-state summary of a :func:`score_query` table: median ``shift_z`` over query donors,
    number of donors at or above ``high``, and number of donors scored.
    """
    med = np.full(n_states, np.nan)
    n_hi = np.zeros(n_states, dtype=int)
    n_sc = np.zeros(n_states, dtype=int)
    if len(table) == 0:
        return med, n_hi, n_sc
    st = table["state"].to_numpy()
    z = table["shift_z"].to_numpy(dtype=float)
    for k in range(n_states):
        sel = (st == k) & np.isfinite(z)
        n_sc[k] = int(sel.sum())
        if sel.any():
            med[k] = float(np.median(z[sel]))
            n_hi[k] = int((z[sel] >= high).sum())
    return med, n_hi, n_sc
