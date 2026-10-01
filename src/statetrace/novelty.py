"""
Distribution-shift diagnostics in a shared latent space.

Three things can happen to a reference state in a query: it is *absent*, it is present but
*shifted* (its query cells sit away from the reference cells of that state), or query cells form
a *novel* population no reference state covers. Per cell StateTrace reports a standardised
reference distance (how many reference standard deviations a cell's distance to its assigned
state's centroid lies above the typical distance of that state's own reference cells) and a
category; per state, a centroid shift in per-dimension standard-deviation units.

A whole-cohort offset (a study or batch effect that moves every query cell the same way) is
estimated as the component-wise median of the per-state centroid displacements and removed
before scoring, so it is reported once as ``cohort_offset`` rather than as shift in every state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import schema


def state_geometry(
    latent: np.ndarray, state_codes: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-state centroid and distance statistics.

    Returns
    -------
    centroids
        (n_states, n_dims).
    geometry
        (n_states, 3): RMS radius, mean and standard deviation of the distance of the state's
        reference cells to its centroid. NaN for states without cells.
    """
    d = latent.shape[1]
    centroids = np.zeros((n_states, d), dtype=np.float32)
    geometry = np.full((n_states, 3), np.nan, dtype=np.float32)
    for k in range(n_states):
        sel = state_codes == k
        if sel.any():
            c = latent[sel].mean(axis=0)
            centroids[k] = c
            dist = np.sqrt(((latent[sel] - c) ** 2).sum(axis=1))
            geometry[k] = (
                np.sqrt((dist**2).mean()),
                dist.mean(),
                dist.std(ddof=1) if sel.sum() > 1 else np.nan,
            )
    return centroids, geometry


def estimate_cohort_offset(
    query_latent: np.ndarray, query_codes: np.ndarray, centroids: np.ndarray, *, min_cells: int = 20
) -> np.ndarray:
    """Component-wise median of (query state centroid - reference state centroid) over states with enough cells."""
    disp = []
    for k in range(centroids.shape[0]):
        sel = query_codes == k
        if sel.sum() >= min_cells:
            disp.append(query_latent[sel].mean(axis=0) - centroids[k])
    if not disp:
        return np.zeros(centroids.shape[1], dtype=np.float32)
    return np.median(np.vstack(disp), axis=0).astype(np.float32)


def raw_distance(
    latent: np.ndarray, assigned_codes: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Euclidean distance of each cell to its assigned state's reference centroid (NaN if unassigned)."""
    assigned_codes = np.asarray(assigned_codes)
    out = np.full(len(assigned_codes), np.nan)
    ok = assigned_codes >= 0
    if ok.any():
        out[ok] = np.sqrt(((latent[ok] - centroids[assigned_codes[ok]]) ** 2).sum(axis=1))
    return out


def reference_distance(
    latent: np.ndarray, assigned_codes: np.ndarray, centroids: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    """
    Standardised distance of each cell to its assigned state's reference centroid:
    ``(distance - mean_ref_distance) / sd_ref_distance`` for that state. Around 0 for cells that
    look like the state's own reference cells, large and positive for out-of-reference cells.
    """
    assigned_codes = np.asarray(assigned_codes)
    out = np.full(len(assigned_codes), np.nan)
    ok = assigned_codes >= 0
    if ok.any():
        dist = raw_distance(latent, assigned_codes, centroids)[ok]
        mean = geometry[assigned_codes[ok], 1]
        sd = geometry[assigned_codes[ok], 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.where(sd > 0, (dist - mean) / sd, np.nan)
    return out


OOD_RULES = ("z", "quantile")


def out_of_reference_mask(
    distance: np.ndarray,
    novelty: np.ndarray,
    assigned_codes: np.ndarray,
    *,
    rule: str = "z",
    z: float = 3.0,
    quantile: float = 0.99,
    calibration: pd.DataFrame | None = None,
    states: np.ndarray | list[str] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Decide which cells lie outside their assigned state's reference distribution.

    ``rule="z"`` flags cells whose standardised reference distance exceeds ``z``.
    ``rule="quantile"`` flags cells whose raw distance exceeds the ``quantile`` (0.95 or 0.99)
    of the held-out distances of that state in ``calibration`` (see
    :func:`holdout_distance_calibration`), so a genuinely new donor of an intact state is
    flagged at about the nominal rate; states without a calibrated quantile fall back to the
    z rule, and so does the whole call when no calibration exists.

    Returns
    -------
    mask, rule_applied
        Boolean mask and the rule that was actually applied (``"z"`` when the quantile rule
        could not be).
    """
    if rule not in OOD_RULES:
        raise ValueError(f"rule must be one of {OOD_RULES}, got {rule!r}")
    nov = np.asarray(novelty, dtype=float)
    z_mask = np.nan_to_num(nov, nan=0.0) > z
    if rule == "z" or calibration is None or states is None:
        return z_mask, "z"
    col = {0.95: "q95_holdout", 0.99: "q99_holdout"}.get(float(quantile))
    if col is None:
        raise ValueError("quantile must be 0.95 or 0.99 (the levels stored with the reference)")
    cal = calibration.reindex([str(s) for s in states])
    q = cal[col].to_numpy(dtype=float)
    codes = np.asarray(assigned_codes)
    dist = np.asarray(distance, dtype=float)
    ok = codes >= 0
    thr = np.full(len(codes), np.nan)
    thr[ok] = q[codes[ok]]
    has_q = np.isfinite(thr)
    mask = np.where(has_q, np.nan_to_num(dist, nan=0.0) > thr, z_mask)
    return mask, "quantile"


def local_support(
    reference_latent: np.ndarray,
    reference_labels: np.ndarray,
    query_latent: np.ndarray,
    assigned_codes: np.ndarray,
    states: np.ndarray | list[str],
    *,
    n_neighbors: int = 15,
    metric: str = "cosine",
    backend: Any = None,
) -> np.ndarray:
    """
    Fraction of each query cell's ``n_neighbors`` nearest *reference* cells that carry the
    state the cell was assigned, in the coordinate system of ``query_latent``.

    This is StateTrace's own measure of local reference support, independent of whether the
    mapping tool abstains or how it scores confidence: a label whose neighbourhood in the
    tool's latent is made of other states has no reference support. ``backend`` may be an
    already fitted object with ``neighbor_votes`` and ``classes_`` (the reference's kNN
    backend, fitted on the state labels); otherwise a
    :class:`~statetrace.backends.knn.KNNBackend` is fitted on the reference latent. NaN for
    cells with no assigned state.
    """
    from .backends.knn import KNNBackend

    codes = np.asarray(assigned_codes)
    states = np.asarray(states).astype(str)
    if backend is None or not hasattr(backend, "neighbor_votes"):
        backend = KNNBackend(n_neighbors=n_neighbors, metric=metric).fit(
            reference_latent, np.asarray(reference_labels).astype(str)
        )
    classes = np.asarray(backend.classes_).astype(str)
    lookup = {c: i for i, c in enumerate(classes)}
    col = np.array([lookup.get(s, -1) for s in states])
    votes = backend.neighbor_votes(query_latent)
    out = np.full(len(codes), np.nan)
    ok = codes >= 0
    ok[ok] &= col[codes[ok]] >= 0
    if ok.any():
        total = votes[ok].sum(axis=1).clip(min=1)
        out[ok] = votes[ok, col[codes[ok]]] / total
    return out


def cell_category(
    confidence: np.ndarray,
    novelty: np.ndarray,
    *,
    uncertain_below: float,
    ood_z: float,
    support: np.ndarray | None = None,
    min_support: float | None = None,
    out_of_reference: np.ndarray | None = None,
) -> np.ndarray:
    """
    ``out_of_reference`` for cells in ``out_of_reference`` (default: novelty z above ``ood_z``);
    else ``uncertain`` when confidence is below ``uncertain_below`` or local reference support
    is below ``min_support``; else ``supported``. Missing confidence or support never makes a
    cell uncertain.
    """
    conf = np.asarray(confidence, dtype=float)
    nov = np.asarray(novelty, dtype=float)
    cat = np.full(len(conf), "supported", dtype=object)
    cat[np.nan_to_num(conf, nan=1.0) < uncertain_below] = "uncertain"
    if support is not None and min_support is not None:
        sup = np.asarray(support, dtype=float)
        cat[np.nan_to_num(sup, nan=1.0) < min_support] = "uncertain"
    ood = (
        np.asarray(out_of_reference, dtype=bool)
        if out_of_reference is not None
        else np.nan_to_num(nov, nan=0.0) > ood_z
    )
    cat[ood] = "out_of_reference"
    return cat


def centroid_shift(
    query_latent: np.ndarray,
    query_codes: np.ndarray,
    centroids: np.ndarray,
    geometry: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Distance between the query centroid of each assigned state and its reference centroid, in
    per-dimension standard-deviation units (RMS radius / sqrt(n_dims)), so a displacement along
    one biological axis is not diluted by the latent dimensionality.
    """
    d = query_latent.shape[1]
    out = np.full(n_states, np.nan)
    for k in range(n_states):
        sel = query_codes == k
        per_dim_sd = geometry[k, 0] / np.sqrt(d)
        if sel.sum() >= 2 and np.isfinite(per_dim_sd) and per_dim_sd > 0:
            c = query_latent[sel].mean(axis=0)
            out[k] = np.sqrt(((c - centroids[k]) ** 2).sum()) / per_dim_sd
    return out


def shift_level(shift: np.ndarray | pd.Series, *, moderate: float, high: float) -> np.ndarray:
    s = np.asarray(shift, dtype=float)
    lvl = np.full(len(s), "unknown", dtype=object)
    lvl[np.isfinite(s) & (s < moderate)] = "low"
    lvl[np.isfinite(s) & (s >= moderate) & (s < high)] = "moderate"
    lvl[np.isfinite(s) & (s >= high)] = "high"
    assert set(lvl) <= set(schema.SHIFT_LEVELS)
    return lvl


def holdout_distance_calibration(
    latent: np.ndarray,
    state_codes: np.ndarray,
    folds: list[tuple[str, np.ndarray]],
    n_states: int,
    states: np.ndarray | list[str],
) -> pd.DataFrame:
    """
    Per-state distance statistics of *held-out* cells to centroids built from the training
    folds only. Unlike :func:`state_geometry`, these include the effect of an unseen group
    (donor), so standardising query distances against them gives an out-of-reference rate
    close to the nominal one for genuinely new donors.

    Returns
    -------
    DataFrame indexed by state with ``n_holdout``, ``mean_holdout``, ``sd_holdout``,
    ``q95_holdout``, ``q99_holdout`` and, for comparison, ``mean_in_sample``, ``sd_in_sample``.
    """
    dists: list[list[float]] = [[] for _ in range(n_states)]
    for _, test in folds:
        train = ~test
        for k in range(n_states):
            tr = train & (state_codes == k)
            te = test & (state_codes == k)
            if tr.sum() < 2 or not te.any():
                continue
            c = latent[tr].mean(axis=0)
            dists[k].extend(np.sqrt(((latent[te] - c) ** 2).sum(axis=1)).tolist())
    _, geom = state_geometry(latent, state_codes, n_states)
    rows = []
    for k, s_ in enumerate(states):
        d = np.asarray(dists[k], dtype=float)
        rows.append(
            {
                "state": str(s_),
                "n_holdout": int(d.size),
                "mean_holdout": float(d.mean()) if d.size else np.nan,
                "sd_holdout": float(d.std(ddof=1)) if d.size > 1 else np.nan,
                "q95_holdout": float(np.quantile(d, 0.95)) if d.size else np.nan,
                "q99_holdout": float(np.quantile(d, 0.99)) if d.size else np.nan,
                "mean_in_sample": float(geom[k, 1]),
                "sd_in_sample": float(geom[k, 2]),
            }
        )
    return pd.DataFrame(rows).set_index("state")


def calibrated_geometry(
    geometry: np.ndarray, calibration: pd.DataFrame | None, states: np.ndarray | list[str]
) -> np.ndarray:
    """
    Geometry array for :func:`reference_distance`: in-sample RMS radius in column 0 (the shift
    unit) and held-out mean / sd in columns 1-2 where available, in-sample otherwise.
    """
    out = np.array(geometry, dtype=np.float32, copy=True)
    if calibration is None:
        return out
    cal = calibration.reindex([str(s) for s in states])
    ok = cal["sd_holdout"].notna().to_numpy() & (cal["sd_holdout"].to_numpy() > 0)
    out[ok, 1] = cal.loc[ok, "mean_holdout"].to_numpy()
    out[ok, 2] = cal.loc[ok, "sd_holdout"].to_numpy()
    return out
