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
        dist = np.sqrt(((latent[ok] - centroids[assigned_codes[ok]]) ** 2).sum(axis=1))
        mean = geometry[assigned_codes[ok], 1]
        sd = geometry[assigned_codes[ok], 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.where(sd > 0, (dist - mean) / sd, np.nan)
    return out


def cell_category(
    confidence: np.ndarray, novelty: np.ndarray, *, uncertain_below: float, ood_z: float
) -> np.ndarray:
    """``out_of_reference`` when novelty z exceeds ``ood_z``; else ``uncertain`` when confidence is low; else ``supported``."""
    conf = np.asarray(confidence, dtype=float)
    nov = np.asarray(novelty, dtype=float)
    cat = np.full(len(conf), "supported", dtype=object)
    cat[np.nan_to_num(conf, nan=1.0) < uncertain_below] = "uncertain"
    cat[np.nan_to_num(nov, nan=0.0) > ood_z] = "out_of_reference"
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
