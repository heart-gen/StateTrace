"""
Distribution-shift diagnostics in a shared latent space.

Three things can happen to a reference state in a query: it is *absent*, it is present but
*shifted* (its query cells sit away from the reference cells of that state), or query cells
form a *novel* population no reference state covers. Per cell, StateTrace reports a
reference-distance z-score (distance to the assigned state's reference centroid in units of
that state's reference spread) and a category; per state, a centroid shift in the same units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema


def state_centroids_and_spread(
    latent: np.ndarray, state_codes: np.ndarray, n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state centroid and RMS radius (root mean squared distance of cells to the centroid)."""
    d = latent.shape[1]
    centroids = np.zeros((n_states, d), dtype=np.float32)
    spread = np.full(n_states, np.nan, dtype=np.float32)
    for k in range(n_states):
        sel = state_codes == k
        if sel.any():
            c = latent[sel].mean(axis=0)
            centroids[k] = c
            spread[k] = np.sqrt(((latent[sel] - c) ** 2).sum(axis=1).mean())
    return centroids, spread


def reference_distance(
    latent: np.ndarray, assigned_codes: np.ndarray, centroids: np.ndarray, spread: np.ndarray
) -> np.ndarray:
    """Distance of each cell to its assigned state's reference centroid, divided by that state's spread."""
    assigned_codes = np.asarray(assigned_codes)
    out = np.full(len(assigned_codes), np.nan)
    ok = assigned_codes >= 0
    if ok.any():
        dist = np.sqrt(((latent[ok] - centroids[assigned_codes[ok]]) ** 2).sum(axis=1))
        sp = spread[assigned_codes[ok]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[ok] = np.where(sp > 0, dist / sp, np.nan)
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
    spread: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Distance between the query centroid of each assigned state and the reference centroid, in spread units."""
    out = np.full(n_states, np.nan)
    for k in range(n_states):
        sel = query_codes == k
        if sel.sum() >= 2 and spread[k] > 0:
            c = query_latent[sel].mean(axis=0)
            out[k] = np.sqrt(((c - centroids[k]) ** 2).sum()) / spread[k]
    return out


def shift_level(shift: np.ndarray | pd.Series, *, moderate: float, high: float) -> np.ndarray:
    s = np.asarray(shift, dtype=float)
    lvl = np.full(len(s), "unknown", dtype=object)
    lvl[np.isfinite(s) & (s < moderate)] = "low"
    lvl[np.isfinite(s) & (s >= moderate) & (s < high)] = "moderate"
    lvl[np.isfinite(s) & (s >= high)] = "high"
    assert set(lvl) <= set(schema.SHIFT_LEVELS)
    return lvl
