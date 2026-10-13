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


def geometric_median(X: np.ndarray, *, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    The point minimising the summed Euclidean distance to the rows of ``X`` (Weiszfeld's
    iteration from the component-wise median). Rotation-invariant and, unlike the
    component-wise median, a single vector that is pulled little by one outlying row even
    when the number of rows is even. With one row it is that row.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 1:
        return X[0].copy()
    y = np.median(X, axis=0)
    for _ in range(max_iter):
        d = np.sqrt(((X - y) ** 2).sum(axis=1))
        if (d < 1e-12).any():  # at a data point: it is the minimiser if its "pull" is small
            w = np.where(d < 1e-12, 0.0, 1.0 / np.maximum(d, 1e-12))
        else:
            w = 1.0 / d
        if w.sum() == 0:
            break
        y_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.sqrt(((y_new - y) ** 2).sum()) < tol:
            y = y_new
            break
        y = y_new
    return y


def estimate_cohort_offset(
    query_latent: np.ndarray, query_codes: np.ndarray, centroids: np.ndarray, *, min_cells: int = 20
) -> np.ndarray:
    """Geometric median of (query state mean - reference state centroid) over states with enough cells."""
    disp = []
    for k in range(centroids.shape[0]):
        sel = query_codes == k
        if sel.sum() >= min_cells:
            disp.append(query_latent[sel].mean(axis=0) - centroids[k])
    if not disp:
        return np.zeros(centroids.shape[1], dtype=np.float32)
    return geometric_median(np.vstack(disp)).astype(np.float32)


def estimate_group_offsets(
    query_latent: np.ndarray,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    centroids: np.ndarray,
    *,
    min_cells: int = 20,
    min_states: int = 3,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-group (donor) latent offset: the geometric median (:func:`geometric_median`) over the
    group's states (with at least ``min_cells`` cells each) of the displacement of its state
    means from the reference centroids. A group carrying fewer than ``min_states`` such states gets
    ``fallback`` (the whole-cohort offset by default). Mapping tools that fit the query with
    surgery place each query donor with its own displacement, which a single cohort offset
    does not remove; a per-donor offset does, at the price of not seeing a shift that moves
    every state of a donor by the same vector, which is indistinguishable from a batch effect.

    Returns
    -------
    offsets, own
        ``(n_groups, n_dims)`` offsets and a boolean per group: ``True`` when it was estimated
        from the group's own cells.
    """
    group_codes = np.asarray(group_codes)
    state_codes = np.asarray(state_codes)
    n_groups = int(group_codes.max()) + 1 if len(group_codes) else 0
    if fallback is None:
        fallback = estimate_cohort_offset(query_latent, state_codes, centroids, min_cells=min_cells)
    offsets = np.tile(np.asarray(fallback, dtype=np.float32), (n_groups, 1))
    own = np.zeros(n_groups, dtype=bool)
    for g in range(n_groups):
        in_g = group_codes == g
        disp = []
        for k in range(centroids.shape[0]):
            sel = in_g & (state_codes == k)
            if sel.sum() >= min_cells:
                disp.append(query_latent[sel].mean(axis=0) - centroids[k])
        if len(disp) >= min_states:
            offsets[g] = geometric_median(np.vstack(disp))
            own[g] = True
    return offsets, own


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
OOD_SCORES = ("centroid", "knn", "joint")
EVIDENCE_COLUMNS = ("group", "true_state", "assigned_state", "distance", "knn_distance", "support")
_QUANTILE_COLUMNS = {0.95: "q95", 0.99: "q99"}


def neighbor_index(
    reference_latent: np.ndarray,
    reference_labels: np.ndarray,
    *,
    backend: Any = None,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> Any:
    """
    The neighbour index used for local support and the kNN distance: the reference's own kNN
    backend when it is one (its ``k`` and metric), else a
    :class:`~statetrace.backends.knn.KNNBackend` fitted on the reference latent.
    """
    from .backends.knn import KNNBackend

    if (
        backend is not None
        and hasattr(backend, "kneighbors")
        and hasattr(backend, "neighbor_codes")
    ):
        return backend
    return KNNBackend(n_neighbors=n_neighbors, metric=metric).fit(
        reference_latent, np.asarray(reference_labels).astype(str)
    )


def neighbor_evidence(
    index: Any, X: np.ndarray, assigned_codes: np.ndarray, states: np.ndarray | list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per cell of ``X``: local reference support (fraction of the k nearest reference cells in
    ``index`` that carry the assigned state) and the kNN distance (distance to the k-th
    nearest reference cell of any state, in the index's metric). Support is NaN for cells
    without an assigned state.
    """
    codes = np.asarray(assigned_codes)
    states = np.asarray(states).astype(str)
    classes = np.asarray(index.classes_).astype(str)
    lookup = {c: i for i, c in enumerate(classes)}
    col = np.array([lookup.get(s_, -1) for s_ in states])
    dist, idx = index.kneighbors(X)
    ncodes = index.neighbor_codes(idx)
    support = np.full(len(codes), np.nan)
    ok = codes >= 0
    ok[ok] &= col[codes[ok]] >= 0
    if ok.any():
        support[ok] = (ncodes[ok] == col[codes[ok]][:, None]).mean(axis=1)
    return support, np.asarray(dist[:, -1], dtype=float)


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
    Fraction of each query cell's nearest *reference* cells that carry the state the cell was
    assigned, in the coordinate system of ``query_latent``; see :func:`neighbor_evidence`.
    """
    index = neighbor_index(
        reference_latent, reference_labels, backend=backend, n_neighbors=n_neighbors, metric=metric
    )
    return neighbor_evidence(index, query_latent, assigned_codes, states)[0]


def holdout_evidence(
    latent: np.ndarray,
    state_codes: np.ndarray,
    group_labels: np.ndarray,
    folds: list[tuple[str, np.ndarray]],
    states: np.ndarray | list[str],
    *,
    index_template: Any = None,
    n_neighbors: int = 15,
    metric: str = "cosine",
    group_offset: bool = True,
    min_cells: int = 20,
    min_states: int = 3,
    assigned_codes: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Per held-out cell, the three novelty quantities measured against the *training* folds
    only: distance to the centroid of its assigned state, kNN distance to the k-th nearest
    training cell, and local support (fraction of the k nearest training cells carrying its
    assigned state). These carry the effect of an unseen group (donor), so they are the null
    against which query cells are judged.

    ``assigned_codes`` are the held-out *predictions* of grouped validation (``-1`` for
    abstentions); the true states are used when they are not given. Query cells are judged
    by their assigned state, and cells assigned to a state are the ones nearest its centroid,
    so a null built from true states would under-estimate each held-out donor's offset
    relative to the query and sit too tight; the null must be assembled the way the query is.
    With ``group_offset`` each held-out group's own latent offset
    (:func:`estimate_group_offsets`, assigned states, training centroids) is removed first,
    exactly as :func:`statetrace.validate_transport` removes a query donor's. One row per
    cell in the order of ``folds``.
    """
    states = np.asarray(states).astype(str)
    labels = states[state_codes]
    n_states = len(states)
    group_labels = np.asarray(group_labels).astype(str)
    assigned_all = np.asarray(state_codes) if assigned_codes is None else np.asarray(assigned_codes)
    rows: list[pd.DataFrame] = []
    for _, test in folds:
        train = ~test
        centroids, _ = state_geometry(latent[train], state_codes[train], n_states)
        present = np.bincount(state_codes[train], minlength=n_states) > 0
        codes = assigned_all[test]
        codes = np.where((codes >= 0) & present[np.clip(codes, 0, n_states - 1)], codes, -1)
        Zt = latent[test]
        if group_offset:
            gcodes = pd.factorize(group_labels[test], sort=True)[0]
            offsets, _ = estimate_group_offsets(
                Zt,
                gcodes,
                codes,
                centroids,
                min_cells=min_cells,
                min_states=min_states,
            )
            Zt = Zt - offsets[gcodes]
        dist = raw_distance(Zt, codes, centroids)
        template = index_template.clone() if hasattr(index_template, "clone") else None
        index = neighbor_index(
            latent[train],
            labels[train],
            backend=template.fit(latent[train], labels[train]) if template is not None else None,
            n_neighbors=n_neighbors,
            metric=metric,
        )
        support, knn = neighbor_evidence(index, Zt, codes, states)
        rows.append(
            pd.DataFrame(
                {
                    "group": np.asarray(group_labels)[test].astype(str),
                    "true_state": labels[test],
                    "assigned_state": np.where(
                        codes >= 0, states[np.clip(codes, 0, n_states - 1)], schema.UNKNOWN_LABEL
                    ),
                    "distance": dist,
                    "knn_distance": knn,
                    "support": support,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)[list(EVIDENCE_COLUMNS)]


def merge_holdout_evidence(
    group_evidence: pd.DataFrame,
    study_evidence: pd.DataFrame,
    states: np.ndarray | list[str],
    *,
    min_cells: int = 20,
    min_groups: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Per state, the study-holdout evidence when at least ``min_groups`` held-out groups carry
    ``min_cells`` or more cells assigned to the state with a finite centroid distance, else the
    group-holdout evidence (a state seen in one study only has no study-held-out centroid).

    Returns the merged evidence, with a ``unit`` column (``"study"`` / ``"group"``), and the
    unit per state. A cell's assigned state can differ between the two holdouts, so a cell
    may appear once in each part or in neither.
    """
    states = np.asarray(states).astype(str)
    e = study_evidence[np.isfinite(study_evidence["distance"].to_numpy(dtype=float))]
    n = e.groupby([e["assigned_state"].astype(str), e["group"].astype(str)]).size()
    n_groups = n[n >= min_cells].groupby(level=0).size()
    units = np.array(
        ["study" if int(n_groups.get(s_, 0)) >= min_groups else "group" for s_ in states],
        dtype=object,
    )
    by_study = set(states[units == "study"])
    a = study_evidence[study_evidence["assigned_state"].astype(str).isin(by_study)]
    b = group_evidence[~group_evidence["assigned_state"].astype(str).isin(by_study)]
    merged = pd.concat([a.assign(unit="study"), b.assign(unit="group")])
    return merged, units


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else np.nan


def holdout_distance_calibration(
    evidence: pd.DataFrame, geometry: np.ndarray, states: np.ndarray | list[str]
) -> pd.DataFrame:
    """
    Per-state statistics of the held-out evidence from :func:`holdout_evidence`: mean, sd and
    95th / 99th percentiles of the centroid distance and of the kNN distance; quantiles of the
    joint score (the larger of the two standardised distances); and the 5th / 10th percentiles
    and median of local support. ``mean_in_sample`` / ``sd_in_sample`` from ``geometry`` are
    kept for comparison. Unlike in-sample statistics these include the effect of an unseen
    group, so standardising query cells against them gives an out-of-reference rate close to
    the nominal one for genuinely new donors.
    """
    rows = []
    by = "assigned_state" if "assigned_state" in evidence else "true_state"
    for k, s_ in enumerate(np.asarray(states).astype(str)):
        e = evidence[evidence[by].astype(str) == s_]
        d = e["distance"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        kd = e["knn_distance"].to_numpy(dtype=float)
        kd = kd[np.isfinite(kd)]
        sup = e["support"].to_numpy(dtype=float)
        sup = sup[np.isfinite(sup)]
        mean, sd = (float(d.mean()), float(d.std(ddof=1))) if d.size > 1 else (np.nan, np.nan)
        kmean, ksd = (float(kd.mean()), float(kd.std(ddof=1))) if kd.size > 1 else (np.nan, np.nan)
        both = np.isfinite(e["distance"].to_numpy(dtype=float)) & np.isfinite(
            e["knn_distance"].to_numpy(dtype=float)
        )
        if both.any() and sd > 0 and ksd > 0:
            zc = (e.loc[both, "distance"].to_numpy(dtype=float) - mean) / sd
            zk = (e.loc[both, "knn_distance"].to_numpy(dtype=float) - kmean) / ksd
            joint = np.maximum(zc, zk)
        else:
            joint = np.array([])
        rows.append(
            {
                "state": s_,
                "n_holdout": int(d.size),
                "n_groups_holdout": int(e["group"].nunique()),
                "mean_holdout": mean,
                "sd_holdout": sd,
                "q95_holdout": _q(d, 0.95),
                "q99_holdout": _q(d, 0.99),
                "mean_in_sample": float(geometry[k, 1]),
                "sd_in_sample": float(geometry[k, 2]),
                "knn_mean_holdout": kmean,
                "knn_sd_holdout": ksd,
                "knn_q95_holdout": _q(kd, 0.95),
                "knn_q99_holdout": _q(kd, 0.99),
                "joint_q95_holdout": _q(joint, 0.95),
                "joint_q99_holdout": _q(joint, 0.99),
                "support_q05_holdout": _q(sup, 0.05),
                "support_q10_holdout": _q(sup, 0.10),
                "support_median_holdout": _q(sup, 0.5),
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


def _per_state(
    calibration: pd.DataFrame | None, states: np.ndarray | list[str], column: str
) -> np.ndarray:
    n = len(states)
    if calibration is None or column not in calibration:
        return np.full(n, np.nan)
    return calibration.reindex([str(s) for s in states])[column].to_numpy(dtype=float)


def knn_novelty(
    knn_distance: np.ndarray,
    assigned_codes: np.ndarray,
    calibration: pd.DataFrame | None,
    states: np.ndarray | list[str],
) -> np.ndarray:
    """kNN distance standardised against the held-out kNN distances of the assigned state (NaN without calibration)."""
    codes = np.asarray(assigned_codes)
    mean = _per_state(calibration, states, "knn_mean_holdout")
    sd = _per_state(calibration, states, "knn_sd_holdout")
    out = np.full(len(codes), np.nan)
    ok = codes >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.asarray(knn_distance, dtype=float)[ok] - mean[codes[ok]]) / sd[codes[ok]]
    out[ok] = np.where(sd[codes[ok]] > 0, z, np.nan)
    return out


def joint_novelty(novelty: np.ndarray, knn_nov: np.ndarray) -> np.ndarray:
    """The larger of the standardised centroid and kNN distances (NaN when either is missing)."""
    a = np.asarray(novelty, dtype=float)
    b = np.asarray(knn_nov, dtype=float)
    out = np.maximum(a, b)
    out[~(np.isfinite(a) & np.isfinite(b))] = np.nan
    return out


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
    score: str = "centroid",
    knn_distance: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Decide which cells lie outside their assigned state's reference distribution.

    ``score`` picks the evidence: ``"centroid"`` (distance to the state centroid), ``"knn"``
    (distance to the k-th nearest reference cell) or ``"joint"`` (the larger of the two
    standardised distances). ``rule="z"`` flags cells whose standardised score exceeds ``z``;
    ``rule="quantile"`` flags cells whose score exceeds the ``quantile`` (0.95 or 0.99) of the
    held-out reference cells of that state (raw distance for the centroid and kNN scores, the
    joint statistic itself for ``"joint"``), so a genuinely new donor of an intact state is
    flagged at about the nominal rate. Cells or states for which the chosen calibration is
    missing fall back to the centroid z rule, and so does the whole call when no calibration
    exists; the second return value records what was applied.
    """
    if rule not in OOD_RULES:
        raise ValueError(f"rule must be one of {OOD_RULES}, got {rule!r}")
    if score not in OOD_SCORES:
        raise ValueError(f"score must be one of {OOD_SCORES}, got {score!r}")
    nov = np.asarray(novelty, dtype=float)
    codes = np.asarray(assigned_codes)
    z_mask = np.nan_to_num(nov, nan=0.0) > z
    applied = {"score": "centroid", "rule": "z", "fallback": False}
    if score == "centroid" and rule == "z":
        return z_mask, applied
    if calibration is None or states is None:
        applied["fallback"] = True
        return z_mask, applied
    qname = _QUANTILE_COLUMNS.get(float(quantile))
    if qname is None:
        raise ValueError("quantile must be 0.95 or 0.99 (the levels stored with the reference)")
    ok = codes >= 0
    if score == "centroid":
        value = np.asarray(distance, dtype=float)
        thr_state = _per_state(calibration, states, f"{qname}_holdout")
    elif score == "knn":
        if knn_distance is None:
            applied["fallback"] = True
            return z_mask, applied
        knn_z = knn_novelty(knn_distance, codes, calibration, states)
        value = knn_z if rule == "z" else np.asarray(knn_distance, dtype=float)
        thr_state = (
            np.full(len(states), float(z))
            if rule == "z"
            else _per_state(calibration, states, f"knn_{qname}_holdout")
        )
    else:
        if knn_distance is None:
            applied["fallback"] = True
            return z_mask, applied
        value = joint_novelty(nov, knn_novelty(knn_distance, codes, calibration, states))
        thr_state = (
            np.full(len(states), float(z))
            if rule == "z"
            else _per_state(calibration, states, f"joint_{qname}_holdout")
        )
    if rule == "z" and score == "centroid":  # pragma: no cover - handled above
        thr_state = np.full(len(states), float(z))
    thr = np.full(len(codes), np.nan)
    thr[ok] = thr_state[codes[ok]]
    has = np.isfinite(thr) & np.isfinite(value)
    mask = np.where(has, np.nan_to_num(value, nan=0.0) > thr, z_mask)
    applied.update({"score": score, "rule": rule, "fallback": bool((~has & ok).any())})
    return mask, applied


def support_thresholds(
    calibration: pd.DataFrame | None,
    states: np.ndarray | list[str],
    *,
    quantile: float = 0.05,
    default: float | None = 0.5,
) -> np.ndarray:
    """
    Per-state minimum local support: the ``quantile`` (0.05 or 0.10) of the support that
    held-out reference cells of the state achieved; ``default`` where no calibration exists
    (NaN disables the rule for that state). :func:`statetrace.validate_transport` caps the
    result at its fixed ``min_support``, so calibration relaxes the rule for states whose own
    cells are mixed and never tightens it.
    """
    col = {0.05: "support_q05_holdout", 0.10: "support_q10_holdout"}.get(float(quantile))
    if col is None:
        raise ValueError("support quantile must be 0.05 or 0.10")
    thr = _per_state(calibration, states, col)
    fill = np.nan if default is None else float(default)
    return np.where(np.isfinite(thr), thr, fill)


def population_thresholds(
    evidence: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    states: np.ndarray | list[str],
    *,
    score: str,
    rule: str,
    z: float,
    quantile: float,
    min_cells: int = 20,
    min_groups: int = 3,
    population_quantile: float = 1.0,
    default: float = 0.3,
) -> np.ndarray:
    """
    Per-state out-of-reference fraction above which a query population is called novel:
    the ``population_quantile`` over held-out reference donors (with at least ``min_cells``
    cells of the state) of the fraction of their cells the active rule flags. With the
    default 1.0 the threshold is the largest fraction any reference donor produced.
    ``default`` where fewer than ``min_groups`` donors can be scored.
    """
    states = np.asarray(states).astype(str)
    out = np.full(len(states), float(default))
    if evidence is None or calibration is None or len(evidence) == 0:
        return out
    e = evidence
    codes, mask = evidence_ood_mask(
        e, calibration, states, score=score, rule=rule, z=z, quantile=quantile
    )
    frac = (
        pd.DataFrame({"group": e["group"].astype(str), "code": codes, "flag": mask})
        .groupby(["code", "group"])
        .agg(n=("flag", "size"), frac=("flag", "mean"))
        .reset_index()
    )
    frac = frac[frac["n"] >= min_cells]
    for k in range(len(states)):
        f = frac.loc[frac["code"] == k, "frac"].to_numpy(dtype=float)
        if f.size >= min_groups:
            out[k] = float(np.quantile(f, population_quantile))
    return out


def evidence_ood_mask(
    evidence: pd.DataFrame,
    calibration: pd.DataFrame,
    states: np.ndarray | list[str],
    *,
    score: str,
    rule: str,
    z: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assigned-state codes of held-out reference cells and the cells the active rule flags."""
    states = np.asarray(states).astype(str)
    by = "assigned_state" if "assigned_state" in evidence else "true_state"
    codes = pd.Index(states).get_indexer(evidence[by].astype(str))
    dist = evidence["distance"].to_numpy(dtype=float)
    mask, _ = out_of_reference_mask(
        dist,
        reference_distance_from_calibration(dist, codes, calibration, states),
        codes,
        rule=rule,
        z=z,
        quantile=quantile,
        calibration=calibration,
        states=states,
        score=score,
        knn_distance=evidence["knn_distance"].to_numpy(dtype=float),
    )
    return codes, mask


def group_evidence_table(
    groups: np.ndarray,
    assigned_codes: np.ndarray,
    support: np.ndarray,
    ood: np.ndarray,
    states: np.ndarray | list[str],
    *,
    support_thresholds: np.ndarray,
    uncertain_below: float,
    min_confidence: float,
    min_cells: int,
    max_ood_fraction: float,
) -> pd.DataFrame:
    """
    Per group (donor) x assigned state, from StateTrace's own cell evidence only (local
    support standing in for confidence, and the out-of-reference flag): whether the group
    *observes* the state (at least ``min_cells`` cells at or above the state's support
    threshold), whether it *recovers* it (observed, median support at least
    ``min_confidence``, out-of-reference fraction at most ``max_ood_fraction``) and the
    fraction of its cells that are uncertain (not out of reference, support below the
    state's threshold or below ``uncertain_below``). The same function measures held-out
    reference groups and query groups, so the two are comparable whichever mapper made the
    query labels. Cells with no assigned state (code ``-1``) are dropped.
    """
    states = np.asarray(states).astype(str)
    codes = np.asarray(assigned_codes)
    keep = codes >= 0
    codes = codes[keep]
    sup = np.nan_to_num(np.asarray(support, dtype=float)[keep], nan=1.0)
    ood = np.asarray(ood, dtype=bool)[keep]
    thr = np.nan_to_num(np.asarray(support_thresholds, dtype=float)[codes], nan=-1.0)
    low_support = sup < thr
    t = (
        pd.DataFrame(
            {
                "group": np.asarray(groups).astype(str)[keep],
                "state": states[codes],
                "supported_label": ~low_support,
                "support": sup,
                "ood": ood,
                "uncertain": ~ood & (low_support | (sup < uncertain_below)),
            }
        )
        .groupby(["group", "state"], sort=True)
        .agg(
            n_cells=("ood", "size"),
            n_supported=("supported_label", "sum"),
            median_support=("support", "median"),
            ood_fraction=("ood", "mean"),
            uncertain_fraction=("uncertain", "mean"),
        )
        .reset_index()
    )
    t["observed"] = t["n_supported"] >= min_cells
    t["recovered"] = (
        t["observed"]
        & (t["median_support"] >= min_confidence)
        & (t["ood_fraction"] <= max_ood_fraction)
    )
    return t


def holdout_group_table(
    evidence: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    states: np.ndarray | list[str],
    *,
    score: str,
    rule: str,
    z: float,
    quantile: float,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    :func:`group_evidence_table` for the held-out reference groups of grouped validation,
    each cell judged against the training groups only and flagged by the active
    out-of-reference rule. ``None`` without evidence or calibration.
    """
    if evidence is None or calibration is None or len(evidence) == 0:
        return None
    codes, ood = evidence_ood_mask(
        evidence, calibration, states, score=score, rule=rule, z=z, quantile=quantile
    )
    return group_evidence_table(
        evidence["group"].to_numpy(),
        codes,
        evidence["support"].to_numpy(dtype=float),
        ood,
        states,
        **kwargs,
    )


def caution_against_holdout(
    query_groups: pd.DataFrame | None,
    holdout_groups: pd.DataFrame | None,
    states: np.ndarray | list[str],
    *,
    alpha: float,
    margin: float,
    min_groups: int,
) -> pd.DataFrame:
    """
    Per state, whether the query's groups (donors) do measurably and materially worse than
    held-out reference groups, both measured by :func:`group_evidence_table`.

    - recovery: among groups observing the state, the query recovers it in a lower share of
      groups (one-sided Fisher exact test, ``p < alpha``), lower by more than ``margin``;
    - uncertainty: the per-group uncertain fractions of the observing query groups exceed
      the reference's (one-sided Mann-Whitney U, ``p < alpha``), with a median more than
      ``margin`` above.

    A state is judged this way (``caution_basis = "holdout"``) when at least ``min_groups``
    held-out reference groups observe it; otherwise ``"fixed"`` and the caller applies the
    fixed cut-offs. Both sides count groups, never cells, so the evidence grows with the
    number of donors, not with cell depth.
    """
    from scipy.stats import fisher_exact, mannwhitneyu

    states = np.asarray(states).astype(str)
    out = pd.DataFrame(
        {
            "caution_basis": "fixed",
            "ref_groups_observed": 0,
            "cmp_recovery_rate": np.nan,
            "ref_recovery_rate": np.nan,
            "recovery_p": np.nan,
            "recovery_worse": False,
            "cmp_uncertain_fraction": np.nan,
            "ref_uncertain_fraction": np.nan,
            "uncertain_p": np.nan,
            "uncertain_worse": False,
        },
        index=pd.Index(states, name="state"),
    )
    if query_groups is None or holdout_groups is None:
        return out
    q_obs = query_groups[query_groups["observed"]]
    r_obs = holdout_groups[holdout_groups["observed"]]
    for s in states:
        qs = q_obs[q_obs["state"] == s]
        rs = r_obs[r_obs["state"] == s]
        out.loc[s, "ref_groups_observed"] = len(rs)
        if len(rs) < min_groups:
            continue
        out.loc[s, "caution_basis"] = "holdout"
        k_r, n_r = int(rs["recovered"].sum()), len(rs)
        out.loc[s, "ref_recovery_rate"] = k_r / n_r
        out.loc[s, "ref_uncertain_fraction"] = float(rs["uncertain_fraction"].median())
        if not len(qs):
            continue
        k_q, n_q = int(qs["recovered"].sum()), len(qs)
        out.loc[s, "cmp_recovery_rate"] = k_q / n_q
        p_rec = float(fisher_exact([[k_q, n_q - k_q], [k_r, n_r - k_r]], alternative="less").pvalue)
        out.loc[s, "recovery_p"] = p_rec
        out.loc[s, "recovery_worse"] = bool(p_rec < alpha and k_r / n_r - k_q / n_q > margin)
        uq = qs["uncertain_fraction"].to_numpy(dtype=float)
        ur = rs["uncertain_fraction"].to_numpy(dtype=float)
        out.loc[s, "cmp_uncertain_fraction"] = float(np.median(uq))
        p_unc = (
            float(mannwhitneyu(uq, ur, alternative="greater").pvalue)
            if np.ptp(np.concatenate([uq, ur])) > 0
            else 1.0
        )
        out.loc[s, "uncertain_p"] = p_unc
        out.loc[s, "uncertain_worse"] = bool(
            p_unc < alpha and np.median(uq) - np.median(ur) > margin
        )
    return out


def reference_distance_from_calibration(
    distance: np.ndarray,
    assigned_codes: np.ndarray,
    calibration: pd.DataFrame | None,
    states: np.ndarray | list[str],
) -> np.ndarray:
    """Standardise raw centroid distances with the held-out mean / sd of the assigned state."""
    codes = np.asarray(assigned_codes)
    mean = _per_state(calibration, states, "mean_holdout")
    sd = _per_state(calibration, states, "sd_holdout")
    out = np.full(len(codes), np.nan)
    ok = codes >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.asarray(distance, dtype=float)[ok] - mean[codes[ok]]) / sd[codes[ok]]
    out[ok] = np.where(sd[codes[ok]] > 0, z, np.nan)
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
