"""
Two-sample tests of a reference state against the query cells assigned to it.

Whether two sets of profiles come from the same distribution is a question with dedicated,
principled answers: the graph cross-match test of scXMatch (Moeller et al., Bioinformatics
2026, doi:10.1093/bioinformatics/btag706) and the E-distance permutation test (Peidli et al.,
Nature Methods 2024). StateTrace does not compete with them; it runs them as optional
*engines* inside the frozen reference space and reports their evidence next to its own
donor-level pseudobulk shift, without letting either decide the transport status. What the
evidence means for transporting the state (reproducible, shifted but still the same state,
out of reference, or not supported for inference) is StateTrace's question.

Two levels are tested for every state, and they are never pooled:

``cell``
    The sampled reference cells of the state against the query cells assigned to it, in the
    frozen cell-level space (:func:`statetrace.expression.build_cell_space`). Sensitive,
    including to a change in spread that a pseudobulk mean cannot see, but cells of one donor
    are not independent replicates.
``donor``
    The held-out reference donor x state pseudobulk profiles (each reference donor
    standardised against the space rebuilt without it, the donors behind the pseudobulk
    shift's null) against the query donor x state profiles, standardised and
    offset-corrected the same way (:func:`statetrace.expression.query_profiles`). Both sides
    are out of sample, so they are exchangeable under no shift; in-sample reference profiles
    would sit artificially close to their own centroid. Respects the replication unit; power
    is bounded by the number of donors.

Every feature, scale and projection comes from the reference alone, and the states tested
are the frozen reference's states, never clusters inferred from the query (scXMatch's own
authors warn that testing de novo clusters is double dipping). Query cells enter a state by
the mapper's assignment, which can only pull them towards the reference (absorption), so the
tests are conservative rather than circular.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import schema

if TYPE_CHECKING:
    from .expression import ExpressionSpace

logger = logging.getLogger(__name__)

SCXMATCH_HINT = "conda install scxmatch -c conda-forge -c bioconda"
TEST_COLUMNS = (
    "state",
    "engine",
    "level",
    "subset",
    "statistic",
    "p_value",
    "z_score",
    "effect_strength_ratio",
    "coverage",
    "n_ref",
    "n_query",
    "n_ref_groups",
    "n_query_groups",
    "status",
    "reason",
)


@dataclass(frozen=True)
class TwoSampleConfig:
    """
    Which two-sample tests run and how.

    Parameters
    ----------
    engines
        Subset of :data:`statetrace.schema.TWO_SAMPLE_ENGINES`. ``"scxmatch"`` needs the
        scxmatch package, which installs only from conda (``conda install scxmatch -c
        conda-forge -c bioconda``; it depends on graph-tool); without it the rows are
        ``unsupported``.
    levels
        Subset of :data:`statetrace.schema.TWO_SAMPLE_LEVELS`.
    n_permutations
        Label permutations for the E-distance p-value.
    edistance_metric
        Cell-wise metric of the E-distance: ``"euclidean"`` (the energy distance, consistent
        against any difference in distribution, as in pertpy) or ``"sqeuclidean"`` (which
        reduces to the squared distance between means and cannot see a change in spread).
    subsample_to_smaller
        Subsample the larger side to the size of the smaller before the E-distance test, as
        in the E-distance benchmark of the scXMatch paper.
    k, metric, rank
        scXMatch's kNN graph size, distance and rank transform (its defaults: 100,
        ``"sqeuclidean"``, ``False``). A full graph is used when either side has at most
        ``k`` profiles.
    max_cells_per_side
        Seeded cap on the cells per side at cell level.
    min_cells, min_groups
        Smallest side tested at cell level (cells) and at donor level (donor profiles); a
        smaller side gives ``status="skipped"``.
    seed
        Seed for the cell caps, the subsampling and the permutations.
    """

    engines: tuple[str, ...] = ("edistance",)
    levels: tuple[str, ...] = ("cell", "donor")
    n_permutations: int = 1000
    edistance_metric: str = "euclidean"
    subsample_to_smaller: bool = True
    k: int = 100
    metric: str = "sqeuclidean"
    rank: bool = False
    max_cells_per_side: int = 500
    min_cells: int = 20
    min_groups: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "engines", tuple(self.engines))
        object.__setattr__(self, "levels", tuple(self.levels))
        bad = set(self.engines) - set(schema.TWO_SAMPLE_ENGINES)
        if bad or not self.engines:
            raise ValueError(f"engines must be a non-empty subset of {schema.TWO_SAMPLE_ENGINES}")
        bad = set(self.levels) - set(schema.TWO_SAMPLE_LEVELS)
        if bad or not self.levels:
            raise ValueError(f"levels must be a non-empty subset of {schema.TWO_SAMPLE_LEVELS}")
        if self.edistance_metric not in ("euclidean", "sqeuclidean"):
            raise ValueError("edistance_metric must be 'euclidean' or 'sqeuclidean'")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if self.k < 1 or self.max_cells_per_side < 2:
            raise ValueError("k must be >= 1 and max_cells_per_side >= 2")
        if self.min_cells < 2 or self.min_groups < 2:
            raise ValueError("min_cells and min_groups must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------------- E-distance
def _pairwise(X: np.ndarray, Y: np.ndarray, metric: str) -> np.ndarray:
    from scipy.spatial.distance import cdist

    return cdist(X, Y, metric=metric)


def energy_distance(X: np.ndarray, Y: np.ndarray, *, metric: str = "euclidean") -> float:
    """
    E-distance ``2 * mean d(X, Y) - mean d(X, X) - mean d(Y, Y)``.

    Within-group means run over all ``n^2`` pairs (the diagonal included), as in pertpy's
    ``Edistance``; with ``metric="euclidean"`` this is the energy distance of Szekely and
    Rizzo, zero only when the two distributions are equal.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return float(
        2.0 * _pairwise(X, Y, metric).mean()
        - _pairwise(X, X, metric).mean()
        - _pairwise(Y, Y, metric).mean()
    )


def edistance_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_permutations: int = 1000,
    metric: str = "euclidean",
    subsample_to_smaller: bool = True,
    seed: int = 0,
    batch: int = 256,
) -> dict[str, float]:
    """
    Permutation test of the E-distance between ``X`` and ``Y``.

    The pooled distance matrix is computed once and the group labels are permuted; each
    permutation needs only the quadratic form of its label vector. Returns ``statistic`` (the
    observed E-distance), ``p_value`` (``(1 + #{permuted >= observed}) / (1 + n_permutations)``),
    ``z_score`` (observed against the permutation mean and sd), ``n_x`` and ``n_y`` (after
    subsampling).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if subsample_to_smaller:
        n = min(len(X), len(Y))
        if len(X) > n:
            X = X[np.sort(rng.choice(len(X), size=n, replace=False))]
        if len(Y) > n:
            Y = Y[np.sort(rng.choice(len(Y), size=n, replace=False))]
    nx, ny = len(X), len(Y)
    if nx < 2 or ny < 2:
        raise ValueError("each side needs at least two profiles")
    P = np.vstack([X, Y])
    D = _pairwise(P, P, metric)
    total = D.sum()
    row = D.sum(axis=1)

    def stat(A: np.ndarray) -> np.ndarray:
        # A: (n, b) indicators of the X side; within_x = a'Da, between = a'r - a'Da
        q = np.einsum("ib,ib->b", A, D @ A)
        ar = row @ A
        within_x = q / nx**2
        between = (ar - q) / (nx * ny)
        within_y = (total - 2.0 * ar + q) / ny**2
        return 2.0 * between - within_x - within_y

    a0 = np.zeros((nx + ny, 1))
    a0[:nx] = 1.0
    observed = float(stat(a0)[0])
    perms = []
    done = 0
    while done < n_permutations:
        b = min(batch, n_permutations - done)
        A = np.zeros((nx + ny, b))
        for j in range(b):
            A[rng.permutation(nx + ny)[:nx], j] = 1.0
        perms.append(stat(A))
        done += b
    null = np.concatenate(perms)
    sd = float(null.std(ddof=1)) if null.size > 1 else np.nan
    return {
        "statistic": observed,
        "p_value": float((1.0 + (null >= observed).sum()) / (1.0 + null.size)),
        "z_score": float((observed - null.mean()) / sd) if sd and sd > 0 else np.nan,
        "n_x": nx,
        "n_y": ny,
    }


# ---------------------------------------------------------------------------- scXMatch
def _require_scxmatch():
    import importlib.util

    if importlib.util.find_spec("scxmatch") is None:
        raise ImportError(
            f"the scxmatch engine requires the scxmatch package. Install it with: {SCXMATCH_HINT} "
            "(conda only: it depends on graph-tool)."
        )
    import scxmatch

    return scxmatch


def crossmatch_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    k: int = 100,
    metric: str = "sqeuclidean",
    rank: bool = False,
) -> dict[str, float]:
    """
    Rosenbaum's cross-match test through scXMatch: ``X`` is the reference side, ``Y`` the
    test side.

    Returns scXMatch's ``p_value``, ``z_score`` (negative when the two sides match across less
    often than expected, i.e. differ), ``a1`` (observed cross-matches), ``E`` and ``var``
    (their null mean and variance), ``coverage`` (fraction of profiles matched) and
    ``effect_strength_ratio`` (``1 - min(a1, E) / E``). A full graph replaces the kNN graph
    when either side has at most ``k`` profiles.
    """
    import anndata as ad

    scx = _require_scxmatch()
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    obs = pd.DataFrame(
        {"side": pd.Categorical(["reference"] * len(X) + ["query"] * len(Y))},
        index=[f"p{i}" for i in range(len(X) + len(Y))],
    )
    adata = ad.AnnData(X=np.vstack([X, Y]), obs=obs)
    kk: int | None = int(k) if min(len(X), len(Y)) > k else None
    # scXMatch reports progress with print(); keep it out of the caller's stdout
    with contextlib.redirect_stdout(io.StringIO()):
        res = scx.test(
            adata=adata,
            group_by="side",
            test_group="query",
            reference="reference",
            metric=metric,
            rank=rank,
            k=kk,
        )
    out = {str(key): float(val) for key, val in dict(res).items()}
    out["k"] = float(kk) if kk is not None else np.nan
    return out


# ------------------------------------------------------------------------------ driver
def _cap(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(idx, size=n, replace=False)) if idx.size > n else idx


def _one(
    engine: str, R: np.ndarray, Q: np.ndarray, cfg: TwoSampleConfig, seed: int
) -> dict[str, Any]:
    try:
        if engine == "edistance":
            r = edistance_test(
                R,
                Q,
                n_permutations=cfg.n_permutations,
                metric=cfg.edistance_metric,
                subsample_to_smaller=cfg.subsample_to_smaller,
                seed=seed,
            )
            return {
                "statistic": r["statistic"],
                "p_value": r["p_value"],
                "z_score": r["z_score"],
                "status": "ok",
                "reason": "",
            }
        r = crossmatch_test(R, Q, k=cfg.k, metric=cfg.metric, rank=cfg.rank)
        return {
            "statistic": r.get("z_score", np.nan),
            "p_value": r.get("p_value", np.nan),
            "z_score": r.get("z_score", np.nan),
            "effect_strength_ratio": r.get("effect_strength_ratio", np.nan),
            "coverage": r.get("coverage", np.nan),
            "status": "ok",
            "reason": "",
        }
    except ImportError as err:
        return {"status": "unsupported", "reason": str(err)}
    except Exception as err:  # noqa: BLE001 - a failed test is a status, never a crash
        logger.warning("two-sample %s failed: %s", engine, err)
        return {"status": "error", "reason": f"{type(err).__name__}: {err}"}


def state_test_inputs(
    space: ExpressionSpace,
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    in_reference: np.ndarray,
    *,
    gene_index: np.ndarray | None = None,
    min_cells_per_group: int = 20,
    correct_cohort_offset: bool = True,
    counts: bool | None = None,
    config: TwoSampleConfig | None = None,
) -> list[dict[str, Any]]:
    """
    The two samples of every state x level x subset, exactly as :func:`state_tests` hands them
    to an engine, so external implementations can be run on identical inputs.

    Each item has ``state``, ``level``, ``subset``, ``n_ref``, ``n_query``, ``n_ref_groups``,
    ``n_query_groups``, ``reference`` and ``query`` (profiles x dimensions, ``None`` when the
    item cannot be tested), ``reference_groups`` and ``query_groups`` (the donor code of each
    row), and ``status`` / ``reason`` (``None`` when testable, else ``skipped`` or
    ``unsupported``). Arguments as for :func:`state_tests`.
    """
    from . import expression as _expr

    cfg = config or TwoSampleConfig()
    rng = np.random.default_rng(cfg.seed)
    states = np.asarray(space.states).astype(str)
    group_codes = np.asarray(group_codes)
    state_codes = np.asarray(state_codes)
    in_reference = np.asarray(in_reference, dtype=bool)
    idx = np.arange(space.n_genes) if gene_index is None else np.asarray(gene_index)
    items: list[dict[str, Any]] = []
    subsets = {
        "all": state_codes,
        "in_reference": np.where(in_reference, state_codes, -1),
    }

    def item(state, level, subset, R=None, Q=None, rg=None, qg=None, status=None, reason=""):
        n_ref = 0 if R is None else int(R.shape[0])
        n_q = 0 if Q is None else int(Q.shape[0])
        out = {
            "state": state,
            "level": level,
            "subset": subset,
            "n_ref": n_ref,
            "n_query": n_q,
            "n_ref_groups": int(np.unique(rg).size) if rg is not None else 0,
            "n_query_groups": int(np.unique(qg).size) if qg is not None else 0,
            "reference": R,
            "query": Q,
            "reference_groups": rg,
            "query_groups": qg,
            "status": status,
            "reason": reason,
        }
        if status is not None:
            out["reference"] = out["query"] = None
        items.append(out)

    if "cell" in cfg.levels:
        if not space.has_cell_space:
            for s in states:
                for sub in subsets:
                    item(
                        s,
                        "cell",
                        sub,
                        status="unsupported",
                        reason="the frozen reference carries no cell-level space "
                        "(format 3 or cell_sample_per_state=0); re-freeze to enable",
                    )
        else:
            ref_states = space.cell_rows["state"].to_numpy()
            ref_groups = space.cell_rows["group"].to_numpy()
            for sub, codes in subsets.items():
                # one projection per subset: the capped query cells of every state
                take = {
                    k: _cap(np.flatnonzero(codes == k), cfg.max_cells_per_side, rng)
                    for k in range(len(states))
                }
                cells = np.sort(np.concatenate([v for v in take.values() if v.size] or [[]]))
                cells = cells.astype(int)
                coords = (
                    _expr.project_cells(space, X[cells], gene_index=idx, counts=counts)
                    if cells.size
                    else np.zeros((0, space.cell_components.shape[0]))
                )
                where = {c: i for i, c in enumerate(cells)}
                for k, s in enumerate(states):
                    r_idx = _cap(np.flatnonzero(ref_states == k), cfg.max_cells_per_side, rng)
                    q_idx = take[k]
                    R = np.asarray(space.cell_coords[r_idx], dtype=np.float64)
                    Q = coords[[where[c] for c in q_idx]] if q_idx.size else coords[:0]
                    small = r_idx.size < cfg.min_cells or q_idx.size < cfg.min_cells
                    item(
                        s,
                        "cell",
                        sub,
                        R,
                        Q,
                        ref_groups[r_idx],
                        group_codes[q_idx],
                        status="skipped" if small else None,
                        reason=f"fewer than {cfg.min_cells} cells on a side" if small else "",
                    )

    if "donor" in cfg.levels:
        held = space.null_profiles
        for sub, codes in subsets.items():
            if held is None:
                for s in states:
                    item(
                        s,
                        "donor",
                        sub,
                        status="unsupported",
                        reason="the frozen reference carries no held-out donor profiles "
                        "(format 3); re-freeze to enable",
                    )
                continue
            q = _expr.query_profiles(
                space,
                X,
                group_codes,
                codes,
                gene_index=idx,
                min_cells=min_cells_per_group,
                correct_cohort_offset=correct_cohort_offset,
                counts=counts,
            )
            null_states = space.null["state"].to_numpy()
            null_groups = space.null["group"].to_numpy()
            q_groups = q["rows"]["group"].to_numpy()
            for k, s in enumerate(states):
                # reference donors held out of the space, as a query donor is: both sides
                # are out of sample, on the query's retained genes
                sel = null_states == k
                R = np.asarray(held[sel][:, idx], dtype=np.float64)
                Q = q["Z"][q["states"] == k]
                small = R.shape[0] < cfg.min_groups or Q.shape[0] < cfg.min_groups
                item(
                    s,
                    "donor",
                    sub,
                    R,
                    Q,
                    null_groups[sel],
                    q_groups[q["states"] == k],
                    status="skipped" if small else None,
                    reason=f"fewer than {cfg.min_groups} donor profiles on a side" if small else "",
                )
    return items


def state_tests(
    space: ExpressionSpace,
    X: Any,
    group_codes: np.ndarray,
    state_codes: np.ndarray,
    in_reference: np.ndarray,
    *,
    gene_index: np.ndarray | None = None,
    min_cells_per_group: int = 20,
    correct_cohort_offset: bool = True,
    counts: bool | None = None,
    config: TwoSampleConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Two-sample tests of every reference state against the query cells assigned to it.

    Parameters
    ----------
    space
        The frozen :class:`~statetrace.expression.ExpressionSpace`.
    X
        Query cells x genes aligned to ``space.genes[gene_index]`` (raw counts, as for
        :func:`~statetrace.expression.score_query`).
    group_codes, state_codes
        Query donor codes and assigned state codes (negative: excluded).
    in_reference
        Per-cell mask of cells not called out of reference; the ``in_reference`` subset tests
        only these.
    gene_index, min_cells_per_group, correct_cohort_offset, counts
        As for :func:`~statetrace.expression.score_query` (donor level).
    config
        :class:`TwoSampleConfig`.

    Returns
    -------
    table, provenance
        One row per state x engine x level x subset with the columns of
        :data:`TEST_COLUMNS`. ``statistic`` is the E-distance for ``edistance`` and the
        cross-match z-score for ``scxmatch``; ``z_score`` is on a common footing for both
        (E-distance against its permutation null, where larger means more different;
        scXMatch's own z, where *more negative* means more different); ``status`` follows
        :data:`statetrace.schema.TEST_STATUS` and ``reason`` says why a row is not ``ok``.
        Provenance records the configuration, the cell-space availability and the gene
        retention.
    """
    cfg = config or TwoSampleConfig()
    items = state_test_inputs(
        space,
        X,
        group_codes,
        state_codes,
        in_reference,
        gene_index=gene_index,
        min_cells_per_group=min_cells_per_group,
        correct_cohort_offset=correct_cohort_offset,
        counts=counts,
        config=cfg,
    )
    records: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        for e_i, engine in enumerate(cfg.engines):
            row = {
                k: it[k]
                for k in (
                    "state",
                    "level",
                    "subset",
                    "n_ref",
                    "n_query",
                    "n_ref_groups",
                    "n_query_groups",
                )
            }
            row["engine"] = engine
            if it["status"] is not None:
                row.update(status=it["status"], reason=it["reason"])
            else:
                row.update(
                    _one(
                        engine,
                        it["reference"],
                        it["query"],
                        cfg,
                        cfg.seed + 7919 * e_i + 104729 * i,
                    )
                )
            records.append(row)
    table = pd.DataFrame.from_records(records)
    for col in TEST_COLUMNS:
        if col not in table:
            table[col] = np.nan
    table = table[list(TEST_COLUMNS)]
    idx = np.arange(space.n_genes) if gene_index is None else np.asarray(gene_index)
    prov = {
        "config": cfg.to_dict(),
        "cell_space": bool(space.has_cell_space),
        "feature_retention": float(len(idx) / space.n_genes) if space.n_genes else np.nan,
        "state_definition": "frozen_reference",
        "query_in_state_definition": False,
    }
    return table, prov
