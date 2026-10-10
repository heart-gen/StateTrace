"""
A seeded generative model of a multi-donor reference in latent space.

Every test and tutorial uses this generator so results are reproducible without
downloading data. Cells of state ``s`` from donor ``d`` are drawn as::

    z = mu_s + delta_d + gamma_ds + eps,   eps ~ N(0, I)

with state centroids ``mu_s ~ N(0, state_separation^2 I)``, a donor offset shared by
all cells of the donor ``delta_d ~ N(0, donor_effect^2 I)`` and an optional
donor x state interaction ``gamma_ds ~ N(0, donor_state_interaction^2 I)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def _allowed_matrix(
    rng: np.random.Generator,
    donors: list[str],
    states: list[str],
    dropout: float | dict[str, list[str]],
) -> np.ndarray:
    allowed = np.ones((len(donors), len(states)), dtype=bool)
    if isinstance(dropout, dict):
        for donor, dropped in dropout.items():
            if donor not in donors:
                raise KeyError(f"unknown donor {donor!r} in donor_state_dropout")
            for state in dropped:
                if state not in states:
                    raise KeyError(f"unknown state {state!r} in donor_state_dropout")
                allowed[donors.index(donor), states.index(state)] = False
        empty = [donors[i] for i in np.where(~allowed.any(axis=1))[0]]
        if empty:
            raise ValueError(f"donor_state_dropout leaves no states for donors {empty}")
        # an explicit table may remove a state from every donor (a withheld state)
        return allowed
    if dropout > 0:
        allowed = rng.random(allowed.shape) >= float(dropout)
        # random dropout: every donor keeps at least one state and every state keeps at least one donor
        for i in np.where(~allowed.any(axis=1))[0]:
            allowed[i, rng.integers(len(states))] = True
        for j in np.where(~allowed.any(axis=0))[0]:
            allowed[rng.integers(len(donors)), j] = True
    return allowed


def _counts_from_latent(
    rng: np.random.Generator, z: np.ndarray, W: np.ndarray, b: np.ndarray
) -> Any:
    """Poisson counts ``X = Poisson(exp(W z + b))`` as a float32 CSR matrix."""
    from scipy import sparse

    rate = np.exp(np.clip(z @ W.T + b, -10.0, 8.0))
    return sparse.csr_matrix(rng.poisson(rate).astype(np.float32))


def simulate_reference(
    *,
    n_donors: int = 8,
    n_states: int = 5,
    n_cells_per_donor: int = 200,
    donor_effect: float = 0.5,
    state_separation: float = 3.0,
    n_latent: int = 20,
    n_genes: int = 200,
    state_proportions: np.ndarray | list[float] | None = None,
    donor_state_dropout: float | dict[str, list[str]] = 0.0,
    donor_state_interaction: float = 0.0,
    seed: int = 0,
    latent_key: str = "X_latent",
    study: str = "ref",
    _shared: dict[str, Any] | None = None,
) -> AnnData:
    """
    Simulate a labelled multi-donor reference.

    Parameters
    ----------
    n_donors, n_states, n_cells_per_donor
        Sizes. Donors are named ``D00, D01, ...`` and states ``S0, S1, ...``.
    donor_effect
        Standard deviation of the per-donor latent offset (0 removes donor structure).
    state_separation
        Standard deviation of the state centroids; larger means easier classification.
    n_latent
        Latent dimensionality stored in ``obsm[latent_key]``.
    n_genes
        If > 0, Poisson counts are generated from random loadings so the PCA fallback
        can be exercised; ``X`` is a ``float32`` CSR matrix. With 0, ``X`` is empty.
    state_proportions
        Optional expected state frequencies (normalised internally).
    donor_state_dropout
        Either a probability that a (donor, state) pair is absent, or an explicit
        mapping ``{donor: [states to remove]}`` to create low-donor-support states.
    donor_state_interaction
        Standard deviation of donor x state offsets (a state that shifts differently
        per donor is harder to transport).
    seed
        Seed for :func:`numpy.random.default_rng`.
    latent_key
        ``obsm`` key that receives the latent coordinates.
    study
        Value written to ``obs["study"]``.

    Returns
    -------
    AnnData
        ``obs`` has ``donor_id``, ``cell_state``, ``cell_type`` (``state // 2``) and
        ``batch``; ``uns["statetrace_synthetic"]`` records the parameters.
    """
    import anndata as ad

    if n_donors < 1 or n_states < 1 or n_cells_per_donor < 1:
        raise ValueError("n_donors, n_states and n_cells_per_donor must be >= 1")
    rng = np.random.default_rng(seed)
    donors = [f"D{i:02d}" for i in range(n_donors)]
    states = [f"S{i}" for i in range(n_states)]

    if state_proportions is None:
        props = np.full(n_states, 1.0 / n_states)
    else:
        props = np.asarray(state_proportions, dtype=float)
        if props.shape != (n_states,) or (props < 0).any() or props.sum() <= 0:
            raise ValueError("state_proportions must be n_states non-negative values")
        props = props / props.sum()

    mu = (
        _shared["mu"]
        if _shared and "mu" in _shared
        else rng.normal(0.0, state_separation, size=(n_states, n_latent))
    )
    delta = rng.normal(0.0, donor_effect, size=(n_donors, n_latent))
    if donor_state_interaction > 0:
        gamma = rng.normal(0.0, donor_state_interaction, size=(n_donors, n_states, n_latent))
    else:
        gamma = np.zeros((n_donors, n_states, n_latent))
    allowed = _allowed_matrix(rng, donors, states, donor_state_dropout)

    donor_idx = np.repeat(np.arange(n_donors), n_cells_per_donor)
    state_idx = np.empty(donor_idx.shape[0], dtype=int)
    for d in range(n_donors):
        p = props * allowed[d]
        p = p / p.sum()
        sel = donor_idx == d
        state_idx[sel] = rng.choice(n_states, size=int(sel.sum()), p=p)

    z = mu[state_idx] + delta[donor_idx] + gamma[donor_idx, state_idx]
    z = z + rng.normal(0.0, 1.0, size=z.shape)
    n_cells = z.shape[0]

    if n_genes > 0:
        W = (
            _shared["W"]
            if _shared and "W" in _shared
            else rng.normal(0.0, 0.5 / np.sqrt(n_latent), size=(n_genes, n_latent))
        )
        b = _shared["b"] if _shared and "b" in _shared else rng.normal(1.0, 0.5, size=n_genes)
        X = _counts_from_latent(rng, z, W, b)
        var = pd.DataFrame(index=pd.Index([f"gene{i}" for i in range(n_genes)], name="gene"))
    else:
        X = np.empty((n_cells, 0), dtype=np.float32)
        var = pd.DataFrame(index=pd.Index([], name="gene"))

    obs = pd.DataFrame(
        {
            "donor_id": pd.Categorical([donors[i] for i in donor_idx], categories=donors),
            "cell_state": pd.Categorical([states[i] for i in state_idx], categories=states),
            "cell_type": pd.Categorical([f"T{i // 2}" for i in state_idx]),
            "batch": pd.Categorical(["B0"] * n_cells),
            "study": pd.Categorical([study] * n_cells),
        },
        index=pd.Index([f"cell{i}" for i in range(n_cells)], name="cell"),
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm[latent_key] = z.astype(np.float32)
    params: dict[str, Any] = {
        "n_donors": n_donors,
        "n_states": n_states,
        "n_cells_per_donor": n_cells_per_donor,
        "donor_effect": donor_effect,
        "state_separation": state_separation,
        "n_latent": n_latent,
        "n_genes": n_genes,
        "donor_state_interaction": donor_state_interaction,
        "seed": seed,
        "latent_key": latent_key,
    }
    adata.uns["statetrace_synthetic"] = params
    return adata


def simulate_transport(
    *,
    n_ref_groups: int = 8,
    n_query_groups: int = 6,
    n_states: int = 6,
    n_cells_per_group: int = 200,
    donor_effect: float = 0.5,
    state_separation: float = 3.0,
    n_latent: int = 20,
    n_genes: int = 0,
    cohort_effect: float = 0.5,
    shifted_states: dict[str, float] | None = None,
    spread_states: dict[str, float] | None = None,
    novel_states: list[str] | None = None,
    absent_states: list[str] | None = None,
    minority_states: dict[str, float] | None = None,
    abundance_change: dict[str, float] | None = None,
    seed: int = 0,
    latent_key: str = "X_latent",
) -> tuple[AnnData, AnnData, pd.DataFrame]:
    """
    Simulate a reference cohort and an independent query cohort sharing the same state
    centroids, with planted transport scenarios and a truth table.

    Parameters
    ----------
    n_ref_groups, n_query_groups
        Donors in each cohort.
    cohort_effect
        Standard deviation of a latent offset shared by every query cell (study effect).
    shifted_states
        ``{state: magnitude}``: the query version of the state is displaced by a random
        direction of this length (in latent units) on top of the cohort effect.
    spread_states
        ``{state: factor}``: the query cells of the state are scaled about their own mean by
        this factor in the latent (a change in spread with the mean kept), the case a
        donor-level pseudobulk mean is not built to see.
    novel_states
        States present only in the query (withheld from the reference).
    absent_states
        States present only in the reference.
    minority_states
        ``{state: fraction}``: the state is carried by only this fraction of reference donors
        (donor-dependent state).
    abundance_change
        ``{state: multiplier}`` applied to the state's expected frequency in the query.
    seed
        Seed; the reference and query use derived seeds.

    Returns
    -------
    reference, query, truth
        Two AnnData objects (``obs``: ``donor_id``, ``cell_state``, ``cell_type``, ``batch``,
        ``study``) and a table with the planted category per state
        (``stable``, ``shifted``, ``spread``, ``novel``, ``absent``, ``minority``), the shift
        magnitude and the spread factor.

    Notes
    -----
    With ``n_genes > 0`` the query counts are generated from the *final* query latent, so the
    cohort effect and every planted shift are present in expression as well as in
    ``obsm[latent_key]``. A mapper that only sees the counts therefore faces the planted
    scenario, and the frozen expression space can detect it.
    """
    rng = np.random.default_rng(seed)
    states = [f"S{i}" for i in range(n_states)]
    shifted_states = dict(shifted_states or {})
    spread_states = dict(spread_states or {})
    novel_states = list(novel_states or [])
    absent_states = list(absent_states or [])
    minority_states = dict(minority_states or {})
    abundance_change = dict(abundance_change or {})
    for s in [*shifted_states, *novel_states, *absent_states, *minority_states, *abundance_change]:
        if s not in states:
            raise KeyError(f"unknown state {s!r}")
    mu = rng.normal(0.0, state_separation, size=(n_states, n_latent))
    shared: dict[str, Any] = {"mu": mu}
    if n_genes > 0:
        shared["W"] = rng.normal(0.0, 0.5 / np.sqrt(n_latent), size=(n_genes, n_latent))
        shared["b"] = rng.normal(1.0, 0.5, size=n_genes)

    ref_donors = [f"D{i:02d}" for i in range(n_ref_groups)]
    ref_drop: dict[str, list[str]] = {d: list(novel_states) for d in ref_donors}
    for s, frac in minority_states.items():
        keep = max(1, int(round(frac * n_ref_groups)))
        for d in rng.permutation(ref_donors)[keep:]:
            ref_drop[str(d)].append(s)
    ref_drop = {d: v for d, v in ref_drop.items() if v}
    reference = simulate_reference(
        n_donors=n_ref_groups,
        n_states=n_states,
        n_cells_per_donor=n_cells_per_group,
        donor_effect=donor_effect,
        state_separation=state_separation,
        n_latent=n_latent,
        n_genes=n_genes,
        donor_state_dropout=ref_drop or 0.0,
        seed=seed + 1,
        latent_key=latent_key,
        study="reference",
        _shared=shared,
    )

    props = np.ones(n_states)
    for s, m in abundance_change.items():
        props[states.index(s)] *= float(m)
    q_drop = (
        {f"D{i:02d}": list(absent_states) for i in range(n_query_groups)} if absent_states else 0.0
    )
    query = simulate_reference(
        n_donors=n_query_groups,
        n_states=n_states,
        n_cells_per_donor=n_cells_per_group,
        donor_effect=donor_effect,
        state_separation=state_separation,
        n_latent=n_latent,
        n_genes=n_genes,
        state_proportions=props,
        donor_state_dropout=q_drop,
        seed=seed + 2,
        latent_key=latent_key,
        study="query",
        _shared=shared,
    )
    query.obs["donor_id"] = pd.Categorical("Q" + query.obs["donor_id"].astype(str).str[1:])
    Z = query.obsm[latent_key].astype(np.float64)
    Z += rng.normal(0.0, cohort_effect, size=n_latent)
    code = query.obs["cell_state"].cat.codes.to_numpy()
    for s, mag in shifted_states.items():
        direction = rng.normal(size=n_latent)
        direction *= float(mag) / np.linalg.norm(direction)
        Z[code == states.index(s)] += direction
    for s, factor in spread_states.items():
        sel = code == states.index(s)
        if sel.any():
            centre = Z[sel].mean(axis=0)
            Z[sel] = centre + float(factor) * (Z[sel] - centre)
    query.obsm[latent_key] = Z.astype(np.float32)
    if n_genes > 0:
        query.X = _counts_from_latent(np.random.default_rng(seed + 3), Z, shared["W"], shared["b"])
    # drop the reference-only categories from the query's categorical so tables stay honest
    query.obs["cell_state"] = query.obs["cell_state"].cat.remove_unused_categories()

    truth = pd.DataFrame({"state": states, "truth": "stable"}).set_index("state")
    for s in minority_states:
        truth.loc[s, "truth"] = "minority"
    for s in shifted_states:
        truth.loc[s, "truth"] = "shifted"
    for s in spread_states:
        truth.loc[s, "truth"] = "spread"
    for s in absent_states:
        truth.loc[s, "truth"] = "absent"
    for s in novel_states:
        truth.loc[s, "truth"] = "novel"
    truth["shift_magnitude"] = [shifted_states.get(s, 0.0) for s in states]
    truth["spread_factor"] = [spread_states.get(s, 1.0) for s in states]
    truth["ref_donor_fraction"] = [
        0.0 if s in novel_states else minority_states.get(s, 1.0) for s in states
    ]
    return reference, query, truth
