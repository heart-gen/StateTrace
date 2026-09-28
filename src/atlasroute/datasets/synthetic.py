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
    elif dropout > 0:
        allowed = rng.random(allowed.shape) >= float(dropout)
    # every donor keeps at least one state and every state keeps at least one donor
    for i in np.where(~allowed.any(axis=1))[0]:
        allowed[i, rng.integers(len(states))] = True
    for j in np.where(~allowed.any(axis=0))[0]:
        allowed[rng.integers(len(donors)), j] = True
    return allowed


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

    Returns
    -------
    AnnData
        ``obs`` has ``donor_id``, ``cell_state``, ``cell_type`` (``state // 2``) and
        ``batch``; ``uns["atlasroute_synthetic"]`` records the parameters.
    """
    import anndata as ad
    from scipy import sparse

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

    mu = rng.normal(0.0, state_separation, size=(n_states, n_latent))
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
        W = rng.normal(0.0, 0.5 / np.sqrt(n_latent), size=(n_genes, n_latent))
        b = rng.normal(1.0, 0.5, size=n_genes)
        rate = np.exp(np.clip(z @ W.T + b, -10.0, 8.0))
        X = sparse.csr_matrix(rng.poisson(rate).astype(np.float32))
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
    adata.uns["atlasroute_synthetic"] = params
    return adata
