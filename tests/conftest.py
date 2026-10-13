from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (larger synthetic references)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


REF_KW = {
    "group_key": "donor_id",
    "state_key": "cell_state",
    "latent_key": "X_latent",
    "min_groups_per_state": 2,
    "min_cells_per_state": 10,
}


@pytest.fixture(scope="session")
def small_ref():
    """6 donors x 4 states x 100 cells, well separated: fast and easy."""
    from statetrace.datasets import simulate_reference

    return simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=100,
        donor_effect=0.5,
        state_separation=4.0,
        n_genes=50,
        seed=11,
    )


@pytest.fixture()
def fitted_small(small_ref):
    from statetrace import Reference

    return Reference(small_ref.copy(), celltype_key="cell_type", **REF_KW).fit()


@pytest.fixture(scope="session")
def transport_pair():
    """Reference and query cohorts with planted scenarios: S1 shifted, S3 minority, S4 absent."""
    from statetrace.datasets import simulate_transport

    return simulate_transport(
        n_ref_groups=8,
        n_query_groups=6,
        n_states=5,
        n_cells_per_group=150,
        donor_effect=0.5,
        state_separation=3.0,
        n_latent=10,
        n_genes=30,
        cohort_effect=0.3,
        shifted_states={"S1": 9.0},
        absent_states=["S4"],
        minority_states={"S3": 0.25},
        seed=5,
    )


@pytest.fixture()
def validated_ref(transport_pair):
    import warnings

    from statetrace import Reference

    ref_adata, _, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        ref.leave_one_group_out()
    return ref


def blurred_reference(n_donors: int = 8, seed: int = 3):
    """
    A reference whose counts separate every state, with a second latent ``X_blur`` in which
    S1 sits on S2 (a mapper that merged them): reproducibility differs by latent.
    """
    import numpy as np

    from statetrace.datasets import simulate_reference

    a = simulate_reference(
        n_donors=n_donors,
        n_states=4,
        n_cells_per_donor=200,
        state_separation=4.0,
        n_latent=10,
        n_genes=200,
        seed=seed,
    )
    s = a.obs["cell_state"].to_numpy()
    z = a.obsm["X_latent"]
    blur = z.copy()
    blur[s == "S1"] += z[s == "S2"].mean(axis=0) - z[s == "S1"].mean(axis=0)
    a.obsm["X_blur"] = blur.astype(np.float32)
    return a
