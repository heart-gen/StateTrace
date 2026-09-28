from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (larger synthetic references, minutes of CPU)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def small_ref():
    """6 donors x 4 states x 100 cells, well separated: fast and easy."""
    from atlasroute.datasets import simulate_reference

    return simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=100,
        donor_effect=0.5,
        state_separation=4.0,
        n_genes=50,
        seed=11,
    )


@pytest.fixture(scope="session")
def medium_ref():
    """8 donors x 5 states x 200 cells with moderate donor structure."""
    from atlasroute.datasets import simulate_reference

    return simulate_reference(
        n_donors=8,
        n_states=5,
        n_cells_per_donor=200,
        donor_effect=0.7,
        state_separation=3.0,
        n_genes=100,
        seed=1,
    )


@pytest.fixture()
def fitted_small(small_ref):
    from atlasroute import Reference

    return Reference(
        small_ref.copy(),
        donor_key="donor_id",
        state_key="cell_state",
        celltype_key="cell_type",
        latent_key="X_latent",
        min_donors_per_state=2,
        min_cells_per_state=10,
    ).fit()
