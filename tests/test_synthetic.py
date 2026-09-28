import numpy as np
import pytest

from atlasroute.datasets import simulate_reference


def test_shapes_and_dtypes():
    a = simulate_reference(
        n_donors=3, n_states=4, n_cells_per_donor=50, n_latent=7, n_genes=20, seed=3
    )
    assert a.n_obs == 150
    assert a.n_vars == 20
    assert a.obsm["X_latent"].shape == (150, 7)
    assert a.obsm["X_latent"].dtype == np.float32
    assert str(a.obs["donor_id"].dtype) == "category"
    assert str(a.obs["cell_state"].dtype) == "category"
    assert list(a.obs["donor_id"].cat.categories) == ["D00", "D01", "D02"]
    assert a.uns["atlasroute_synthetic"]["seed"] == 3


def test_seed_determinism():
    a = simulate_reference(seed=5, n_donors=3, n_cells_per_donor=30)
    b = simulate_reference(seed=5, n_donors=3, n_cells_per_donor=30)
    np.testing.assert_array_equal(a.obsm["X_latent"], b.obsm["X_latent"])
    assert (a.obs["cell_state"] == b.obs["cell_state"]).all()
    c = simulate_reference(seed=6, n_donors=3, n_cells_per_donor=30)
    assert not np.array_equal(a.obsm["X_latent"], c.obsm["X_latent"])


def test_explicit_dropout_removes_states_from_donors():
    a = simulate_reference(
        n_donors=4,
        n_states=3,
        n_cells_per_donor=60,
        donor_state_dropout={"D00": ["S2"], "D01": ["S2"]},
        seed=0,
    )
    tab = a.obs.groupby(["donor_id", "cell_state"], observed=False).size().unstack()
    assert tab.loc["D00", "S2"] == 0
    assert tab.loc["D01", "S2"] == 0
    assert tab.loc["D02", "S2"] > 0


def test_random_dropout_keeps_every_state_and_donor():
    a = simulate_reference(
        n_donors=5, n_states=6, n_cells_per_donor=40, donor_state_dropout=0.5, seed=9
    )
    assert a.obs["cell_state"].nunique() == 6
    assert a.obs["donor_id"].nunique() == 5


def test_zero_genes_gives_empty_x():
    a = simulate_reference(n_genes=0, n_donors=2, n_cells_per_donor=10)
    assert a.n_vars == 0
    assert a.X.shape == (20, 0)


def test_bad_arguments():
    with pytest.raises(ValueError):
        simulate_reference(n_donors=0)
    with pytest.raises(ValueError):
        simulate_reference(n_states=3, state_proportions=[1, 1])
    with pytest.raises(KeyError):
        simulate_reference(donor_state_dropout={"D99": ["S0"]})
