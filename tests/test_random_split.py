import pytest

from atlasroute import Reference
from atlasroute.datasets import simulate_reference

KW = {
    "donor_key": "donor_id",
    "state_key": "cell_state",
    "latent_key": "X_latent",
    "min_donors_per_state": 2,
    "min_cells_per_state": 10,
}


def test_random_split_schema(fitted_small):
    res = fitted_small.random_split(n_splits=4)
    assert res.method == "random_split"
    assert res.overall["n_folds"] == 4
    assert res.cell_predictions.shape[0] == fitted_small.latent_.shape[0]
    assert fitted_small.validation_ is not res  # only LODO is stored on the reference
    assert fitted_small.validate(method="random_split", n_splits=3).overall["n_folds"] == 3


def test_too_many_splits_raises():
    a = simulate_reference(n_donors=2, n_states=3, n_cells_per_donor=6, seed=0)
    ref = Reference(a, **KW).fit()
    with pytest.raises(ValueError, match="n_splits"):
        ref.random_split(n_splits=50)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_split_is_at_least_as_optimistic_as_lodo(seed):
    a = simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=100,
        state_separation=2.0,
        donor_effect=1.5,
        seed=seed,
    )
    ref = Reference(a, **KW).fit()
    lodo = ref.leave_one_donor_out().overall["accuracy"]
    split = ref.random_split(n_splits=5).overall["accuracy"]
    assert split >= lodo - 0.02


def test_no_donor_effect_closes_the_gap():
    a = simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=100,
        state_separation=3.0,
        donor_effect=0.0,
        seed=4,
    )
    ref = Reference(a, **KW).fit()
    lodo = ref.leave_one_donor_out().overall["accuracy"]
    split = ref.random_split(n_splits=5).overall["accuracy"]
    assert abs(split - lodo) < 0.05
