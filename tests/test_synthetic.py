import numpy as np
import pytest

from statetrace.datasets import simulate_reference, simulate_transport


def test_shapes_and_dtypes():
    a = simulate_reference(
        n_donors=3, n_states=4, n_cells_per_donor=50, n_latent=7, n_genes=20, seed=3
    )
    assert a.n_obs == 150 and a.n_vars == 20
    assert a.obsm["X_latent"].shape == (150, 7) and a.obsm["X_latent"].dtype == np.float32
    assert (
        str(a.obs["donor_id"].dtype) == "category" and str(a.obs["cell_state"].dtype) == "category"
    )
    assert list(a.obs["donor_id"].cat.categories) == ["D00", "D01", "D02"]
    assert (a.obs["study"] == "ref").all()
    assert a.uns["statetrace_synthetic"]["seed"] == 3


def test_seed_determinism():
    a = simulate_reference(seed=5, n_donors=3, n_cells_per_donor=30)
    b = simulate_reference(seed=5, n_donors=3, n_cells_per_donor=30)
    np.testing.assert_array_equal(a.obsm["X_latent"], b.obsm["X_latent"])
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
    assert tab.loc["D00", "S2"] == 0 and tab.loc["D01", "S2"] == 0 and tab.loc["D02", "S2"] > 0


def test_random_dropout_keeps_every_state_and_donor():
    a = simulate_reference(
        n_donors=5, n_states=6, n_cells_per_donor=40, donor_state_dropout=0.5, seed=9
    )
    assert a.obs["cell_state"].nunique() == 6 and a.obs["donor_id"].nunique() == 5


def test_zero_genes_gives_empty_x():
    a = simulate_reference(n_genes=0, n_donors=2, n_cells_per_donor=10)
    assert a.n_vars == 0 and a.X.shape == (20, 0)


def test_bad_arguments():
    with pytest.raises(ValueError):
        simulate_reference(n_donors=0)
    with pytest.raises(ValueError):
        simulate_reference(n_states=3, state_proportions=[1, 1])
    with pytest.raises(KeyError):
        simulate_reference(donor_state_dropout={"D99": ["S0"]})
    with pytest.raises(ValueError, match="no states"):
        simulate_reference(n_donors=2, n_states=2, donor_state_dropout={"D00": ["S0", "S1"]})


def test_simulate_transport_scenarios(transport_pair):
    ref, qry, truth = transport_pair
    assert (
        truth.loc["S1", "truth"] == "shifted"
        and truth.loc["S4", "truth"] == "absent"
        and truth.loc["S3", "truth"] == "minority"
    )
    assert "S4" not in qry.obs["cell_state"].cat.categories
    assert qry.obs["donor_id"].str.startswith("Q").all()
    assert set(ref.obs["study"]) == {"reference"} and set(qry.obs["study"]) == {"query"}
    n_ref_donors_s3 = ref.obs.loc[ref.obs["cell_state"] == "S3", "donor_id"].nunique()
    assert n_ref_donors_s3 == 2
    # shared centroids: the unshifted states sit at the same place in both cohorts (up to cohort/donor noise)
    for s in ("S0", "S2"):
        cr = ref.obsm["X_latent"][(ref.obs["cell_state"] == s).to_numpy()].mean(axis=0)
        cq = qry.obsm["X_latent"][(qry.obs["cell_state"] == s).to_numpy()].mean(axis=0)
        assert np.linalg.norm(cr - cq) < 3.0
    cr = ref.obsm["X_latent"][(ref.obs["cell_state"] == "S1").to_numpy()].mean(axis=0)
    cq = qry.obsm["X_latent"][(qry.obs["cell_state"] == "S1").to_numpy()].mean(axis=0)
    assert np.linalg.norm(cr - cq) > 6.0


def test_simulate_transport_novel_state():
    ref, qry, truth = simulate_transport(
        n_ref_groups=4,
        n_query_groups=3,
        n_states=4,
        n_cells_per_group=60,
        novel_states=["S3"],
        seed=1,
    )
    assert (ref.obs["cell_state"] != "S3").all() and (qry.obs["cell_state"] == "S3").any()
    assert truth.loc["S3", "truth"] == "novel"
    with pytest.raises(KeyError):
        simulate_transport(n_states=3, novel_states=["S9"])
