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


def test_anisotropic_states_and_a_withheld_state_between_two():
    from statetrace.datasets import simulate_transport

    ref, qry, truth = simulate_transport(
        n_ref_groups=6,
        n_query_groups=4,
        n_states=4,
        n_cells_per_group=400,
        n_latent=8,
        cohort_effect=0.0,
        donor_effect=0.1,
        novel_states=["S3"],
        novel_between={"S3": ("S0", "S1", 0.3)},
        state_anisotropy=3.0,
        anisotropic_dims=2,
        seed=4,
    )
    assert truth.loc["S3", "truth"] == "novel" and "S3" not in set(ref.obs["cell_state"])
    z, s = ref.obsm["X_latent"], ref.obs["cell_state"].astype(str).to_numpy()
    for k in ("S0", "S1", "S2"):
        ev = np.sort(np.linalg.eigvalsh(np.cov(z[s == k].T)))
        assert ev[-2] > 6 and ev[-3] < 2  # two stretched directions (sd 3), the rest about 1
    zq, sq = qry.obsm["X_latent"], qry.obs["cell_state"].astype(str).to_numpy()
    c0, c1 = z[s == "S0"].mean(axis=0), z[s == "S1"].mean(axis=0)
    gap = np.linalg.norm(zq[sq == "S3"].mean(axis=0) - (0.7 * c0 + 0.3 * c1))
    assert gap < 0.1 * np.linalg.norm(c1 - c0)
    with pytest.raises(ValueError, match="novel_states"):
        simulate_transport(n_states=4, novel_between={"S3": ("S0", "S1", 0.5)})
    with pytest.raises(ValueError, match="anchors"):
        simulate_transport(
            n_states=4, novel_states=["S2", "S3"], novel_between={"S3": ("S2", "S1", 0.5)}
        )


def test_chimera_state_shares_its_anchor_latent_and_differs_in_a_few_genes():
    ref, qry, truth = simulate_transport(
        n_ref_groups=4,
        n_query_groups=3,
        n_states=4,
        n_cells_per_group=300,
        n_latent=8,
        n_genes=100,
        cohort_effect=0.0,
        donor_effect=0.1,
        novel_states=["S3"],
        novel_chimera={"S3": ("S0", "S1", 10)},
        seed=2,
    )
    assert truth.loc["S3", "truth"] == "novel" and "S3" not in set(ref.obs["cell_state"])
    zq, sq = qry.obsm["X_latent"], qry.obs["cell_state"].astype(str).to_numpy()
    zr, sr = ref.obsm["X_latent"], ref.obs["cell_state"].astype(str).to_numpy()
    # the latent centroid is the anchor's (within the donor noise)
    assert np.linalg.norm(zq[sq == "S3"].mean(axis=0) - zr[sr == "S0"].mean(axis=0)) < 0.5
    # a handful of genes differ in mean expression from S0, the rest do not
    X = np.asarray(qry.X.todense())
    lib = X.sum(axis=1, keepdims=True)
    logn = np.log1p(X / lib * 1e4)
    diff = np.abs(logn[sq == "S3"].mean(axis=0) - logn[sq == "S0"].mean(axis=0))
    top = np.sort(diff)[::-1]
    assert top[0] > 2 * np.median(diff) and (diff > top[9] * 0.5).sum() >= 10
    with pytest.raises(ValueError, match="n_genes > 0"):
        simulate_transport(
            n_states=4, novel_states=["S3"], novel_chimera={"S3": ("S0", "S1", 5)}, n_genes=0
        )
    with pytest.raises(ValueError, match="must differ"):
        simulate_transport(
            n_states=4, novel_states=["S3"], novel_chimera={"S3": ("S0", "S0", 5)}, n_genes=50
        )
    with pytest.raises(ValueError, match="not both"):
        simulate_transport(
            n_states=4,
            novel_states=["S3"],
            novel_between={"S3": ("S0", "S1", 0.5)},
            novel_chimera={"S3": ("S0", "S1", 5)},
            n_genes=50,
        )
