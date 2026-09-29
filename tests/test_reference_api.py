import pytest
from conftest import REF_KW

from statetrace import KNNBackend, Reference, StateReference, schema
from statetrace.datasets import simulate_reference


def test_alias_and_keys():
    a = simulate_reference(n_donors=2, n_cells_per_donor=10)
    assert Reference is StateReference
    with pytest.raises(TypeError):
        Reference(a, "donor_id", "cell_state")
    with pytest.raises(TypeError, match="group_key"):
        Reference(a, state_key="cell_state")
    with pytest.raises(TypeError, match="not both"):
        Reference(a, group_key="donor_id", donor_key="batch", state_key="cell_state")
    r = Reference(a, donor_key="donor_id", state_key="cell_state")
    assert r.group_key == "donor_id" and r.donor_key == "donor_id"


def test_validate_before_fit_and_bad_options(fitted_small):
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    ref = Reference(a, **REF_KW)
    with pytest.raises(RuntimeError, match="fit"):
        ref.leave_one_group_out()
    with pytest.raises(RuntimeError, match="fit"):
        ref.freeze("x")
    with pytest.raises(ValueError, match="unknown validation method"):
        fitted_small.validate(method="bogus")
    with pytest.raises(TypeError, match="unknown LODOConfig"):
        fitted_small.leave_one_group_out(bogus=1)
    with pytest.raises(NotImplementedError, match="refit_latent"):
        fitted_small.leave_one_group_out(refit_latent=True)
    assert fitted_small.validate(method="leave_one_donor_out").method == "leave_one_group_out"


def test_backend_instance_and_bad_backend():
    a = simulate_reference(n_donors=3, n_states=2, n_cells_per_donor=40, seed=2)
    ref = Reference(a, backend=KNNBackend(n_neighbors=5), **REF_KW).fit()
    assert ref.backend_.get_params()["n_neighbors"] == 5 and ref.donors_ is ref.groups_
    with pytest.raises(TypeError):
        Reference(a, group_key="donor_id", state_key="cell_state", backend=object())


def test_pca_fallback_end_to_end():
    a = simulate_reference(n_donors=4, n_states=3, n_cells_per_donor=60, n_genes=40, seed=7)
    ref = Reference(
        a,
        group_key="donor_id",
        state_key="cell_state",
        latent_key=None,
        n_pcs=8,
        min_groups_per_state=1,
        min_cells_per_state=1,
    ).fit()
    assert ref.latent_provenance_["source"] == "pca" and ref.latent_.shape[1] == 8
    res = ref.leave_one_group_out()
    assert res.latent["source"] == "pca" and res.latent["refit_latent"] is False


def test_fit_writes_uns_and_geometry(fitted_small):
    assert schema.UNS_REFERENCE in fitted_small.adata.uns
    assert (
        fitted_small.centroids_.shape == (4, fitted_small.latent_.shape[1])
        and (fitted_small.spread_ > 0).all()
    )
    assert "4 states" in repr(fitted_small)
    assert list(fitted_small.state_table().columns[:5]) == [
        "n_cells",
        "n_groups",
        "n_studies",
        "eligible",
        "reason",
    ]
