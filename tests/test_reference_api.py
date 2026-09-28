import pytest

from atlasroute import AtlasReference, KNNBackend, Reference
from atlasroute.datasets import simulate_reference


def test_alias():
    assert Reference is AtlasReference


def test_keyword_only():
    a = simulate_reference(n_donors=2, n_cells_per_donor=10)
    with pytest.raises(TypeError):
        Reference(a, "donor_id", "cell_state")


def test_validate_before_fit_raises():
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    ref = Reference(a, donor_key="donor_id", state_key="cell_state", latent_key="X_latent")
    with pytest.raises(RuntimeError, match="fit"):
        ref.leave_one_donor_out()
    with pytest.raises(RuntimeError, match="fit"):
        ref.freeze("x")


def test_bad_method_and_option(fitted_small):
    with pytest.raises(ValueError, match="unknown validation method"):
        fitted_small.validate(method="bogus")
    with pytest.raises(TypeError, match="unknown LODOConfig"):
        fitted_small.leave_one_donor_out(bogus=1)


def test_refit_latent_reserved(fitted_small):
    with pytest.raises(NotImplementedError, match="refit_latent"):
        fitted_small.leave_one_donor_out(refit_latent=True)


def test_backend_instance_and_bad_backend():
    a = simulate_reference(n_donors=3, n_states=2, n_cells_per_donor=40, seed=2)
    ref = Reference(
        a,
        donor_key="donor_id",
        state_key="cell_state",
        latent_key="X_latent",
        backend=KNNBackend(n_neighbors=5),
        min_donors_per_state=1,
        min_cells_per_state=1,
    ).fit()
    assert ref.backend_.get_params()["n_neighbors"] == 5
    with pytest.raises(TypeError):
        Reference(a, donor_key="donor_id", state_key="cell_state", backend=object())


def test_pca_fallback_end_to_end():
    a = simulate_reference(n_donors=4, n_states=3, n_cells_per_donor=60, n_genes=40, seed=7)
    ref = Reference(
        a,
        donor_key="donor_id",
        state_key="cell_state",
        latent_key=None,
        n_pcs=8,
        min_donors_per_state=1,
        min_cells_per_state=1,
    ).fit()
    assert ref.latent_provenance_["source"] == "pca"
    assert ref.latent_.shape[1] == 8
    res = ref.leave_one_donor_out()
    assert res.latent["source"] == "pca" and res.latent["refit_latent"] is False


def test_fit_writes_uns_and_centroids(fitted_small):
    from atlasroute import schema

    assert schema.UNS_REFERENCE in fitted_small.adata.uns
    assert fitted_small.centroids_.shape == (4, fitted_small.latent_.shape[1])
    assert "4 states" in repr(fitted_small)


def test_stub_modules_raise():
    import atlasroute as ar
    from atlasroute import aggregation, calibration, novelty, reporting

    with pytest.raises(NotImplementedError):
        ar.transport(None, None, donor_key="d")
    with pytest.raises(NotImplementedError):
        ar.harmonize_features(None, None)
    with pytest.raises(NotImplementedError):
        calibration.Calibrator()
    with pytest.raises(NotImplementedError):
        novelty.novelty_scores()
    with pytest.raises(NotImplementedError):
        aggregation.pseudobulk()
    with pytest.raises(NotImplementedError):
        reporting.TransportReport()
