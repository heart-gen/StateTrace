import numpy as np
import pytest

from atlasroute import resolve_latent, schema
from atlasroute.datasets import simulate_reference


def test_obsm_path_returns_copy_and_provenance():
    a = simulate_reference(n_donors=2, n_cells_per_donor=30, n_latent=6)
    X, prov = resolve_latent(a, latent_key="X_latent")
    assert X.dtype == np.float32 and X.shape == (60, 6)
    assert prov["source"] == "obsm" and prov["key"] == "X_latent" and prov["n_dims"] == 6
    X[0, 0] = 1e6
    assert a.obsm["X_latent"][0, 0] != 1e6


def test_unknown_key_raises():
    a = simulate_reference(n_donors=2, n_cells_per_donor=10)
    with pytest.raises(KeyError, match="X_scVI"):
        resolve_latent(a, latent_key="X_scVI")


def test_pca_fallback_on_counts():
    a = simulate_reference(n_donors=3, n_cells_per_donor=40, n_genes=30, seed=4)
    X, prov = resolve_latent(a, latent_key=None, n_pcs=10)
    assert prov["source"] == "pca" and prov["n_dims"] == 10 and prov["hvg"] is None
    assert X.shape == (120, 10)
    assert schema.OBSM_PCA in a.obsm
    assert a.obsm[schema.OBSM_PCA].shape == (120, 10)


def test_pca_fallback_caps_components():
    a = simulate_reference(n_donors=2, n_cells_per_donor=20, n_genes=5)
    X, prov = resolve_latent(a, latent_key=None, n_pcs=50)
    assert prov["n_dims"] == 4 and X.shape[1] == 4


def test_pca_fallback_warns_on_non_counts():
    a = simulate_reference(n_donors=2, n_cells_per_donor=20, n_genes=10)
    a.X = np.log1p(a.X.toarray()) + 0.1
    with pytest.warns(UserWarning, match="raw counts"):
        resolve_latent(a, latent_key=None, n_pcs=3)


def test_pca_fallback_uses_hvg_when_many_genes():
    a = simulate_reference(n_donors=2, n_cells_per_donor=50, n_genes=2500, seed=1)
    X, prov = resolve_latent(a, latent_key=None, n_pcs=5)
    assert prov["hvg"] == 2000 and X.shape == (100, 5)
