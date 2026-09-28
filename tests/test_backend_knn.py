import json

import numpy as np
import pytest

from atlasroute.backends import BACKENDS, get_backend
from atlasroute.backends.base import MappingBackend, SupportsNeighborSupport
from atlasroute.backends.knn import KNNBackend


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    centres = np.array([[6, 0, 0, 0, 0], [0, 6, 0, 0, 0], [-6, 0, 0, 0, 0]], dtype=float)
    X = np.vstack([c + rng.normal(0, 1, (40, 5)) for c in centres])
    y = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40, dtype=object)
    return X.astype(np.float32), y


def test_protocol_conformance():
    b = KNNBackend()
    assert isinstance(b, MappingBackend)
    assert isinstance(b, SupportsNeighborSupport)
    assert b.name == "knn"


def test_registry():
    assert "knn" in BACKENDS
    b = get_backend("knn", n_neighbors=7)
    assert isinstance(b, KNNBackend) and b.n_neighbors == 7
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("nope")


def test_predict_proba_rows_sum_to_one_and_align(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=5).fit(X, y)
    P = b.predict_proba(X)
    assert P.shape == (120, 3)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert list(b.classes_) == ["a", "b", "c"]
    assert (b.classes_[P.argmax(axis=1)] == y).mean() > 0.95


def test_neighbor_votes_sum_to_k(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=9).fit(X, y)
    V = b.neighbor_votes(X)
    assert V.shape == (120, 3)
    assert (V.sum(axis=1) == 9).all()
    assert V.dtype.kind == "i"


def test_cosine_matches_brute_force_ranking(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=3, metric="cosine").fit(X, y)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = Xn @ Xn.T
    for i in range(5):
        expected = set(np.argsort(-cos[i])[:3])
        got = set(b._model.kneighbors(b._prep(X[i : i + 1]), return_distance=False)[0])
        assert got == expected


def test_clone_and_params_json(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=4, metric="euclidean", weights="distance").fit(X, y)
    c = b.clone()
    assert c.get_params() == b.get_params()
    with pytest.raises(RuntimeError):
        _ = c.classes_
    json.dumps(b.get_params())


def test_k_capped_to_training_size(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=500).fit(X[:10], y[:10])
    assert b.predict_proba(X[:3]).shape == (3, 1)


def test_bad_params():
    with pytest.raises(ValueError):
        KNNBackend(metric="manhattan")
    with pytest.raises(ValueError):
        KNNBackend(n_neighbors=0)
