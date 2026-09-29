import json

import numpy as np
import pytest

from statetrace.backends import BACKENDS, get_backend
from statetrace.backends.base import AnnDataBackend, MappingBackend, SupportsNeighborSupport
from statetrace.backends.knn import KNNBackend


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    centres = np.array([[6, 0, 0, 0, 0], [0, 6, 0, 0, 0], [-6, 0, 0, 0, 0]], dtype=float)
    X = np.vstack([c + rng.normal(0, 1, (40, 5)) for c in centres])
    y = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40, dtype=object)
    return X.astype(np.float32), y


def test_protocol_conformance():
    b = KNNBackend()
    assert isinstance(b, MappingBackend) and isinstance(b, SupportsNeighborSupport)
    assert not isinstance(b, AnnDataBackend)
    assert b.name == "knn"


def test_registry():
    assert set(BACKENDS) >= {"knn", "scanvi"}
    assert get_backend("knn", n_neighbors=7).n_neighbors == 7
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
    V = KNNBackend(n_neighbors=9).fit(X, y).neighbor_votes(X)
    assert V.shape == (120, 3) and (V.sum(axis=1) == 9).all() and V.dtype.kind == "i"


def test_cosine_matches_brute_force_ranking(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=3, metric="cosine").fit(X, y)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = Xn @ Xn.T
    for i in range(5):
        assert set(b._model.kneighbors(b._prep(X[i : i + 1]), return_distance=False)[0]) == set(
            np.argsort(-cos[i])[:3]
        )


def test_clone_and_params_json(toy):
    X, y = toy
    b = KNNBackend(n_neighbors=4, metric="euclidean", weights="distance").fit(X, y)
    c = b.clone()
    assert c.get_params() == b.get_params()
    with pytest.raises(RuntimeError):
        _ = c.classes_
    json.dumps(b.get_params())


def test_k_capped_and_bad_params(toy):
    X, y = toy
    assert KNNBackend(n_neighbors=500).fit(X[:10], y[:10]).predict_proba(X[:3]).shape == (3, 1)
    with pytest.raises(ValueError):
        KNNBackend(metric="manhattan")
    with pytest.raises(ValueError):
        KNNBackend(n_neighbors=0)
