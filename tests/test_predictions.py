import numpy as np
import pandas as pd
import pytest

from statetrace import Predictions, schema
from statetrace.datasets import simulate_reference


def _proba(n, states, rng):
    P = rng.dirichlet(np.ones(len(states)) * 0.5, size=n)
    return pd.DataFrame(P, index=[f"c{i}" for i in range(n)], columns=states)


def test_from_arrays_with_proba():
    rng = np.random.default_rng(0)
    P = _proba(20, ["A", "B", "C"], rng)
    p = Predictions.from_arrays(P.index, P.idxmax(axis=1).to_numpy(), proba=P, backend="toy")
    assert p.states == ["A", "B", "C"] and len(p) == 20 and p.backend == "toy"
    np.testing.assert_allclose(p.confidence.to_numpy(), P.max(axis=1).to_numpy())
    m = p.margin().to_numpy()
    assert (m >= 0).all() and (m <= p.confidence.to_numpy() + 1e-12).all()
    assert (p.second_best() != p.labels).all()
    A = p.aligned_proba(["C", "A", "Z"])
    np.testing.assert_allclose(A[:, 0], P["C"]) and np.testing.assert_allclose(A[:, 2], 0)
    sub = p.subset(P.index[:5])
    assert len(sub) == 5 and sub.proba.shape == (5, 3)
    assert list(p.to_frame().columns) == [
        "label",
        "confidence",
        "margin",
        "second_best",
        "uncertainty",
        "support",
        "backend",
    ]


def test_labels_only_confidence_from_uncertainty():
    idx = [f"c{i}" for i in range(4)]
    p = Predictions.from_arrays(
        idx, ["A", "B", "A", "B"], uncertainty=[0.1, 0.9, 0.0, 0.5], backend="scpoli"
    )
    np.testing.assert_allclose(p.confidence.to_numpy(), [0.9, 0.1, 1.0, 0.5])
    A = p.aligned_proba(["A", "B"])
    assert A[0, 0] == pytest.approx(0.9) and A[1, 1] == pytest.approx(0.1)
    assert p.margin().isna().all() and p.second_best().isna().all()
    with pytest.warns(UserWarning, match="not in \\[0, 1\\]"):
        q = Predictions.from_arrays(idx, ["A", "B", "A", "B"], uncertainty=[1, 3, 0, 2])
    assert q.confidence.max() == 1.0 and q.confidence.min() == 0.0


def test_from_obs_and_dataframe_and_tool_conventions():
    a = simulate_reference(n_donors=2, n_states=3, n_cells_per_donor=10, seed=0)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(3), size=a.n_obs)
    a.obs["cell_type_pred"] = np.array(["S0", "S1", "S2"])[P.argmax(axis=1)]
    a.obs["cell_type_uncert"] = 1 - P.max(axis=1)
    a.obsm["soft"] = P
    a.uns["soft_columns"] = ["S0", "S1", "S2"]
    sp = Predictions.from_scpoli(a)
    assert (
        sp.backend == "scpoli"
        and sp.uncertainty is not None
        and np.allclose(sp.confidence, P.max(axis=1))
    )
    ob = Predictions.from_obs(a, "cell_type_pred", proba_key="soft")
    assert ob.proba.shape == (a.n_obs, 3) and list(ob.proba.columns) == ["S0", "S1", "S2"]
    sv = Predictions.from_scanvi(a, label_key="cell_type_pred", proba=ob.proba)
    assert sv.backend == "scanvi" and sv.proba is not None
    a.obs["conf_score"] = P.max(axis=1)
    a.obs["predicted_labels"] = a.obs["cell_type_pred"]
    ct = Predictions.from_celltypist(a)
    assert ct.backend == "celltypist" and np.allclose(ct.confidence, P.max(axis=1))
    df = pd.DataFrame(
        {
            "cell_type_pred_knn": a.obs["cell_type_pred"].to_numpy(),
            "cell_type_pred_knn_prob": P.max(axis=1),
            "mapping.score": 0.7,
        },
        index=a.obs_names,
    )
    sy = Predictions.from_symphony(df, mapping_score_col="mapping.score")
    assert sy.backend == "symphony" and (sy.support == 0.7).all()
    az = Predictions.from_azimuth(
        df.rename(
            columns={
                "cell_type_pred_knn": "predicted.celltype.l2",
                "cell_type_pred_knn_prob": "predicted.celltype.l2.score",
            }
        )
    )
    assert az.backend == "azimuth" and len(az) == a.n_obs
    with pytest.raises(KeyError):
        Predictions.from_obs(a, "nope")
    with pytest.raises(KeyError):
        Predictions.from_obs(a, "cell_type_pred", proba_key="missing")


def test_unknown_label_is_not_a_state():
    p = Predictions.from_arrays(["a", "b"], ["S0", schema.UNKNOWN_LABEL])
    assert p.states == ["S0"]
