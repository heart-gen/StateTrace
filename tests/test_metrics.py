import numpy as np
import pandas as pd
import pytest

from atlasroute import LODOConfig, StabilityThresholds, metrics, schema


def _table():
    # donor d1: S0 3 cells (2 correct), S1 2 cells (both correct);  d2: S0 2 cells (0 correct, predicted S1)
    rows = [
        ("d1", "S0", "S0"),
        ("d1", "S0", "S0"),
        ("d1", "S0", "S1"),
        ("d1", "S1", "S1"),
        ("d1", "S1", "S1"),
        ("d2", "S0", "S1"),
        ("d2", "S0", "S1"),
        ("d2", "S1", "S1"),
        ("d2", "S1", schema.UNKNOWN_LABEL),
    ]
    df = pd.DataFrame(rows, columns=["donor", "true_state", "predicted_state"])
    df["probability"] = 0.9
    df["second_best"] = "S1"
    df["margin"] = 0.5
    df["knn_support"] = 5
    df["abstained"] = df["predicted_state"] == schema.UNKNOWN_LABEL
    df["fold"] = df["donor"]
    df.index = [f"c{i}" for i in range(len(df))]
    return df


def test_donor_state_table_exact():
    dm = metrics.donor_state_table(_table(), min_donor_cells=2).set_index(["donor", "state"])
    assert dm.loc[("d1", "S0"), "n_cells"] == 3
    assert dm.loc[("d1", "S0"), "recall"] == pytest.approx(2 / 3)
    assert dm.loc[("d1", "S0"), "n_predicted"] == 2
    assert dm.loc[("d1", "S0"), "precision"] == pytest.approx(1.0)
    assert dm.loc[("d1", "S1"), "n_predicted"] == 3  # 2 true + 1 S0 mislabelled
    assert dm.loc[("d1", "S1"), "precision"] == pytest.approx(2 / 3)
    assert dm.loc[("d2", "S0"), "recall"] == 0.0
    assert np.isnan(dm.loc[("d2", "S0"), "precision"])  # never predicted in d2
    # abstained cell lowers recall but not precision
    assert dm.loc[("d2", "S1"), "recall"] == pytest.approx(0.5)
    assert dm.loc[("d2", "S1"), "precision"] == pytest.approx(
        1 / 3
    )  # 3 predicted S1 in d2, 1 correct
    assert dm["scored"].all()
    dm2 = metrics.donor_state_table(_table(), min_donor_cells=3)
    assert dm2.set_index(["donor", "state"]).loc[("d2", "S0"), "scored"] == False  # noqa: E712


def test_donor_summary():
    ds = metrics.donor_summary(_table()).set_index("donor")
    assert ds.loc["d1", "accuracy"] == pytest.approx(4 / 5)
    assert ds.loc["d1", "balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert ds.loc["d2", "abstention_rate"] == pytest.approx(1 / 4)
    assert ds.loc["d2", "n_states"] == 2


def test_state_table_and_pooled_metrics():
    df = _table()
    dm = metrics.donor_state_table(df, min_donor_cells=1)
    st = metrics.state_table(
        dm, df, states=["S0", "S1"], thresholds=StabilityThresholds(min_donors=2)
    )
    assert list(st.index) == ["S0", "S1"]
    assert st.loc["S0", "n_donors_scored"] == 2
    assert st.loc["S0", "median_donor_recall"] == pytest.approx((2 / 3 + 0) / 2)
    assert st.loc["S0", "min_donor_recall"] == 0.0
    assert st.loc["S0", "pooled_recall"] == pytest.approx(2 / 5)
    assert st.loc["S0", "precision"] == pytest.approx(1.0)
    assert st.loc["S1", "precision"] == pytest.approx(3 / 6)
    assert st.loc["S0", "n_cells_reference"] == 5
    assert st.loc["S0", "donor_recovery_rate"] == pytest.approx(0.5)
    assert set(st["lodo_stability"]) <= set(schema.STABILITY_CATEGORIES)
    # variance is NaN with one scored donor
    dm1 = dm[dm["donor"] == "d1"]
    st1 = metrics.state_table(
        dm1, df[df["donor"] == "d1"], states=["S0", "S1"], thresholds=StabilityThresholds()
    )
    assert np.isnan(st1.loc["S0", "recall_variance"])
    assert (st1["lodo_stability"] == "insufficient_donors").all()


@pytest.mark.parametrize(
    "n,med,mn,rr,expected",
    [
        (2, 0.9, 0.9, 1.0, "insufficient_donors"),
        (5, 0.4, 0.1, 0.9, "fragile"),
        (5, 0.8, 0.7, 0.3, "fragile"),
        (5, 0.9, 0.8, 0.9, "stable"),
        (5, 0.9, 0.2, 0.9, "moderate"),
        (5, 0.6, 0.55, 0.85, "moderate"),
        (np.nan, 0.6, 0.55, 0.85, "insufficient_donors"),
    ],
)
def test_stability_branches(n, med, mn, rr, expected):
    row = {
        "n_donors_scored": n,
        "median_donor_recall": med,
        "min_donor_recall": mn,
        "donor_recovery_rate": rr,
    }
    assert metrics.stability_category(row, StabilityThresholds()) == expected


def test_confusion_matrix_totals():
    df = _table()
    cm = metrics.confusion_matrix(df, states=["S0", "S1"])
    assert list(cm.columns) == ["S0", "S1", schema.UNKNOWN_LABEL]
    assert cm.sum().sum() == len(df)
    assert cm.loc["S0"].sum() == 5 and cm.loc["S1"].sum() == 4
    cm2 = metrics.confusion_matrix(df[~df["abstained"]], states=["S0", "S1"])
    assert schema.UNKNOWN_LABEL not in cm2.columns


def test_overall_metrics():
    df = _table()
    o = metrics.overall_metrics(df, metrics.donor_summary(df))
    assert o["accuracy"] == pytest.approx(5 / 9)
    assert o["n_cells"] == 9 and o["n_donors"] == 2 and o["n_states"] == 2 and o["n_folds"] == 2
    assert o["abstention_rate"] == pytest.approx(1 / 9)


def test_config_roundtrip():
    cfg = LODOConfig(abstain_below=0.6, thresholds=StabilityThresholds(min_donors=4))
    assert LODOConfig.from_dict(cfg.to_dict()) == cfg
