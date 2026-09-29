import numpy as np
import pandas as pd
import pytest

from statetrace import LODOConfig, StabilityThresholds, metrics, schema


def _table():
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
    df = pd.DataFrame(rows, columns=["group", "true_state", "predicted_state"])
    df["probability"] = 0.9
    df["second_best"] = "S1"
    df["margin"] = 0.5
    df["support"] = 0.8
    df["abstained"] = df["predicted_state"] == schema.UNKNOWN_LABEL
    df["fold"] = df["group"]
    df.index = [f"c{i}" for i in range(len(df))]
    return df


def test_group_state_table_exact():
    dm = metrics.group_state_table(_table(), min_group_cells=2).set_index(["group", "state"])
    assert dm.loc[("d1", "S0"), "n_cells"] == 3 and dm.loc[("d1", "S0"), "recall"] == pytest.approx(
        2 / 3
    )
    assert dm.loc[("d1", "S0"), "n_predicted"] == 2 and dm.loc[
        ("d1", "S0"), "precision"
    ] == pytest.approx(1.0)
    assert dm.loc[("d1", "S1"), "n_predicted"] == 3 and dm.loc[
        ("d1", "S1"), "precision"
    ] == pytest.approx(2 / 3)
    assert dm.loc[("d2", "S0"), "recall"] == 0.0 and np.isnan(dm.loc[("d2", "S0"), "precision"])
    assert dm.loc[("d2", "S1"), "recall"] == pytest.approx(0.5) and dm.loc[
        ("d2", "S1"), "precision"
    ] == pytest.approx(1 / 3)
    assert dm["scored"].all()
    assert (
        not metrics.group_state_table(_table(), min_group_cells=3)
        .set_index(["group", "state"])
        .loc[("d2", "S0"), "scored"]
    )


def test_group_summary():
    ds = metrics.group_summary(_table()).set_index("group")
    assert ds.loc["d1", "accuracy"] == pytest.approx(4 / 5)
    assert ds.loc["d1", "balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)
    assert ds.loc["d2", "abstention_rate"] == pytest.approx(1 / 4) and ds.loc["d2", "n_states"] == 2


def test_state_table_and_pooled_metrics():
    df = _table()
    dm = metrics.group_state_table(df, min_group_cells=1)
    st = metrics.state_table(
        dm, df, states=["S0", "S1"], thresholds=StabilityThresholds(min_groups=2)
    )
    assert list(st.index) == ["S0", "S1"] and st.loc["S0", "n_groups_scored"] == 2
    assert (
        st.loc["S0", "median_group_recall"] == pytest.approx((2 / 3 + 0) / 2)
        and st.loc["S0", "min_group_recall"] == 0.0
    )
    assert st.loc["S0", "pooled_recall"] == pytest.approx(2 / 5) and st.loc[
        "S0", "precision"
    ] == pytest.approx(1.0)
    assert (
        st.loc["S1", "precision"] == pytest.approx(3 / 6) and st.loc["S0", "n_cells_reference"] == 5
    )
    assert st.loc["S0", "group_recovery_rate"] == pytest.approx(0.5)
    assert set(st["reproducibility"]) <= set(schema.STABILITY_CATEGORIES)
    dm1 = dm[dm["group"] == "d1"]
    st1 = metrics.state_table(
        dm1, df[df["group"] == "d1"], states=["S0", "S1"], thresholds=StabilityThresholds()
    )
    assert (
        np.isnan(st1.loc["S0", "recall_variance"])
        and (st1["reproducibility"] == "insufficient").all()
    )


@pytest.mark.parametrize(
    "n,med,mn,rr,expected",
    [
        (2, 0.9, 0.9, 1.0, "insufficient"),
        (5, 0.4, 0.1, 0.9, "fragile"),
        (5, 0.8, 0.7, 0.3, "fragile"),
        (5, 0.9, 0.8, 0.9, "stable"),
        (5, 0.9, 0.2, 0.9, "moderate"),
        (5, 0.6, 0.55, 0.85, "moderate"),
        (np.nan, 0.6, 0.55, 0.85, "insufficient"),
    ],
)
def test_reproducibility_branches(n, med, mn, rr, expected):
    row = {
        "n_groups_scored": n,
        "median_group_recall": med,
        "min_group_recall": mn,
        "group_recovery_rate": rr,
    }
    assert metrics.reproducibility_category(row, StabilityThresholds()) == expected


def test_confusion_and_overall():
    df = _table()
    cm = metrics.confusion_matrix(df, states=["S0", "S1"])
    assert list(cm.columns) == ["S0", "S1", schema.UNKNOWN_LABEL] and cm.sum().sum() == len(df)
    assert (
        schema.UNKNOWN_LABEL
        not in metrics.confusion_matrix(df[~df["abstained"]], states=["S0", "S1"]).columns
    )
    o = metrics.overall_metrics(df, metrics.group_summary(df))
    assert (
        o["accuracy"] == pytest.approx(5 / 9)
        and o["n_groups"] == 2
        and o["abstention_rate"] == pytest.approx(1 / 9)
    )


def test_config_roundtrip():
    cfg = LODOConfig(abstain_below=0.6, thresholds=StabilityThresholds(min_groups=4))
    assert LODOConfig.from_dict(cfg.to_dict()) == cfg
