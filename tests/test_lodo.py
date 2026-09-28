import numpy as np
import pytest

from atlasroute import LODOConfig, Reference, ValidationResult, schema
from atlasroute.datasets import simulate_reference
from atlasroute.metrics import CELL_COLUMNS

STATE_COLS = [
    "n_donors_reference",
    "n_donors_scored",
    "n_cells_reference",
    "median_donor_recall",
    "min_donor_recall",
    "recall_variance",
    "recall_iqr",
    "donor_recovery_rate",
    "pooled_recall",
    "precision",
    "f1",
    "lodo_stability",
    "eligible",
]


def test_lodo_output_schema(fitted_small, small_ref):
    res = fitted_small.leave_one_donor_out()
    assert isinstance(res, ValidationResult)
    assert res.method == "leave_one_donor_out"
    assert list(res.state_metrics.columns) == STATE_COLS
    assert list(res.state_metrics.index) == list(fitted_small.states_)
    assert list(res.cell_predictions.columns) == list(CELL_COLUMNS)
    assert list(res.cell_predictions.index) == list(small_ref.obs_names)
    assert res.cell_predictions["fold"].nunique() == 6
    assert res.overall["n_folds"] == 6 and res.overall["n_donors"] == 6
    assert list(res.donor_metrics.columns) == [
        "donor",
        "state",
        "n_cells",
        "n_predicted",
        "recall",
        "precision",
        "f1",
        "scored",
    ]
    observed = small_ref.obs.groupby(["donor_id", "cell_state"], observed=True).size()
    assert len(res.donor_metrics) == len(observed)
    assert list(res.donor_summary.columns) == [
        "donor",
        "n_cells",
        "n_states",
        "accuracy",
        "balanced_accuracy",
        "abstention_rate",
    ]
    assert res.confusion_matrix.shape == (4, 4)
    assert res.confusion_matrix.to_numpy().sum() == small_ref.n_obs
    assert fitted_small.validation_ is res
    assert "fixed across folds" in res.summary()


def test_lodo_is_accurate_on_easy_data(fitted_small):
    res = fitted_small.leave_one_donor_out()
    assert res.overall["balanced_accuracy"] > 0.8
    assert (res.state_metrics["lodo_stability"] == "stable").all()


def test_cell_level_quantities_are_consistent(fitted_small):
    res = fitted_small.leave_one_donor_out()
    cp = res.cell_predictions
    assert ((cp["margin"] >= -1e-9) & (cp["margin"] <= cp["probability"] + 1e-9)).all()
    assert ((cp["probability"] > 0) & (cp["probability"] <= 1)).all()
    assert (cp["knn_support"] <= 15).all() and (cp["knn_support"] >= 1).all()
    assert (cp["second_best"] != cp["predicted_state"]).all()
    assert not cp["abstained"].any()


def test_abstention_creates_unknown_column():
    a = simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=100,
        state_separation=2.0,
        donor_effect=1.5,
        seed=11,
    )
    ref = Reference(
        a,
        donor_key="donor_id",
        state_key="cell_state",
        latent_key="X_latent",
        min_donors_per_state=2,
        min_cells_per_state=10,
    ).fit()
    res = ref.leave_one_donor_out(abstain_below=0.9)
    assert res.overall["abstention_rate"] > 0
    assert schema.UNKNOWN_LABEL in res.confusion_matrix.columns
    cp = res.cell_predictions
    assert (cp.loc[cp["abstained"], "predicted_state"] == schema.UNKNOWN_LABEL).all()
    assert (cp.loc[cp["abstained"], "probability"] < 0.9).all()
    assert (cp.loc[~cp["abstained"], "probability"] >= 0.9).all()
    plain = ref.leave_one_donor_out()
    assert plain.overall["accuracy"] >= res.overall["accuracy"]


def test_per_donor_confusion_and_n_jobs(fitted_small):
    res = fitted_small.leave_one_donor_out(per_donor_confusion=True, n_jobs=2)
    assert set(res.per_donor_confusion) == set(fitted_small.donors_)
    total = sum(cm.to_numpy().sum() for cm in res.per_donor_confusion.values())
    assert total == res.overall["n_cells"]
    res1 = fitted_small.leave_one_donor_out(n_jobs=1)
    assert res1.overall["accuracy"] == pytest.approx(res.overall["accuracy"])


def test_low_donor_support_state_is_flagged():
    a = simulate_reference(
        n_donors=6,
        n_states=4,
        n_cells_per_donor=120,
        state_separation=4.0,
        seed=8,
        donor_state_dropout={d: ["S3"] for d in ["D00", "D01", "D02", "D03"]},
    )
    ref = Reference(
        a,
        donor_key="donor_id",
        state_key="cell_state",
        latent_key="X_latent",
        min_donors_per_state=3,
        min_cells_per_state=10,
    )
    with pytest.warns(UserWarning, match="S3"):
        ref.fit()
    res = ref.leave_one_donor_out()
    assert res.state_metrics.loc["S3", "n_donors_reference"] == 2
    assert res.state_metrics.loc["S3", "lodo_stability"] == "insufficient_donors"
    assert not res.state_metrics.loc["S3", "eligible"]
    assert res.state_metrics.loc["S0", "lodo_stability"] == "stable"


def test_donor_effect_lowers_lodo_accuracy():
    def acc(effect):
        a = simulate_reference(
            n_donors=6,
            n_states=4,
            n_cells_per_donor=100,
            state_separation=2.0,
            donor_effect=effect,
            seed=21,
        )
        ref = Reference(
            a,
            donor_key="donor_id",
            state_key="cell_state",
            latent_key="X_latent",
            min_donors_per_state=2,
            min_cells_per_state=10,
        ).fit()
        return ref.leave_one_donor_out().overall["accuracy"]

    assert acc(2.0) < acc(0.2)


def test_state_specific_donor_interaction_is_worse():
    a = simulate_reference(
        n_donors=8,
        n_states=3,
        n_cells_per_donor=120,
        state_separation=2.0,
        donor_effect=0.3,
        donor_state_interaction=0.0,
        seed=5,
    )
    b = simulate_reference(
        n_donors=8,
        n_states=3,
        n_cells_per_donor=120,
        state_separation=2.0,
        donor_effect=0.3,
        donor_state_interaction=6.0,
        seed=5,
    )
    kw = {
        "donor_key": "donor_id",
        "state_key": "cell_state",
        "latent_key": "X_latent",
        "min_donors_per_state": 2,
        "min_cells_per_state": 10,
    }
    ra = Reference(a, **kw).fit().leave_one_donor_out()
    rb = Reference(b, **kw).fit().leave_one_donor_out()
    assert (
        rb.state_metrics["median_donor_recall"].mean()
        < ra.state_metrics["median_donor_recall"].mean()
    )


def test_to_obs_writes_predictions(fitted_small):
    res = fitted_small.leave_one_donor_out()
    a = fitted_small.adata
    res.to_obs(a)
    assert schema.OBS_LODO_STATE in a.obs
    assert (
        a.obs[schema.OBS_LODO_STATE].astype(str)
        == res.cell_predictions["predicted_state"].astype(str)
    ).all()
    assert schema.UNS_VALIDATION in a.uns


def test_result_dir_roundtrip(fitted_small, tmp_path):
    res = fitted_small.leave_one_donor_out(per_donor_confusion=True)
    d = res.to_dir(tmp_path / "val")
    back = ValidationResult.from_dir(d)
    assert back.method == res.method
    assert back.overall == res.overall
    assert list(back.state_metrics.index) == list(res.state_metrics.index)
    assert back.config == res.config
    assert set(back.per_donor_confusion) == set(res.per_donor_confusion)
    assert len(back.cell_predictions) == len(res.cell_predictions)


@pytest.mark.slow
def test_lodo_scale():
    a = simulate_reference(n_donors=20, n_states=8, n_cells_per_donor=500, seed=3)
    ref = Reference(a, donor_key="donor_id", state_key="cell_state", latent_key="X_latent").fit()
    res = ref.leave_one_donor_out(n_jobs=2)
    assert res.overall["n_folds"] == 20
    assert res.cell_predictions.shape[0] == a.n_obs
