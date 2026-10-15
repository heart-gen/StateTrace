import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from conftest import REF_KW, blurred_reference
from sklearn.linear_model import LogisticRegression

from statetrace import (
    CallableBackend,
    Predictions,
    Reference,
    ValidationResult,
    schema,
    summarize_holdout,
)
from statetrace.datasets import simulate_reference, simulate_transport
from statetrace.metrics import CELL_COLUMNS

STATE_COLS = [
    "n_groups_reference",
    "n_groups_scored",
    "n_cells_reference",
    "median_group_recall",
    "min_group_recall",
    "recall_variance",
    "recall_iqr",
    "group_recovery_rate",
    "pooled_recall",
    "precision",
    "f1",
    "reproducibility",
    "eligible",
]


def test_logo_output_schema(fitted_small, small_ref):
    res = fitted_small.leave_one_group_out(reproducibility_space="latent")
    assert (
        isinstance(res, ValidationResult)
        and res.method == "leave_one_group_out"
        and res.group_key == "donor_id"
    )
    assert list(res.state_metrics.columns) == STATE_COLS and list(res.state_metrics.index) == list(
        fitted_small.states_
    )
    assert list(res.cell_predictions.columns) == list(CELL_COLUMNS)
    assert list(res.cell_predictions.index) == list(small_ref.obs_names)
    assert res.overall["n_folds"] == 6 and res.overall["n_groups"] == 6
    assert res.donor_metrics is res.group_metrics and res.donor_summary is res.group_summary
    observed = small_ref.obs.groupby(["donor_id", "cell_state"], observed=True).size()
    assert len(res.group_metrics) == len(observed)
    assert res.confusion_matrix.to_numpy().sum() == small_ref.n_obs
    assert fitted_small.validation_ is res and "fixed across folds" in res.summary()
    assert (
        res.overall["balanced_accuracy"] > 0.8
        and (res.state_metrics["reproducibility"] == "stable").all()
    )
    assert fitted_small.leave_one_donor_out().overall == res.overall


def test_cell_level_quantities(fitted_small):
    cp = fitted_small.leave_one_group_out().cell_predictions
    assert ((cp["margin"] >= -1e-9) & (cp["margin"] <= cp["probability"] + 1e-9)).all()
    assert ((cp["support"] > 0) & (cp["support"] <= 1)).all() and (
        cp["second_best"] != cp["predicted_state"]
    ).all()
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
    ref = Reference(a, **REF_KW).fit()
    res = ref.leave_one_group_out(abstain_below=0.9)
    assert (
        res.overall["abstention_rate"] > 0 and schema.UNKNOWN_LABEL in res.confusion_matrix.columns
    )
    cp = res.cell_predictions
    assert (cp.loc[cp["abstained"], "probability"] < 0.9).all() and (
        cp.loc[~cp["abstained"], "probability"] >= 0.9
    ).all()


def test_per_group_confusion_and_n_jobs(fitted_small):
    res = fitted_small.leave_one_group_out(per_group_confusion=True, n_jobs=2)
    assert set(res.per_group_confusion) == set(fitted_small.groups_)
    assert (
        sum(cm.to_numpy().sum() for cm in res.per_group_confusion.values())
        == res.overall["n_cells"]
    )
    assert fitted_small.leave_one_group_out(n_jobs=1).overall["accuracy"] == pytest.approx(
        res.overall["accuracy"]
    )


def test_low_group_support_state_is_flagged():
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
        group_key="donor_id",
        state_key="cell_state",
        latent_key="X_latent",
        min_groups_per_state=3,
        min_cells_per_state=10,
    )
    with pytest.warns(UserWarning, match="S3"):
        ref.fit()
    res = ref.leave_one_group_out()
    assert (
        res.state_metrics.loc["S3", "n_groups_reference"] == 2
        and res.state_metrics.loc["S3", "reproducibility"] == "insufficient"
    )
    assert (
        not res.state_metrics.loc["S3", "eligible"]
        and res.state_metrics.loc["S0", "reproducibility"] == "stable"
    )


def test_group_effect_lowers_holdout_accuracy():
    def acc(effect):
        a = simulate_reference(
            n_donors=6,
            n_states=4,
            n_cells_per_donor=100,
            state_separation=2.0,
            donor_effect=effect,
            seed=21,
        )
        return Reference(a, **REF_KW).fit().leave_one_group_out().overall["accuracy"]

    assert acc(2.0) < acc(0.2)


def test_random_split_is_at_least_as_optimistic():
    for seed in (1, 2, 3):
        a = simulate_reference(
            n_donors=6,
            n_states=4,
            n_cells_per_donor=100,
            state_separation=2.0,
            donor_effect=1.5,
            seed=seed,
        )
        ref = Reference(a, **REF_KW).fit()
        lodo = ref.leave_one_group_out().overall["accuracy"]
        split = ref.random_split(n_splits=5)
        assert (
            split.method == "random_split"
            and split.overall["n_folds"] == 5
            and ref.validation_ is not split
        )
        assert split.overall["accuracy"] >= lodo - 0.02
    with pytest.raises(ValueError, match="n_splits"):
        Reference(
            simulate_reference(n_donors=2, n_states=3, n_cells_per_donor=6), **REF_KW
        ).fit().random_split(n_splits=50)


def test_leave_one_study_out(transport_pair):
    ref_adata, qry, _ = transport_pair
    both = ad.concat([ref_adata, qry], join="inner", merge="same", index_unique="-")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(both, study_key="study", **REF_KW).fit()
    res = ref.leave_one_study_out()
    assert (
        res.method == "leave_one_study_out"
        and res.group_key == "study"
        and res.overall["n_folds"] == 2
    )
    assert set(res.group_summary["group"]) == {"reference", "query"}
    assert ref.validation_ is None  # only group holdout is stored on the reference
    with pytest.raises(ValueError, match="study_key"):
        Reference(both, **REF_KW).fit().leave_one_study_out()


def test_anndata_backend_via_callable(small_ref):
    def fn(train, test, state_key):
        clf = LogisticRegression(max_iter=500).fit(
            train.obsm["X_latent"], train.obs[state_key].astype(str)
        )
        P = clf.predict_proba(test.obsm["X_latent"])
        return Predictions.from_arrays(
            test.obs_names, clf.classes_[P.argmax(axis=1)], proba=P, states=list(clf.classes_)
        )

    ref = Reference(small_ref.copy(), backend=CallableBackend(fn, name="logreg"), **REF_KW).fit()
    assert ref.uses_anndata_backend
    res = ref.leave_one_group_out()
    assert res.backend["name"] == "logreg" and res.overall["balanced_accuracy"] > 0.8
    assert res.latent["source"] == "external" and "fixed across folds" not in res.summary()


def test_summarize_holdout_from_external_table(fitted_small):
    full = fitted_small.leave_one_group_out()
    cp = full.cell_predictions[["group", "true_state", "predicted_state"]]
    res = summarize_holdout(cp, states=fitted_small.states_, backend="scpoli")
    assert res.backend["name"] == "scpoli" and res.overall["accuracy"] == pytest.approx(
        full.overall["accuracy"]
    )
    assert res.cell_predictions["probability"].isna().all() and list(
        res.cell_predictions.columns
    ) == list(CELL_COLUMNS)


def test_to_obs_and_dir_roundtrip(fitted_small, tmp_path):
    res = fitted_small.leave_one_group_out(per_group_confusion=True)
    res.to_obs(fitted_small.adata)
    assert (
        schema.OBS_HOLDOUT_STATE in fitted_small.adata.obs
        and schema.UNS_VALIDATION in fitted_small.adata.uns
    )
    back = ValidationResult.from_dir(res.to_dir(tmp_path / "val"))
    assert (
        back.overall == res.overall and back.group_key == "donor_id" and back.config == res.config
    )
    assert set(back.per_group_confusion) == set(res.per_group_confusion)


@pytest.mark.slow
def test_logo_scale():
    a = simulate_reference(n_donors=20, n_states=8, n_cells_per_donor=500, seed=3)
    res = (
        Reference(a, group_key="donor_id", state_key="cell_state", latent_key="X_latent")
        .fit()
        .leave_one_group_out(n_jobs=2)
    )
    assert res.overall["n_folds"] == 20 and res.cell_predictions.shape[0] == a.n_obs


def test_expression_reproducibility_is_the_same_in_every_latent(tmp_path):
    a = blurred_reference()
    calls = {}
    for key in ("X_latent", "X_blur"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = Reference(a.copy(), **{**REF_KW, "latent_key": key}).fit()
        latent = ref.leave_one_group_out(reproducibility_space="latent").state_metrics
        res = ref.leave_one_group_out(reproducibility_space="expression")
        sm = res.state_metrics
        # the latent holdout keeps its metrics; its call moves to reproducibility_latent
        assert sm[STATE_COLS[:-2]].equals(latent[STATE_COLS[:-2]])
        assert sm["reproducibility_latent"].equals(latent["reproducibility"])
        assert sm["reproducibility"].equals(sm["reproducibility_expression"])
        calls[key] = sm
    assert (calls["X_latent"]["reproducibility_latent"] == "stable").all()
    assert calls["X_blur"].loc["S1", "reproducibility_latent"] == "fragile"
    # the same reference cells get the same call whichever latent they sit in
    for col in ("reproducibility", "median_group_recall_expression", "n_groups_scored_expression"):
        assert calls["X_latent"][col].equals(calls["X_blur"][col])
    assert (calls["X_blur"]["reproducibility"] == "stable").all()
    assert "expression space" in res.summary()
    back = ValidationResult.from_dir(res.to_dir(tmp_path / "val"))
    assert back.config.reproducibility_space == "expression"
    assert list(back.state_metrics["reproducibility_latent"]) == list(sm["reproducibility_latent"])


def test_expression_reproducibility_needs_the_cell_space(small_ref, tmp_path):
    from statetrace import ExpressionConfig, LODOConfig

    with pytest.raises(ValueError, match="reproducibility_space"):
        LODOConfig(reproducibility_space="pca")
    for expression in (False, ExpressionConfig(cell_sample_per_state=0)):
        ref = Reference(small_ref.copy(), expression=expression, **REF_KW).fit()
        with pytest.raises(ValueError, match="cell-level"):
            ref.leave_one_group_out(reproducibility_space="expression")
    ref = Reference(small_ref.copy(), **REF_KW).fit()
    ref.leave_one_group_out()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref.freeze(tmp_path / "ref.statetrace")
    loaded = Reference.load(tmp_path / "ref.statetrace").fit()
    with pytest.raises(ValueError, match="AnnData"):
        loaded.leave_one_group_out(reproducibility_space="expression")


def test_auto_options_resolve_to_what_the_reference_supports():
    a = blurred_reference()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(a.copy(), **REF_KW).fit()
    res = ref.leave_one_group_out()
    # counts and a cell-level space: expression; one study: the group null
    assert res.config.reproducibility_space == "expression"
    assert res.config.calibration_unit == "group"
    assert "reproducibility_expression" in res.state_metrics
    assert "expression space" in res.summary()
    # no expression space: the latent call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = Reference(a.copy(), **REF_KW, expression=False).fit()
    assert bare.leave_one_group_out().config.reproducibility_space == "latent"
    # two studies: the study-held-out null
    b = a.copy()
    donors = sorted(b.obs["donor_id"].astype(str).unique())
    b.obs["study"] = pd.Categorical(
        np.where(b.obs["donor_id"].astype(str).isin(donors[: len(donors) // 2]), "A", "B")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = Reference(b, **REF_KW, study_key="study").fit()
    res2 = two.leave_one_group_out(reproducibility_space="latent")
    assert res2.config.calibration_unit == "study"
    assert res2.config.reproducibility_space == "latent"


def test_results_do_not_depend_on_n_jobs():
    a = blurred_reference()
    donors = sorted(a.obs["donor_id"].astype(str).unique())
    a.obs["study"] = pd.Categorical(
        np.where(a.obs["donor_id"].astype(str).isin(donors[: len(donors) // 2]), "A", "B")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(a, **REF_KW, study_key="study").fit()
    serial = ref.leave_one_group_out(n_jobs=1)
    parallel = ref.leave_one_group_out(n_jobs=2)
    assert serial.config.calibration_unit == "study"
    assert serial.config.reproducibility_space == "expression"
    pd.testing.assert_frame_equal(serial.cell_predictions, parallel.cell_predictions)
    pd.testing.assert_frame_equal(serial.holdout_evidence, parallel.holdout_evidence)
    pd.testing.assert_frame_equal(serial.distance_calibration, parallel.distance_calibration)
    pd.testing.assert_frame_equal(serial.state_metrics, parallel.state_metrics)
    pd.testing.assert_frame_equal(
        ref.random_split(n_jobs=1).cell_predictions, ref.random_split(n_jobs=2).cell_predictions
    )
