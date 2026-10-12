import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import REF_KW, blurred_reference

from statetrace import (
    Predictions,
    Reference,
    TransportConfig,
    TransportRoute,
    schema,
    validate_transport,
)
from statetrace.datasets import simulate_transport
from statetrace.transport import STATE_COLUMNS, transport_status


def test_route_schema_and_statuses(validated_ref, transport_pair):
    _, qry, truth = transport_pair
    route = validate_transport(
        validated_ref, qry, group_key="donor_id", query_state_key="cell_state"
    )
    assert isinstance(route, TransportRoute) and list(route.states.columns) == list(STATE_COLUMNS)
    assert list(route.states.index) == list(validated_ref.states_)
    assert len(route.cells) == qry.n_obs and set(route.cells["category"]) <= set(
        schema.CELL_CATEGORIES
    )
    st = route.states
    # S2 is intact; S0 is intact too but, at this seed, catches misassigned cells of the
    # shifted S1, which the calibrated population call reports as an out-of-reference cloud
    assert st.loc["S2", "status"] == "supported"
    assert st.loc["S0", "status"] == "supported_with_caution" and st.loc["S0", "novel_population"]
    s0 = route.cells[route.cells["state"] == "S0"]
    assert (s0.loc[s0["true_state"] == "S0", "category"] == "supported").mean() > 0.95
    assert (s0.loc[s0["true_state"] == "S1", "category"] == "out_of_reference").all()
    assert st.loc["S1", "status"] == "shifted" and st.loc["S1", "shift_level"] == "high"
    assert st.loc["S4", "status"] == "absent" and st.loc["S3", "status"] == "insufficient"
    assert (
        st.loc["S3", "ref_reproducibility"] == "insufficient"
        and st.loc["S0", "ref_reproducibility"] == "stable"
    )
    assert st.loc["S0", "groups_recovered"] == 6 and st.loc["S0", "query_groups"] == 6
    assert st.loc["S0", "query_recall"] > 0.9 and np.isnan(st.loc["S4", "query_recall"])
    assert route.supported_states == ["S2"] and route.states_with(["shifted", "absent"]) == [
        "S1",
        "S4",
    ]
    assert "supported=1" in route.summary()


def test_unvalidated_reference_gives_caution(transport_pair):
    ref_adata, qry, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
    route = validate_transport(ref, qry, group_key="donor_id")
    assert (route.states.loc[["S0", "S2"], "ref_reproducibility"] == "unvalidated").all()
    assert (route.states.loc[["S0", "S2"], "status"] == "supported_with_caution").all()
    route2 = validate_transport(
        ref, qry, group_key="donor_id", config=TransportConfig(require_reference_validation=False)
    )
    assert route2.states.loc["S2", "status"] == "supported"


def test_novel_state_is_out_of_reference():
    ref_adata, qry, _ = simulate_transport(
        n_ref_groups=6,
        n_query_groups=4,
        n_states=5,
        n_cells_per_group=150,
        n_latent=10,
        novel_states=["S4"],
        seed=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata, **REF_KW).fit()
        ref.leave_one_group_out()
    assert "S4" not in ref.states_
    route = validate_transport(ref, qry, group_key="donor_id", query_state_key="cell_state")
    novel = route.cells[route.cells["true_state"] == "S4"]
    assert (novel["category"] == "out_of_reference").mean() > 0.9
    assert route.reference["cohort_offset_norm"] is not None
    assert (
        route.cells.loc[route.cells["true_state"] != "S4", "category"] == "out_of_reference"
    ).mean() < 0.1
    absorbing = novel["state"].mode().iloc[0]
    assert route.states.loc[absorbing, "ood_fraction"] > 0.2
    assert route.states.loc[absorbing, "novel_population"]
    assert route.states.loc[absorbing, "status"] == "supported_with_caution"
    pb = route.pseudobulk(states=[absorbing], min_cells=1)
    assert pb.obs["n_cells"].sum() == int(
        (
            (route.cells["state"] == absorbing) & (route.cells["category"] != "out_of_reference")
        ).sum()
    )
    ab = route.state_abundance(normalize=False)
    assert "unknown" in ab.columns and ab["unknown"].sum() == int(
        (route.cells["category"] == "out_of_reference").sum()
    )


def test_external_predictions_match_internal(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    internal = validate_transport(validated_ref, qry, group_key="donor_id")
    Z = qry.obsm["X_latent"]
    P = validated_ref.backend_.predict_proba(Z)
    preds = Predictions.from_arrays(
        qry.obs_names,
        validated_ref.backend_.classes_[P.argmax(axis=1)],
        proba=P,
        states=list(validated_ref.backend_.classes_),
        backend="ext",
    )
    external = validate_transport(validated_ref, qry, group_key="donor_id", predictions=preds)
    assert external.predictions.backend == "ext"
    assert (external.states["status"] == internal.states["status"]).all()
    # labels-only predictions from a tool that never saw S2: unknown labels become out_of_reference
    lab = preds.labels.replace({"S2": "weird"})
    loose = validate_transport(
        validated_ref,
        qry,
        group_key="donor_id",
        predictions=Predictions(labels=lab, backend="labels"),
    )
    assert loose.states.loc["S2", "status"] == "absent"
    assert (loose.cells.loc[lab == "weird", "category"] == "out_of_reference").all()


def test_errors(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    with pytest.raises(KeyError, match="nope"):
        validate_transport(validated_ref, qry, group_key="nope")
    q2 = qry.copy()
    del q2.obsm["X_latent"]
    with pytest.raises(KeyError, match="obsm"):
        validate_transport(validated_ref, q2, group_key="donor_id")
    q3 = qry.copy()
    q3.obsm["X_latent"] = np.zeros((qry.n_obs, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dims"):
        validate_transport(validated_ref, q3, group_key="donor_id")
    with pytest.raises(ValueError, match="cover"):
        validate_transport(
            validated_ref,
            qry,
            group_key="donor_id",
            predictions=Predictions.from_arrays(qry.obs_names[:5], ["S0"] * 5),
        )


def test_status_rule_table():
    cfg = TransportConfig()
    base = {
        "n_cells": 500,
        "groups_observed": 6,
        "groups_recovered": 6,
        "recovery_rate": 1.0,
        "ref_reproducibility": "stable",
        "shift_level": "low",
        "ood_fraction": 0.0,
        "uncertain_fraction": 0.0,
        "groups_recovered_in_reference": 6,
        "shift_in_reference_level": "low",
    }
    assert transport_status(base, cfg) == "supported"
    assert transport_status({**base, "n_cells": 10}, cfg) == "absent"
    assert transport_status({**base, "ref_reproducibility": "fragile"}, cfg) == "insufficient"
    assert transport_status({**base, "groups_recovered": 2}, cfg) == "insufficient"
    assert (
        transport_status(
            {
                **base,
                "groups_recovered": 2,
                "shift_level": "high",
                "groups_recovered_in_reference": 1,
            },
            cfg,
        )
        == "shifted"
    )
    assert (
        transport_status({**base, "ref_reproducibility": "fragile", "shift_level": "high"}, cfg)
        == "insufficient"
    )
    assert (
        transport_status({**base, "shift_level": "high", "groups_recovered_in_reference": 1}, cfg)
        == "shifted"
    )
    assert (
        transport_status({**base, "ood_fraction": 0.5, "groups_recovered_in_reference": 1}, cfg)
        == "shifted"
    )
    assert (
        transport_status({**base, "ood_fraction": 0.5, "shift_in_reference_level": "high"}, cfg)
        == "shifted"
    )
    # an out-of-reference cloud next to an intact state is a caution, not a shift
    assert transport_status({**base, "ood_fraction": 0.5}, cfg) == "supported_with_caution"
    assert transport_status({**base, "recovery_rate": 0.5}, cfg) == "supported_with_caution"
    assert transport_status({**base, "uncertain_fraction": 0.3}, cfg) == "supported_with_caution"
    assert transport_status({**base, "shift_level": "moderate"}, cfg) == "supported_with_caution"
    # a tool with neither confidence nor a latent cannot earn full support
    assert transport_status({**base, "confidence_source": "none"}, cfg) == "supported_with_caution"
    assert transport_status({**base, "confidence_source": "support"}, cfg) == "supported"
    # absence is judged on cells with local reference support
    assert transport_status({**base, "n_cells_supported": 10}, cfg) == "absent"


def test_config_validation():
    with pytest.raises(ValueError, match="ood_rule"):
        TransportConfig(ood_rule="nope")
    with pytest.raises(ValueError, match="ood_quantile"):
        TransportConfig(ood_rule="quantile", ood_quantile=0.5)
    with pytest.raises(ValueError, match="min_support"):
        TransportConfig(min_support=2.0)
    TransportConfig(min_support=None, ood_rule="quantile", ood_quantile=0.95)


def test_local_support_is_computed_in_the_latent(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    Z = qry.obsm["X_latent"]
    P = validated_ref.backend_.predict_proba(Z)
    labels = validated_ref.backend_.classes_[P.argmax(axis=1)]
    preds = Predictions.from_arrays(
        qry.obs_names, labels, proba=P, states=list(validated_ref.backend_.classes_), backend="ext"
    )
    route = validate_transport(validated_ref, qry, group_key="donor_id", predictions=preds)
    sup = route.cells["support"]
    assert route.reference["support"] == {"source": "latent_knn", "n_neighbors": 15}
    assert route.reference["confidence_source"] == "backend"
    assert sup.notna().all() and sup.between(0, 1).all()
    # the kNN backend's confidence is its own vote fraction, so support agrees with it closely
    assert np.corrcoef(sup, route.cells["confidence"])[0, 1] > 0.9
    # a label the latent neighbourhood does not carry has no support -> uncertain, not observed
    swapped = labels.copy()
    s0 = swapped == "S0"
    swapped[s0] = "S2"
    preds2 = Predictions.from_arrays(qry.obs_names, swapped, backend="ext")
    route2 = validate_transport(validated_ref, qry, group_key="donor_id", predictions=preds2)
    cells2 = route2.cells
    assert (cells2.loc[s0, "support"] < 0.5).mean() > 0.9
    assert (cells2.loc[s0, "category"] != "supported").mean() > 0.9
    assert route2.states.loc["S0", "status"] == "absent"
    assert route2.states.loc["S2", "n_cells_supported"] < route2.states.loc["S2", "n_cells"]
    # a tool's own support is kept as given
    preds3 = Predictions.from_arrays(
        qry.obs_names, labels, proba=P, states=list(preds.states), support=np.ones(len(labels))
    )
    route3 = validate_transport(validated_ref, qry, group_key="donor_id", predictions=preds3)
    assert route3.reference["support"]["source"] == "backend"
    assert (route3.cells["support"] == 1.0).all()
    # disabling the rule restores observation on every assigned cell
    route4 = validate_transport(
        validated_ref,
        qry,
        group_key="donor_id",
        predictions=preds2,
        config=TransportConfig(min_support=None),
    )
    assert (route4.states["n_cells_supported"] == route4.states["n_cells"]).all()


def test_labels_only_predictions(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    Z = qry.obsm["X_latent"]
    labels = validated_ref.backend_.classes_[validated_ref.backend_.predict_proba(Z).argmax(axis=1)]
    preds = Predictions.from_arrays(qry.obs_names, labels, backend="labels")
    assert preds.confidence.isna().all()
    # with the query latent, local support stands in for confidence
    route = validate_transport(validated_ref, qry, group_key="donor_id", predictions=preds)
    assert route.reference["confidence_source"] == "support"
    assert (route.states["confidence_source"] == "support").all()
    assert route.cells["confidence"].notna().all()
    assert route.states.loc["S2", "status"] == "supported"
    assert route.states.loc["S4", "status"] == "absent"
    # without a latent, recovery is judged on counts and every state is at most a caution
    q = qry.copy()
    del q.obsm["X_latent"]
    with pytest.warns(UserWarning, match="no confidence"):
        bare = validate_transport(validated_ref, q, group_key="donor_id", predictions=preds)
    assert bare.reference["confidence_source"] == "none"
    assert bare.reference["support"]["source"] == "none"
    assert not (bare.states["status"] == "supported").any()
    assert (bare.states.loc[["S0", "S2"], "status"] == "supported_with_caution").all()
    assert bare.states.loc["S4", "status"] == "absent"
    assert bare.cells["confidence"].isna().all()
    assert (bare.cells["category"] != "uncertain").all()


def test_ood_rule_quantile(transport_pair):
    import warnings

    ref_adata, qry, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        ref.leave_one_group_out()
    cal = ref.distance_calibration_
    r_q = validate_transport(
        ref,
        qry,
        group_key="donor_id",
        query_state_key="cell_state",
        config=TransportConfig(ood_rule="quantile", ood_quantile=0.95),
    )
    assert r_q.reference["ood_rule"] == "quantile"
    cells = r_q.cells
    # the mask is exactly "distance above the state's held-out 95th percentile"
    thr = cells["state"].map(cal["q95_holdout"]).to_numpy(dtype=float)
    expect = np.nan_to_num(cells["distance"].to_numpy(dtype=float), nan=0.0) > thr
    known = cells["state"] != schema.UNKNOWN_LABEL
    assert ((cells["category"] == "out_of_reference").to_numpy()[known] == expect[known]).all()
    # intact states from new donors are flagged near the nominal 5 %
    stable = cells["true_state"].isin(["S0", "S2"])
    rate = (cells.loc[stable, "category"] == "out_of_reference").mean()
    assert 0.0 < rate < 0.15
    # no calibration -> the z rule is applied and recorded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref2 = Reference(ref_adata.copy(), **REF_KW).fit()
    r_fallback = validate_transport(
        ref2,
        qry,
        group_key="donor_id",
        config=TransportConfig(ood_rule="quantile", require_reference_validation=False),
    )
    assert r_fallback.reference["ood_rule"] == "z"
    r_z = validate_transport(
        ref2, qry, group_key="donor_id", config=TransportConfig(require_reference_validation=False)
    )
    assert (r_fallback.cells["category"] == r_z.cells["category"]).all()


def test_outputs_and_report(validated_ref, transport_pair, tmp_path):
    _, qry, _ = transport_pair
    route = validate_transport(validated_ref, qry, group_key="donor_id")
    ab = route.state_abundance(states="supported")
    assert list(ab.columns) == ["S2"] and ab.shape[0] == 6 and (ab.to_numpy() <= 1).all()
    ab_all = route.state_abundance(normalize=False)
    assert ab_all.to_numpy().sum() == qry.n_obs
    pb = route.pseudobulk()
    assert (
        set(pb.obs["state"]) == {"S2"}
        and pb.n_vars == qry.n_vars
        and (pb.obs["n_cells"] >= 10).all()
    )
    pb2 = route.pseudobulk(states=["S1"], min_cells=1, exclude_out_of_reference=False)
    assert (
        route.pseudobulk(states=["S1"], min_cells=1).n_obs == 0
    )  # every S1 cell is out of reference
    assert set(pb2.obs["state"]) == {"S1"}
    route.to_obs()
    assert (
        schema.OBS_STATE in qry.obs
        and schema.OBS_CATEGORY in qry.obs
        and schema.UNS_TRANSPORT in qry.uns
    )
    out = route.report(tmp_path / "report")
    assert (
        (out / "report.md").exists()
        and (out / "states.csv").exists()
        and (out / "cells.csv.gz").exists()
    )
    text = (out / "report.md").read_text()
    assert "| S1 |" in text and "shifted" in text and "supported" in text
    detached = validate_transport(validated_ref, qry, group_key="donor_id", keep_query=False)
    with pytest.raises(ValueError, match="query AnnData"):
        detached.pseudobulk()


def test_holdout_calibration_lowers_false_ood(transport_pair):
    import warnings

    from statetrace import TransportConfig

    ref_adata, qry, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        res = ref.leave_one_group_out()
    cal = res.distance_calibration
    assert cal is not None and list(cal.index) == list(ref.states_)
    assert (cal["n_holdout"] > 0).all()
    # held-out distances are measured after removing each held-out donor's own offset, so
    # they are no longer bounded below by the in-sample ones; they must still be finite
    assert cal["mean_holdout"].notna().all() and (cal["sd_holdout"] > 0).all()
    r_hold = validate_transport(ref, qry, group_key="donor_id", query_state_key="cell_state")
    r_in = validate_transport(
        ref,
        qry,
        group_key="donor_id",
        query_state_key="cell_state",
        config=TransportConfig(novelty_calibration="in_sample"),
    )
    assert (
        r_hold.reference["novelty_calibration"] == "holdout"
        and r_in.reference["novelty_calibration"] == "in_sample"
    )
    stable = r_hold.cells["true_state"].isin(["S0", "S2"])
    f_hold = (r_hold.cells.loc[stable, "category"] == "out_of_reference").mean()
    f_in = (r_in.cells.loc[stable, "category"] == "out_of_reference").mean()
    assert f_hold <= f_in + 0.01 and f_hold < 0.05
    assert r_hold.states.loc["S2", "status"] == "supported"


def test_knn_and_joint_scores_and_calibrated_thresholds(transport_pair):
    import warnings

    ref_adata, qry, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        res = ref.leave_one_group_out()
    cal = res.distance_calibration
    for col in (
        "knn_mean_holdout",
        "knn_sd_holdout",
        "knn_q95_holdout",
        "joint_q95_holdout",
        "support_q05_holdout",
        "n_groups_holdout",
    ):
        assert col in cal and cal[col].notna().all()
    ev = res.holdout_evidence
    assert list(ev.columns) == [
        "group",
        "true_state",
        "assigned_state",
        "distance",
        "knn_distance",
        "support",
    ]
    assert list(ev.index) == list(ref_adata.obs_names)
    assert ev["support"].between(0, 1).all() and (ev["knn_distance"] > 0).all()
    base = validate_transport(ref, qry, group_key="donor_id", query_state_key="cell_state")
    cells = base.cells
    assert cells["knn_distance"].notna().all() and cells["knn_novelty"].notna().all()
    assert cells["joint_novelty"].notna().all()
    assert (cells["joint_novelty"] >= cells["novelty"] - 1e-9).all()
    assert base.reference["support_calibration"] == "holdout"
    assert base.reference["population_calibration"] == "holdout"
    assert (base.states["support_threshold"] <= 0.5).all()
    # calibrated population threshold: at least the floor, and above max_ood_fraction only
    # when a reference donor produced that much
    assert (base.states["ood_threshold"] >= 0.1).all()
    shifted = cells["true_state"] == "S1"
    for score, rule, q in (
        ("knn", "quantile", 0.95),
        ("joint", "quantile", 0.95),
        ("joint", "z", 0.99),
    ):
        r = validate_transport(
            ref,
            qry,
            group_key="donor_id",
            query_state_key="cell_state",
            config=TransportConfig(ood_score=score, ood_rule=rule, ood_quantile=q),
        )
        assert r.reference["ood_score"] == score and r.reference["ood_rule"] == rule
        assert not r.reference["ood_fallback"]
        flagged = r.cells["category"] == "out_of_reference"
        assert flagged[shifted].mean() > 0.9
        assert flagged[r.cells["true_state"].isin(["S0", "S2"])].mean() < 0.1
    # the joint quantile rule is exactly "joint score above the state's held-out quantile"
    r = validate_transport(
        ref,
        qry,
        group_key="donor_id",
        config=TransportConfig(ood_score="joint", ood_rule="quantile", ood_quantile=0.99),
    )
    thr = r.cells["state"].map(cal["joint_q99_holdout"]).to_numpy(dtype=float)
    known = r.cells["state"] != schema.UNKNOWN_LABEL
    expect = r.cells["joint_novelty"].to_numpy(dtype=float) > thr
    assert ((r.cells["category"] == "out_of_reference").to_numpy()[known] == expect[known]).all()
    # without validation the knn and joint scores are unavailable and the call falls back
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = Reference(ref_adata.copy(), **REF_KW).fit()
    r_bare = validate_transport(
        bare,
        qry,
        group_key="donor_id",
        config=TransportConfig(ood_score="joint", require_reference_validation=False),
    )
    assert r_bare.reference["ood_fallback"] and r_bare.reference["ood_score"] == "centroid"
    assert r_bare.cells["knn_novelty"].isna().all()
    assert r_bare.reference["support_calibration"] == "fixed"
    assert r_bare.reference["population_calibration"] == "fixed"
    assert (r_bare.states["ood_threshold"] == 0.3).all()
    # fixed modes are honoured on a validated reference too
    r_fixed = validate_transport(
        ref,
        qry,
        group_key="donor_id",
        config=TransportConfig(support_calibration="fixed", population_calibration="fixed"),
    )
    assert (r_fixed.states["support_threshold"] == 0.5).all()
    assert (r_fixed.states["ood_threshold"] == 0.3).all()
    with pytest.raises(ValueError, match="ood_score"):
        TransportConfig(ood_score="nope")
    with pytest.raises(ValueError, match="support_quantile"):
        TransportConfig(support_quantile=0.5)


def test_group_offsets_cover_query_donors_with_their_own_displacement(transport_pair):
    import warnings

    from statetrace import novelty

    ref_adata, qry, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        ref.leave_one_group_out()
    # give every query donor its own latent displacement, as a surgery-fitted mapper would
    q = qry.copy()
    rng = np.random.default_rng(0)
    Z = np.asarray(q.obsm["X_latent"], dtype=np.float32)
    donors, codes = np.unique(q.obs["donor_id"].astype(str), return_inverse=True)
    shifts = rng.normal(0, 1.5, size=(len(donors), Z.shape[1])).astype(np.float32)
    q.obsm["X_latent"] = Z + shifts[codes]
    cfg_q = TransportConfig(ood_rule="quantile", ood_quantile=0.95)
    r_group = validate_transport(
        ref, q, group_key="donor_id", query_state_key="cell_state", config=cfg_q
    )
    r_cohort = validate_transport(
        ref,
        q,
        group_key="donor_id",
        query_state_key="cell_state",
        config=dataclasses.replace(cfg_q, latent_group_offset=False),
    )
    assert r_group.reference["latent_offset"]["mode"] == "group"
    assert r_group.reference["latent_offset"]["groups_with_own_offset"] == len(donors)
    assert r_cohort.reference["latent_offset"]["mode"] == "cohort"
    intact = r_group.cells["true_state"].isin(["S0", "S2"])
    f_group = (r_group.cells.loc[intact, "category"] == "out_of_reference").mean()
    f_cohort = (r_cohort.cells.loc[intact, "category"] == "out_of_reference").mean()
    assert f_cohort > 0.2 and f_group < 0.1
    # the planted state shift and the absent state are still read as such
    assert r_group.states.loc["S1", "status"] == "shifted"
    assert r_group.states.loc["S4", "status"] == "absent"
    # the estimator recovers the planted displacements up to the cohort median
    offsets, own = novelty.estimate_group_offsets(
        q.obsm["X_latent"],
        codes,
        pd.Index(ref.states_).get_indexer(q.obs["cell_state"].astype(str)),
        ref.centroids_,
    )
    assert own.all()
    resid = offsets - shifts
    # per-state means from ~25 cells carry sampling noise of about 0.3 per dimension
    assert np.abs(resid - resid.mean(axis=0)).mean() < 0.5


def test_geometric_median_resists_one_outlying_state():
    from statetrace.novelty import geometric_median

    rng = np.random.default_rng(1)
    cluster = rng.normal(0, 0.1, size=(3, 10))
    outlier = np.full((1, 10), 9.0)
    X = np.vstack([cluster, outlier])
    gm = geometric_median(X)
    assert np.linalg.norm(gm - cluster.mean(axis=0)) < 0.3
    assert np.allclose(geometric_median(X[:1]), X[0])
    # with two of four rows displaced in different directions the component-wise median
    # averages a cluster row with a displaced row in every coordinate and drifts; the
    # geometric median is pulled far less
    two = np.vstack(
        [
            cluster[:2],
            np.full((1, 10), 9.0),
            -np.full((1, 10), 9.0) * np.r_[np.ones(5), -np.ones(5)],
        ]
    )
    gm2 = geometric_median(two)
    cm2 = np.median(two, axis=0)
    assert np.linalg.norm(gm2 - cluster[:2].mean(axis=0)) < np.linalg.norm(
        cm2 - cluster[:2].mean(axis=0)
    )


def test_raw_and_residual_displacement_are_both_reported(validated_ref, transport_pair, tmp_path):
    _, qry, _ = transport_pair
    route = validate_transport(validated_ref, qry, group_key="donor_id")
    st, gr = route.states, route.groups
    for col in ("latent_shift_raw", "expression_shift_raw", "latent_shift", "expression_shift"):
        assert col in st and st[col].notna().any()
    for col in (
        "latent_offset_sd",
        "expression_offset_rms",
        "expression_distance_raw",
        "expression_shift_z_raw",
    ):
        assert col in gr and gr[col].notna().any()
    # the residual decides; the raw value is kept beside it, and the planted shift of S1 is
    # visible in both
    assert st.loc["S1", "latent_shift_raw"] > 4 and st.loc["S1", "latent_shift"] > 4
    assert (gr["latent_offset_sd"] >= 0).all()
    text = (route.report(tmp_path / "rep") / "report.md").read_text()
    assert "raw/residual" in text and "| donor |" in text


def _four_studies(seed: int = 1, study_state_effect: float = 1.0, drop: str | None = None):
    """16 donors in 4 studies with a state-specific study effect; study st3 is the query."""
    from statetrace.datasets import simulate_reference

    a = simulate_reference(
        n_donors=16,
        n_states=4,
        n_cells_per_donor=150,
        donor_effect=0.5,
        state_separation=4.0,
        n_latent=10,
        n_genes=0,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 100)
    study = a.obs["donor_id"].cat.codes.to_numpy() // 4
    state = a.obs["cell_state"].cat.codes.to_numpy()
    a.obs["study"] = pd.Categorical([f"st{k}" for k in study])
    eta = rng.normal(0.0, study_state_effect, size=(4, 4, 10))
    a.obsm["X_latent"] = (a.obsm["X_latent"] + eta[study, state]).astype(np.float32)
    keep = study < 3
    if drop is not None:  # the state is seen in study st0 only
        keep &= (study == 0) | (a.obs["cell_state"].to_numpy() != drop)
    ref, qry = a[keep].copy(), a[study == 3].copy()
    for d in (ref, qry):
        for c in ("donor_id", "study", "cell_state"):
            d.obs[c] = d.obs[c].cat.remove_unused_categories()
    return ref, qry


def test_study_calibration_holds_false_rate_on_a_new_study():
    from statetrace import LODOConfig

    ref_adata, qry = _four_studies()
    cfg = TransportConfig(ood_rule="quantile", ood_quantile=0.95)
    frac, cal = {}, {}
    for unit in ("group", "study"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = Reference(ref_adata.copy(), study_key="study", **REF_KW).fit()
            res = ref.leave_one_group_out(calibration_unit=unit)
            route = validate_transport(
                ref, qry, group_key="donor_id", query_state_key="cell_state", config=cfg
            )
        assert route.reference["novelty_calibration_unit"] == unit
        assert (res.distance_calibration["calibration_unit"] == unit).all()
        cal[unit] = res.distance_calibration
        frac[unit] = (route.cells["category"] == "out_of_reference").mean()
    # every query state is intact: the donor-level null misses the study effect
    assert frac["group"] > 0.12 and frac["study"] < 0.10
    assert (cal["study"]["q95_holdout"] > cal["group"]["q95_holdout"]).all()
    ev = res.holdout_evidence
    assert set(ev["unit"]) == {"study"} and ev["group"].isin(ref_adata.obs["donor_id"]).all()
    with pytest.raises(ValueError, match="calibration_unit"):
        LODOConfig(calibration_unit="donor")


def test_study_calibration_falls_back_per_state(tmp_path):
    from statetrace.validation import ValidationResult

    ref_adata, _ = _four_studies(drop="S3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), study_key="study", **REF_KW).fit()
        res = ref.leave_one_group_out(calibration_unit="study")
    units = res.distance_calibration["calibration_unit"]
    assert units["S3"] == "group" and (units.drop("S3") == "study").all()
    assert res.distance_calibration["q95_holdout"].notna().all()
    back = ValidationResult.from_dir(res.to_dir(tmp_path / "val"))
    assert back.config.calibration_unit == "study"
    assert list(back.distance_calibration["calibration_unit"]) == list(units)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        no_study = Reference(ref_adata.copy(), **REF_KW).fit()
        with pytest.raises(ValueError, match="study labels"):
            no_study.leave_one_group_out(calibration_unit="study")


def test_transport_reports_both_reproducibility_calls():
    a = blurred_reference(n_donors=12)
    ref_donors = [f"D{i:02d}" for i in range(8)]
    in_ref = a.obs["donor_id"].isin(ref_donors).to_numpy()
    ref_adata, qry = a[in_ref].copy(), a[~in_ref].copy()
    for d in (ref_adata, qry):
        d.obs["donor_id"] = d.obs["donor_id"].cat.remove_unused_categories()
    status = {}
    for space in ("latent", "expression"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = Reference(ref_adata.copy(), **{**REF_KW, "latent_key": "X_blur"}).fit()
            ref.leave_one_group_out(reproducibility_space=space)
            route = validate_transport(ref, qry, group_key="donor_id", latent_key="X_blur")
        st = route.states
        assert route.reference["reproducibility_space"] == space
        assert st.loc["S1", "ref_reproducibility_latent"] in ("moderate", "fragile")
        if space == "latent":
            assert st["ref_reproducibility"].equals(st["ref_reproducibility_latent"])
            assert st["ref_reproducibility_expression"].isna().all()
        else:
            assert st["ref_reproducibility"].equals(st["ref_reproducibility_expression"])
            assert (st["ref_reproducibility"] == "stable").all()
        status[space] = st["status"]
    assert status["latent"]["S1"] != "supported"


def test_knn_score_finds_a_withheld_state_between_elongated_states():
    from sklearn.metrics import roc_auc_score

    ref_adata, qry, _ = simulate_transport(
        n_ref_groups=8,
        n_query_groups=6,
        n_states=6,
        n_cells_per_group=300,
        n_latent=10,
        cohort_effect=0.3,
        novel_states=["S5"],
        novel_between={"S5": ("S2", "S3", 0.35)},
        state_anisotropy=3.0,
        anisotropic_dims=2,
        seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_adata.copy(), **REF_KW).fit()
        ref.leave_one_group_out()
        route = validate_transport(ref, qry, group_key="donor_id", query_state_key="cell_state")
    cells = route.cells.reindex(qry.obs_names)
    novel = qry.obs["cell_state"].astype(str).to_numpy() == "S5"
    auc = {
        col: roc_auc_score(novel, cells[col].fillna(cells[col].max()))
        for col in ("novelty", "knn_novelty", "joint_novelty")
    }
    # the elongated absorbing states hide the population from the centroid distance
    assert auc["knn_novelty"] > auc["novelty"] + 0.15
    assert auc["joint_novelty"] > auc["novelty"]
