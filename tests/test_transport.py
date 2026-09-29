import warnings

import numpy as np
import pytest
from conftest import REF_KW

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
    assert st.loc["S0", "status"] == "supported" and st.loc["S2", "status"] == "supported"
    assert st.loc["S1", "status"] == "shifted" and st.loc["S1", "shift_level"] == "high"
    assert st.loc["S4", "status"] == "absent" and st.loc["S3", "status"] == "insufficient"
    assert (
        st.loc["S3", "ref_reproducibility"] == "insufficient"
        and st.loc["S0", "ref_reproducibility"] == "stable"
    )
    assert st.loc["S0", "groups_recovered"] == 6 and st.loc["S0", "query_groups"] == 6
    assert st.loc["S0", "query_recall"] > 0.9 and np.isnan(st.loc["S4", "query_recall"])
    assert route.supported_states == ["S0", "S2"] and route.states_with(["shifted", "absent"]) == [
        "S1",
        "S4",
    ]
    assert "supported=2" in route.summary()


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
    assert (route2.states.loc[["S0", "S2"], "status"] == "supported").all()


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


def test_outputs_and_report(validated_ref, transport_pair, tmp_path):
    _, qry, _ = transport_pair
    route = validate_transport(validated_ref, qry, group_key="donor_id")
    ab = route.state_abundance(states="supported")
    assert list(ab.columns) == ["S0", "S2"] and ab.shape[0] == 6 and (ab.to_numpy() <= 1).all()
    ab_all = route.state_abundance(normalize=False)
    assert ab_all.to_numpy().sum() == qry.n_obs
    pb = route.pseudobulk()
    assert (
        set(pb.obs["state"]) == {"S0", "S2"}
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
    assert (cal["mean_holdout"] >= cal["mean_in_sample"] - 1e-6).all()
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
    assert f_hold <= f_in and f_hold < 0.05
    assert (r_hold.states.loc[["S0", "S2"], "status"] == "supported").all()
