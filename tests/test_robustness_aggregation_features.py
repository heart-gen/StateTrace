import numpy as np
import pandas as pd
import pytest

from statetrace import (
    KNNBackend,
    Predictions,
    compare_backends,
    harmonize_features,
    pseudobulk,
    state_abundance,
    validate_transport,
)
from statetrace.datasets import simulate_reference


def test_compare_backends_routes(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    r15 = validate_transport(validated_ref, qry, group_key="donor_id")
    Z = qry.obsm["X_latent"]
    b5 = KNNBackend(n_neighbors=5).fit(
        validated_ref.latent_, validated_ref.states_[validated_ref.state_codes_]
    )
    P = b5.predict_proba(Z)
    p5 = Predictions.from_arrays(
        qry.obs_names,
        b5.classes_[P.argmax(axis=1)],
        proba=P,
        states=list(b5.classes_),
        backend="knn5",
    )
    r5 = validate_transport(validated_ref, qry, group_key="donor_id", predictions=p5)
    cmp_ = compare_backends({"knn15": r15, "knn5": r5})
    assert cmp_.cell_agreement > 0.9 and cmp_.pairwise.loc["knn15", "knn5"] > 0.9
    assert cmp_.state_concordance.loc["S0", "concordance"] > 0.9
    assert cmp_.status_table is not None and cmp_.status_table["status_agreement"].mean() >= 0.8
    assert cmp_.status_table.loc["S2", "consensus"] == "supported"
    assert "status agreement" in cmp_.summary()
    # predictions only: no status table; disagreement shows up
    flipped = Predictions(labels=p5.labels.replace({"S0": "S2", "S2": "S0"}), backend="flip")
    cmp2 = compare_backends({"knn5": p5, "flip": flipped})
    assert cmp2.status_table is None and cmp2.cell_agreement < 0.5
    with pytest.raises(ValueError):
        compare_backends({"only": p5})


def test_state_abundance_and_pseudobulk():
    a = simulate_reference(n_donors=3, n_states=3, n_cells_per_donor=40, n_genes=12, seed=4)
    cells = pd.DataFrame(
        {"group": a.obs["donor_id"].astype(str), "state": a.obs["cell_state"].astype(str)},
        index=a.obs_names,
    )
    ab = state_abundance(cells)
    np.testing.assert_allclose(ab.sum(axis=1), 1.0)
    ab2 = state_abundance(cells, states=["S0", "S9"])
    assert list(ab2.columns) == ["S0", "S9"] and (ab2["S9"] == 0).all()
    pb = pseudobulk(a, group_key="donor_id", state_labels="cell_state", min_cells=1)
    assert pb.n_obs == 9 and pb.n_vars == 12
    manual = np.asarray(
        a[(a.obs["donor_id"] == "D00") & (a.obs["cell_state"] == "S0")].X.sum(axis=0)
    ).ravel()
    np.testing.assert_allclose(pb["D00||S0"].X.ravel(), manual)
    assert pb.obs["n_cells"].sum() == a.n_obs
    pb_sub = pseudobulk(
        a, group_key="donor_id", state_labels=a.obs["cell_state"], states=["S1"], min_cells=1
    )
    assert set(pb_sub.obs["state"]) == {"S1"}


def test_harmonize_features_intersection_and_orthologs():
    ref = ["A", "B", "C", "D"]
    fm = harmonize_features(ref, ["C", "A", "E"])
    assert list(fm.mapping["reference"]) == ["A", "C"] and fm.retention == 0.5
    assert fm.unmapped_reference == ["B", "D"] and fm.unmapped_query == ["E"] and fm.ambiguous == []
    table = pd.DataFrame(
        {"human": ["A", "B", "B", "C", "D", "Z"], "macaque": ["a", "b1", "b2", "c", "c", "z"]}
    )
    fm2 = harmonize_features(
        ref,
        ["a", "b1", "b2", "c", "q"],
        ortholog_table=table,
        reference_col="human",
        query_col="macaque",
    )
    assert list(fm2.mapping["reference"]) == ["A"] and set(fm2.ambiguous) == {"B", "C", "D"}
    assert "1/4 reference features retained" in fm2.summary()
    a = simulate_reference(n_donors=2, n_states=2, n_cells_per_donor=5, n_genes=4)
    a.var_names = ["a", "b1", "c", "q"]
    out = fm2.apply(a)
    assert list(out.var_names) == ["A"] and list(out.var["query_feature"]) == ["a"]
    with pytest.raises(ValueError, match="unique"):
        harmonize_features(["A", "A"], ["A"])


def test_consensus_status_reads_agreement_within_the_reported_margin():
    from statetrace.robustness import consensus_status

    status = pd.DataFrame(
        {
            "a": ["supported", "shifted", "insufficient", "supported"],
            "b": ["supported", "supported_with_caution", "absent", "supported"],
            "c": ["supported_with_caution", "shifted", "absent", "supported"],
        },
        index=pd.Index(["B", "CD4", "Mega", "Mono"], name="state"),
    )
    alt = pd.DataFrame(
        {
            "a": ["supported_with_caution", "", "", ""],
            "b": ["supported_with_caution", "shifted", "", ""],
            "c": ["", "", "", ""],
        },
        index=status.index,
    )
    out = consensus_status(status, alt)
    assert list(
        out.loc[
            "Mono",
            [
                "status_agreement",
                "consensus",
                "consensus_support",
                "agreement_within_tolerance",
                "consensus_within_tolerance",
            ],
        ]
    ) == [True, "supported", 1.0, True, "supported"]
    # B: two supported, one caution; a and b sit at the cutoff (alternative caution), so caution is consistent with every backend
    assert (
        not out.loc["B", "status_agreement"]
        and out.loc["B", "consensus"] == "supported"
        and out.loc["B", "consensus_support"] == pytest.approx(2 / 3)
    )
    assert (
        out.loc["B", "agreement_within_tolerance"]
        and out.loc["B", "consensus_within_tolerance"] == "supported_with_caution"
    )
    # with only a at the cutoff, b (supported, no alternative) and c (caution) still disagree
    one = alt.copy()
    one.loc["B", "b"] = ""
    assert not consensus_status(status, one).loc["B", "agreement_within_tolerance"]
    # CD4: b's alternative is shifted, which a and c carry
    assert (
        out.loc["CD4", "agreement_within_tolerance"]
        and out.loc["CD4", "consensus_within_tolerance"] == "shifted"
    )
    # Mega: a real disagreement (no alternative bridges it)
    assert (
        not out.loc["Mega", "agreement_within_tolerance"]
        and out.loc["Mega", "consensus_within_tolerance"] is None
        and out.loc["Mega", "consensus"] == "absent"
    )
    assert (
        out["agreement_within_tolerance"].mean() == 0.75 and out["status_agreement"].mean() == 0.25
    )
    # without alternatives, within-tolerance agreement is exact agreement
    exact = consensus_status(status)
    assert (exact["agreement_within_tolerance"] == exact["status_agreement"]).all()


def test_compare_backends_consensus_columns(validated_ref, transport_pair):
    _, qry, _ = transport_pair
    r15 = validate_transport(validated_ref, qry, group_key="donor_id")
    b5 = KNNBackend(n_neighbors=5).fit(
        validated_ref.latent_, validated_ref.states_[validated_ref.state_codes_]
    )
    P = b5.predict_proba(qry.obsm["X_latent"])
    p5 = Predictions.from_arrays(
        qry.obs_names,
        b5.classes_[P.argmax(axis=1)],
        proba=P,
        states=list(b5.classes_),
        backend="knn5",
    )
    r5 = validate_transport(validated_ref, qry, group_key="donor_id", predictions=p5)
    for tol in ("cutoff", "interval", "none"):
        cmp_ = compare_backends({"knn15": r15, "knn5": r5}, tolerance=tol)
        t = cmp_.status_table
        assert {
            "consensus",
            "consensus_support",
            "agreement_within_tolerance",
            "consensus_within_tolerance",
        } <= set(t.columns)
        assert cmp_.status_agreement_within_tolerance >= cmp_.status_agreement
        assert (
            t["consensus_support"].between(0.5, 1.0).all() and "consensus status" in cmp_.summary()
        )
    with pytest.raises(ValueError, match="tolerance"):
        compare_backends({"knn15": r15, "knn5": r5}, tolerance="loose")
