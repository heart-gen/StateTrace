import dataclasses
import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import REF_KW
from hypothesis import given, settings
from hypothesis import strategies as hst

from statetrace import (
    Predictions,
    Reference,
    TransportConfig,
    TwoSampleConfig,
    schema,
    validate_transport,
)
from statetrace import twosample as ts
from statetrace.datasets import simulate_transport

HAS_SCXMATCH = importlib.util.find_spec("scxmatch") is not None


@pytest.fixture(scope="module")
def spread_pair():
    """No cohort effect; S1 displaced, S2 spread (mean kept), the rest intact."""
    return simulate_transport(
        n_ref_groups=6,
        n_query_groups=5,
        n_states=4,
        n_cells_per_group=160,
        n_latent=8,
        n_genes=80,
        donor_effect=0.3,
        state_separation=3.0,
        cohort_effect=0.0,
        shifted_states={"S1": 4.0},
        spread_states={"S2": 2.5},
        seed=3,
    )


@pytest.fixture(scope="module")
def spread_ref(spread_pair):
    ref_a, _, _ = spread_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Reference(ref_a.copy(), **REF_KW).fit()


def _oracle(ref, qry):
    codes = qry.obs["cell_state"].astype(str).map({s: i for i, s in enumerate(ref.states_)})
    P = np.zeros((qry.n_obs, len(ref.states_)))
    P[np.arange(qry.n_obs), codes.to_numpy()] = 1.0
    return Predictions.from_arrays(
        qry.obs_names,
        qry.obs["cell_state"].astype(str).to_numpy(),
        proba=P,
        states=list(ref.states_),
        backend="oracle",
    )


def test_energy_distance_and_permutation_test():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(150, 5))
    B = rng.normal(size=(120, 5))
    assert ts.energy_distance(A, A) == pytest.approx(0.0, abs=1e-12)
    null = ts.edistance_test(A, B, n_permutations=199, seed=1)
    assert null["p_value"] > 0.05 and null["n_x"] == null["n_y"] == 120
    mean = ts.edistance_test(A, B + 0.8, n_permutations=199, seed=1)
    assert mean["p_value"] <= 0.01 and mean["z_score"] > 3
    # a change in spread with the mean kept: the energy distance sees it, the squared
    # Euclidean form reduces to a difference of means and does not
    wide = B * 2.0
    assert ts.edistance_test(A, wide, n_permutations=199, seed=1)["p_value"] <= 0.01
    sq = ts.edistance_test(A, wide, n_permutations=199, seed=1, metric="sqeuclidean")
    assert sq["p_value"] > 0.01
    full = ts.edistance_test(A, B, n_permutations=50, subsample_to_smaller=False)
    assert full["n_x"] == 150 and full["n_y"] == 120
    with pytest.raises(ValueError, match="two profiles"):
        ts.edistance_test(A[:1], B, n_permutations=10)


@settings(max_examples=25, deadline=None)
@given(
    n=hst.integers(min_value=2, max_value=12),
    m=hst.integers(min_value=2, max_value=12),
    shift=hst.floats(min_value=-3, max_value=3),
    seed=hst.integers(min_value=0, max_value=10_000),
)
def test_energy_distance_properties(n, m, shift, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    Y = rng.normal(size=(m, 3)) + shift
    e = ts.energy_distance(X, Y)
    assert e >= -1e-9 and e == pytest.approx(ts.energy_distance(Y, X))
    r = ts.edistance_test(X, Y, n_permutations=20, seed=seed, subsample_to_smaller=False)
    assert 0 < r["p_value"] <= 1 and r["statistic"] == pytest.approx(e)


def test_permutation_p_is_calibrated_under_the_null():
    rng = np.random.default_rng(7)
    p = [
        ts.edistance_test(
            rng.normal(size=(40, 4)), rng.normal(size=(40, 4)), n_permutations=99, seed=i
        )["p_value"]
        for i in range(60)
    ]
    assert np.mean(np.asarray(p) <= 0.05) <= 0.15 and 0.3 < np.median(p) < 0.7


def test_config_validation():
    cfg = TwoSampleConfig(engines=["edistance", "scxmatch"], levels=["donor"])
    assert cfg.engines == ("edistance", "scxmatch") and cfg.levels == ("donor",)
    assert cfg.to_dict()["n_permutations"] == 1000
    for bad in (
        {"engines": ("mmd",)},
        {"engines": ()},
        {"levels": ("cluster",)},
        {"edistance_metric": "cosine"},
        {"n_permutations": 0},
        {"min_groups": 1},
    ):
        with pytest.raises(ValueError):
            TwoSampleConfig(**bad)
    with pytest.raises(TypeError, match="TwoSampleConfig"):
        TransportConfig(two_sample={"engines": ("edistance",)})


def test_frozen_cell_space_round_trip(spread_ref, tmp_path):
    import copy

    spread_ref = copy.deepcopy(spread_ref)  # freezing marks the module fixture frozen
    sp = spread_ref.expression_
    assert sp.has_cell_space and sp.null_profiles is not None
    assert sp.null_profiles.shape == (len(sp.null), sp.n_genes)
    n_per_state = sp.cell_rows.groupby("state").size()
    assert (n_per_state <= 500).all() and len(n_per_state) == 4
    assert sp.cell_coords.shape == (len(sp.cell_rows), sp.cell_components.shape[0])
    assert sp.cell_components.shape[1] == sp.n_genes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = spread_ref.freeze(tmp_path / "ref", overwrite=True)
    back = Reference.load(path).expression_
    for name in ("cell_center", "cell_scale", "cell_components", "cell_coords", "null_profiles"):
        assert np.allclose(getattr(back, name), getattr(sp, name))
    pd.testing.assert_frame_equal(back.cell_rows, sp.cell_rows)


def test_engines_are_reported_and_never_decide(spread_ref, spread_pair, tmp_path):
    _, qry, _ = spread_pair
    preds = _oracle(spread_ref, qry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = validate_transport(spread_ref, qry, group_key="donor_id", predictions=preds)
        route = validate_transport(
            spread_ref,
            qry,
            group_key="donor_id",
            predictions=preds,
            config=TransportConfig(
                two_sample=TwoSampleConfig(engines=("edistance", "scxmatch"), n_permutations=199)
            ),
        )
    assert base.two_sample is None and base.reference["two_sample"] is None
    pd.testing.assert_frame_equal(base.states, route.states)
    t = route.two_sample
    assert tuple(t.columns) == ts.TEST_COLUMNS
    assert len(t) == 4 * 2 * 2 * 2  # states x engines x levels x subsets
    assert set(t["status"]) <= set(schema.TEST_STATUS)
    assert set(t["subset"]) == set(schema.TWO_SAMPLE_SUBSETS)
    e = t[(t["engine"] == "edistance") & (t["subset"] == "all")].set_index(["level", "state"])
    assert (e["status"] == "ok").all()
    donor = e.loc["donor"]
    assert (donor["n_ref"] == 6).all() and (donor["n_query"] == 5).all()
    # the displaced and the spread state carry the largest donor-level evidence
    assert set(donor["z_score"].nlargest(2).index) == {"S1", "S2"}
    assert donor.loc["S1", "p_value"] < 0.05
    sx = t[t["engine"] == "scxmatch"]
    # a side too small for any engine is skipped before the engine is reached
    assert (sx["status"] == "skipped").sum() == (
        t[t["engine"] == "edistance"]["status"] == "skipped"
    ).sum()
    if HAS_SCXMATCH:
        assert set(sx["status"]) <= {"ok", "skipped"}
    else:
        assert set(sx["status"]) <= {"unsupported", "skipped"}
        unsup = sx[sx["status"] == "unsupported"]
        assert len(unsup) and unsup["reason"].str.contains("conda install scxmatch").all()
    out = route.report(tmp_path / "report")
    assert pd.read_csv(out / "two_sample.csv").shape == t.shape
    md = (out / "report.md").read_text()
    assert "Two-sample engines (reported, not decisive)" in md
    assert "Two-sample" not in (base.report(tmp_path / "base") / "report.md").read_text()
    prov = route.reference["two_sample"]
    assert prov["state_definition"] == "frozen_reference" and prov["cell_space"]
    assert not prov["query_in_state_definition"]


def test_inputs_are_what_the_engines_see(spread_ref, spread_pair):
    _, qry, _ = spread_pair
    sp = spread_ref.expression_
    codes = qry.obs["cell_state"].astype(str).map({s: i for i, s in enumerate(sp.states)})
    groups, _ = pd.factorize(qry.obs["donor_id"].astype(str), sort=True)
    args = (sp, qry[:, sp.genes].X, groups, codes.to_numpy(), np.ones(qry.n_obs, dtype=bool))
    cfg = TwoSampleConfig(n_permutations=49, levels=("donor",), subsample_to_smaller=False)
    items = ts.state_test_inputs(*args, config=cfg)
    table, _ = ts.state_tests(*args, config=cfg)
    assert len(items) == len(table) == 4 * 2
    it = next(i for i in items if i["state"] == "S1" and i["subset"] == "all")
    assert it["status"] is None and it["reference"].shape == (6, sp.n_genes)
    assert it["query"].shape == (5, sp.n_genes) and len(set(it["query_groups"])) == 5
    row = table[(table["state"] == "S1") & (table["subset"] == "all")].iloc[0]
    assert row["statistic"] == pytest.approx(ts.energy_distance(it["reference"], it["query"]))


def test_format3_space_reports_unsupported(spread_ref, spread_pair):
    _, qry, _ = spread_pair
    old = dataclasses.replace(
        spread_ref.expression_,
        null_profiles=None,
        cell_center=None,
        cell_scale=None,
        cell_components=None,
        cell_coords=None,
        cell_rows=None,
    )
    codes = qry.obs["cell_state"].astype(str).map({s: i for i, s in enumerate(old.states)})
    groups, _ = pd.factorize(qry.obs["donor_id"].astype(str), sort=True)
    table, prov = ts.state_tests(
        old,
        qry[:, old.genes].X,
        groups,
        codes.to_numpy(),
        np.ones(qry.n_obs, dtype=bool),
        config=TwoSampleConfig(n_permutations=10),
    )
    assert (table["status"] == "unsupported").all() and not prov["cell_space"]
    assert table["reason"].str.contains("re-freeze").all()


def test_query_overlapping_the_reference_is_refused(spread_ref, spread_pair):
    ref_a, _, _ = spread_pair
    leak = ref_a[ref_a.obs["donor_id"].isin(["D00", "D01"])].copy()
    with pytest.raises(ValueError, match="also reference groups"):
        validate_transport(spread_ref, leak, group_key="donor_id")
    with pytest.warns(UserWarning, match="not frozen"):
        ctrl = validate_transport(
            spread_ref,
            leak,
            group_key="donor_id",
            config=TransportConfig(allow_group_overlap=True),
        )
    ind = ctrl.reference["independence"]
    assert ind["shared_groups"] == 2 and ind["allow_group_overlap"]
    assert ind["state_definition"] == "fitted_reference"


def test_spread_scenario_keeps_the_mean(spread_pair):
    _, qry, truth = spread_pair
    assert truth.loc["S2", "truth"] == "spread" and truth.loc["S2", "spread_factor"] == 2.5
    assert truth.loc["S0", "spread_factor"] == 1.0
    Z = qry.obsm["X_latent"]
    s0 = (qry.obs["cell_state"] == "S0").to_numpy()
    s2 = (qry.obs["cell_state"] == "S2").to_numpy()
    assert Z[s2].std(axis=0).mean() > 2.0 * Z[s0].std(axis=0).mean()


@pytest.mark.skipif(HAS_SCXMATCH, reason="scxmatch is installed")
def test_crossmatch_without_scxmatch_names_the_install():
    with pytest.raises(ImportError, match="conda install scxmatch"):
        ts.crossmatch_test(np.zeros((5, 2)), np.ones((5, 2)))


def test_crossmatch_detects_a_shift():
    pytest.importorskip("scxmatch")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(120, 6))
    shifted = ts.crossmatch_test(A, rng.normal(size=(100, 6)) + 1.0, k=20)
    assert shifted["z_score"] < -3 and shifted["p_value"] < 0.01
    small = ts.crossmatch_test(A[:15], rng.normal(size=(12, 6)), k=20)
    assert np.isnan(small["k"]) and 0 <= small["p_value"] <= 1
