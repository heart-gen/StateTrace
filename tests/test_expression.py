import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import REF_KW

from statetrace import (
    ExpressionConfig,
    ExpressionSpace,
    Predictions,
    Reference,
    TransportConfig,
    build_expression_space,
    validate_transport,
)
from statetrace import expression as ex
from statetrace.datasets import simulate_transport


@pytest.fixture(scope="module")
def shifted_pair():
    """Hard cell: weak separation, strong donor effect, S1 displaced far in the query."""
    return simulate_transport(
        n_ref_groups=8,
        n_query_groups=6,
        n_states=6,
        n_cells_per_group=250,
        n_latent=10,
        n_genes=300,
        donor_effect=1.0,
        state_separation=2.0,
        cohort_effect=0.5,
        shifted_states={"S1": 8.0},
        seed=1,
    )


@pytest.fixture(scope="module")
def shifted_ref(shifted_pair):
    ref_a, _, _ = shifted_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Reference(ref_a.copy(), **REF_KW).fit()
        ref.leave_one_group_out()
    return ref


def test_space_is_built_and_frozen_with_reference(shifted_ref, tmp_path):
    sp = shifted_ref.expression_
    assert isinstance(sp, ExpressionSpace) and shifted_ref.has_expression
    assert sp.n_genes == 300 and sp.mu.shape == (6, 300) and sp.components.shape[1] == 300
    assert sp.rows["n_cells"].min() >= ExpressionConfig().min_cells
    assert (sp.null_stats["n_holdout"] == 8).all() and np.isfinite(sp.null_stats["sd"]).all()
    assert (sp.null["distance"] > 0).all()
    # decomposition: distance^2 = distance_pc^2 + residual^2
    d2 = sp.null["distance_pc"] ** 2 + sp.null["residual"] ** 2
    assert np.allclose(d2, sp.null["distance"] ** 2, atol=1e-4)
    # round trip through the frozen format
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = shifted_ref.freeze(tmp_path / "ref", overwrite=True)
    back = Reference.load(path).fit()
    assert back.has_expression and back.expression_.params == sp.params
    assert np.array_equal(back.expression_.genes, sp.genes)
    assert np.allclose(back.expression_.mu, sp.mu) and np.allclose(back.expression_.sigma, sp.sigma)
    pd.testing.assert_frame_equal(back.expression_.null_stats, sp.null_stats)


def test_reference_without_genes_or_disabled_has_no_space(transport_pair, small_ref):
    ref_a, _, _ = transport_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = Reference(ref_a.copy(), expression=False, **REF_KW).fit()
        cfg = Reference(
            small_ref.copy(), expression=ExpressionConfig(n_genes=10, min_cells=5), **REF_KW
        ).fit()
    assert off.expression_ is None and not off.has_expression
    assert cfg.expression_.n_genes == 10
    with pytest.raises(TypeError, match="ExpressionConfig"):
        Reference(ref_a.copy(), expression="yes", **REF_KW)


def test_shift_detected_in_expression_with_kept_identity(shifted_ref, shifted_pair):
    """A mapper that keeps the identity of the displaced state: expression space sees the shift."""
    _, qry, _ = shifted_pair
    P = np.zeros((qry.n_obs, len(shifted_ref.states_)))
    P[np.arange(qry.n_obs), qry.obs["cell_state"].cat.codes.to_numpy()] = 1.0
    preds = Predictions.from_arrays(
        qry.obs_names,
        qry.obs["cell_state"].astype(str).to_numpy(),
        proba=P,
        states=list(shifted_ref.states_),
        backend="oracle",
    )
    q_nolatent = qry.copy()
    del q_nolatent.obsm["X_latent"]
    route = validate_transport(shifted_ref, q_nolatent, group_key="donor_id", predictions=preds)
    st = route.states
    assert (st["shift_source"] == "expression").all() and st["latent_shift"].isna().all()
    assert st.loc["S1", "expression_shift"] > 4 and st.loc["S1", "shift_level"] == "high"
    assert st.loc["S1", "groups_shifted"] >= 5 and st.loc["S1", "status"] == "shifted"
    stable = st.drop(index="S1")
    assert (stable["expression_shift"] < 2).all() and (stable["shift_level"] == "low").all()
    assert (stable["groups_shifted"] == 0).all()
    e = route.expression
    assert set(e.columns) >= {*ex.METRICS, "shift_z", "shift_p", "in_reference_only"}
    s1 = e[(e["state"] == "S1") & ~e["in_reference_only"]]
    assert len(s1) == 6 and (s1["shift_p"] < 0.2).all()
    g = route.groups
    assert g.loc[g["state"] == "S1", "shifted"].sum() >= 5
    assert not g.loc[g["state"] == "S0", "shifted"].any()
    assert (
        route.reference["expression"]["available"]
        and route.reference["expression"]["feature_retention"] == 1.0
    )


def test_latent_fallback_and_forced_sources(shifted_ref, shifted_pair):
    _, qry, _ = shifted_pair
    lat = validate_transport(
        shifted_ref, qry, group_key="donor_id", config=TransportConfig(shift_source="latent")
    )
    assert (lat.states["shift_source"] == "latent").all()
    assert np.allclose(lat.states["shift"], lat.states["latent_shift"], equal_nan=True)
    assert lat.expression is None
    auto = validate_transport(shifted_ref, qry, group_key="donor_id")
    scored = auto.states["groups_scored_expression"] > 0
    assert (auto.states.loc[scored, "shift_source"] == "expression").all()
    q_nogenes = qry[:, :0].copy()
    with pytest.warns(UserWarning, match="no expression space"):
        forced = validate_transport(
            shifted_ref,
            q_nogenes,
            group_key="donor_id",
            config=TransportConfig(shift_source="expression"),
        )
    assert (forced.states["shift_source"] == "none").all()
    with pytest.raises(ValueError, match="shift_source"):
        TransportConfig(shift_source="both")


def test_missing_genes_warn_and_still_score(shifted_ref, shifted_pair):
    _, qry, _ = shifted_pair
    keep = qry.var_names[: int(0.7 * qry.n_vars)]
    with pytest.warns(UserWarning, match="retained"):
        route = validate_transport(shifted_ref, qry[:, keep].copy(), group_key="donor_id")
    prov = route.reference["expression"]
    assert prov["available"] and abs(prov["feature_retention"] - 0.7) < 0.01
    assert not prov["pc_decomposition"] and route.expression["distance_pc"].isna().all()
    foreign = qry.copy()
    foreign.var_names = pd.Index([f"other_{g}" for g in qry.var_names])
    with pytest.warns(UserWarning, match="none of the reference"):
        none = validate_transport(shifted_ref, foreign, group_key="donor_id")
    assert none.expression is None and not none.reference["expression"]["available"]
    assert (none.states["shift_source"] == "latent").all()
    # a renamed query mapped back through a feature map
    from statetrace import harmonize_features

    renamed = qry.copy()
    renamed.var_names = pd.Index([f"q_{g}" for g in qry.var_names])
    table = pd.DataFrame({"reference": qry.var_names, "query": renamed.var_names})
    fmap = harmonize_features(qry.var_names, renamed.var_names, ortholog_table=table)
    mapped = validate_transport(shifted_ref, renamed, group_key="donor_id", feature_map=fmap)
    assert mapped.reference["expression"]["feature_retention"] == 1.0


def test_pseudobulk_profiles_and_helpers():
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(60, 8)).astype(np.float32)
    g = np.repeat([0, 1, 2], 20)
    s = np.tile(np.repeat([0, 1], 10), 3)
    s[:3] = -1  # excluded cells
    prof, rows = ex.pseudobulk_profiles(X, g, s, min_cells=5)
    assert len(rows) == 6 and rows.loc[0, "n_cells"] == 7
    assert (prof >= 0).all() and prof.shape == (6, 8)
    mean_prof, _ = ex.pseudobulk_profiles(X, g, s, min_cells=5, counts=False)
    assert np.allclose(mean_prof[0], X[3:10].mean(axis=0))
    assert ex.counts_like(X) and not ex.counts_like(X + 0.5)
    Z = rng.normal(size=(9, 8)) + np.array([0, 0, 0, 0, 0, 0, 0, 5.0])
    off = ex.estimate_cohort_offset(Z, np.repeat([0, 1, 2], 3))
    assert off is not None and off[-1] > 3
    assert ex.estimate_cohort_offset(Z, np.repeat([0, 1], [5, 4])) is None
    med, hi, n = ex.summarise_states(
        pd.DataFrame({"state": [0, 0, 1], "shift_z": [1.0, 5.0, np.nan]}), 3, high=4.0
    )
    assert med[0] == 3.0 and hi[0] == 1 and n[0] == 2 and np.isnan(med[1]) and n[1] == 0


def test_tiny_reference_skips_space_with_warning():
    from statetrace.datasets import simulate_reference

    with pytest.warns(UserWarning, match="expression space not built"):
        ref = Reference(
            simulate_reference(n_donors=2, n_states=3, n_cells_per_donor=6), **REF_KW
        ).fit()
    assert ref.expression_ is None


def test_build_space_non_count_input_warns(small_ref):
    X = np.log1p(small_ref.X.toarray())
    with pytest.warns(UserWarning, match="raw counts"):
        sp = build_expression_space(
            X,
            small_ref.var_names,
            small_ref.obs["donor_id"].cat.codes.to_numpy(),
            small_ref.obs["cell_state"].cat.codes.to_numpy(),
            np.asarray(small_ref.obs["cell_state"].cat.categories),
            config=ExpressionConfig(min_cells=5),
        )
    assert sp.params["input"] == "non_count"


def test_holdout_null_does_not_depend_on_n_jobs():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(6), 120)
    s = np.tile(np.repeat(np.arange(4), 30), 6)
    X = rng.poisson(1.0 + s[:, None] * np.linspace(0, 1, 40), size=(len(g), 40)).astype(float)
    prof, rows = ex.pseudobulk_profiles(X, g, s, min_cells=10)
    cfg = ex.ExpressionConfig(n_components=5)
    serial, z1 = ex.holdout_null(prof, rows, 4, cfg, return_profiles=True)
    parallel, z2 = ex.holdout_null(
        prof, rows, 4, ex.ExpressionConfig(n_components=5, n_jobs=2), return_profiles=True
    )
    pd.testing.assert_frame_equal(serial, parallel)
    np.testing.assert_allclose(z1, z2)
    assert len(serial) == len(rows)
