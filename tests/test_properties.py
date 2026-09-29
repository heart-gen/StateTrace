import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as hs

from statetrace import StabilityThresholds, metrics, schema
from statetrace.datasets import simulate_reference

labels = hs.sampled_from(["S0", "S1", "S2"])


@hs.composite
def prediction_tables(draw):
    n = draw(hs.integers(min_value=1, max_value=60))
    groups = draw(hs.lists(hs.sampled_from(["d1", "d2", "d3"]), min_size=n, max_size=n))
    true = draw(hs.lists(labels, min_size=n, max_size=n))
    pred = draw(
        hs.lists(hs.sampled_from(["S0", "S1", "S2", schema.UNKNOWN_LABEL]), min_size=n, max_size=n)
    )
    df = pd.DataFrame({"group": groups, "true_state": true, "predicted_state": pred})
    df["abstained"] = df["predicted_state"] == schema.UNKNOWN_LABEL
    df["fold"] = df["group"]
    return df


@given(prediction_tables())
@settings(max_examples=60, deadline=None)
def test_metric_invariants(df):
    dm = metrics.group_state_table(df, min_group_cells=1)
    assert ((dm["recall"] >= 0) & (dm["recall"] <= 1)).all() and dm["precision"].dropna().between(
        0, 1
    ).all()
    ds = metrics.group_summary(df)
    assert ds["accuracy"].between(0, 1).all()
    st_ = metrics.state_table(dm, df, states=["S0", "S1", "S2"], thresholds=StabilityThresholds())
    assert set(st_["reproducibility"]) <= set(schema.STABILITY_CATEGORIES)
    assert metrics.confusion_matrix(df, states=["S0", "S1", "S2"]).to_numpy().sum() == len(df)
    assert (
        metrics.overall_metrics(df, ds)["accuracy"]
        == (df["true_state"] == df["predicted_state"]).mean()
    )


@given(
    n_donors=hs.integers(2, 6),
    n_states=hs.integers(2, 5),
    n_cells=hs.integers(5, 40),
    seed=hs.integers(0, 10_000),
)
@settings(max_examples=25, deadline=None)
def test_generator_invariants(n_donors, n_states, n_cells, seed):
    a = simulate_reference(
        n_donors=n_donors,
        n_states=n_states,
        n_cells_per_donor=n_cells,
        n_genes=0,
        n_latent=4,
        seed=seed,
    )
    assert a.n_obs == n_donors * n_cells and a.obs["donor_id"].nunique() == n_donors
    assert (
        a.obs["cell_state"].cat.categories.size == n_states
        and np.isfinite(a.obsm["X_latent"]).all()
    )
    b = simulate_reference(
        n_donors=n_donors,
        n_states=n_states,
        n_cells_per_donor=n_cells,
        n_genes=0,
        n_latent=4,
        seed=seed,
    )
    np.testing.assert_array_equal(a.obsm["X_latent"], b.obsm["X_latent"])
