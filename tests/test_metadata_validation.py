import warnings

import numpy as np
import pandas as pd
import pytest

from statetrace import validate_metadata
from statetrace.datasets import simulate_reference


def test_missing_key_raises():
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    with pytest.raises(KeyError, match="donor"):
        validate_metadata(a, group_key="donor", state_key="cell_state")


def test_nan_label_raises():
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    a.obs["cell_state"] = a.obs["cell_state"].astype(object)
    a.obs.iloc[0, a.obs.columns.get_loc("cell_state")] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_metadata(a, group_key="donor_id", state_key="cell_state")


def test_single_group_raises():
    a = simulate_reference(n_donors=1, n_cells_per_donor=50)
    with pytest.raises(ValueError, match="two groups"):
        validate_metadata(
            a,
            group_key="donor_id",
            state_key="cell_state",
            min_groups_per_state=1,
            min_cells_per_state=1,
        )


def test_string_columns_are_coerced_and_study_counted():
    a = simulate_reference(n_donors=3, n_states=2, n_cells_per_donor=60)
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = validate_metadata(
            a,
            group_key="donor_id",
            state_key="cell_state",
            study_key="study",
            min_groups_per_state=1,
            min_cells_per_state=1,
        )
    assert isinstance(a.obs["donor_id"].dtype, pd.CategoricalDtype)
    assert rep.n_groups == 3 and rep.n_states == 2 and rep.n_cells == 180 and rep.n_studies == 1
    assert list(rep.state_table.columns) == [
        "n_cells",
        "n_groups",
        "n_studies",
        "eligible",
        "reason",
    ]
    assert (rep.state_table["n_studies"] == 1).all()


def test_threshold_warning_and_eligibility():
    a = simulate_reference(
        n_donors=4,
        n_states=3,
        n_cells_per_donor=100,
        donor_state_dropout={"D00": ["S2"], "D01": ["S2"], "D02": ["S2"]},
        seed=2,
    )
    with pytest.warns(UserWarning, match="S2") as rec:
        rep = validate_metadata(
            a,
            group_key="donor_id",
            state_key="cell_state",
            min_groups_per_state=3,
            min_cells_per_state=10,
        )
    assert rec[0].filename == __file__
    assert rep.state_table.loc["S2", "n_groups"] == 1 and not rep.state_table.loc["S2", "eligible"]
    assert (
        rep.state_table.loc["S2", "reason"] == "few_groups"
        and rep.state_table.loc["S0", "eligible"]
    )
    assert np.isnan(rep.state_table["n_studies"]).all()
