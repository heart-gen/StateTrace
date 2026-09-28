import warnings

import numpy as np
import pandas as pd
import pytest

from atlasroute import validate_metadata
from atlasroute.datasets import simulate_reference


def test_missing_key_raises():
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    with pytest.raises(KeyError, match="donor"):
        validate_metadata(a, donor_key="donor", state_key="cell_state")


def test_nan_label_raises():
    a = simulate_reference(n_donors=3, n_cells_per_donor=20)
    a.obs["cell_state"] = a.obs["cell_state"].astype(object)
    a.obs.iloc[0, a.obs.columns.get_loc("cell_state")] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_metadata(a, donor_key="donor_id", state_key="cell_state")


def test_single_donor_raises():
    a = simulate_reference(n_donors=1, n_cells_per_donor=50)
    with pytest.raises(ValueError, match="two donors"):
        validate_metadata(
            a,
            donor_key="donor_id",
            state_key="cell_state",
            min_donors_per_state=1,
            min_cells_per_state=1,
        )


def test_string_columns_are_coerced_to_categorical():
    a = simulate_reference(n_donors=3, n_states=2, n_cells_per_donor=60)
    a.obs["donor_id"] = a.obs["donor_id"].astype(str)
    a.obs["cell_state"] = a.obs["cell_state"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = validate_metadata(
            a,
            donor_key="donor_id",
            state_key="cell_state",
            min_donors_per_state=1,
            min_cells_per_state=1,
        )
    assert isinstance(a.obs["donor_id"].dtype, pd.CategoricalDtype)
    assert isinstance(a.obs["cell_state"].dtype, pd.CategoricalDtype)
    assert rep.n_donors == 3 and rep.n_states == 2 and rep.n_cells == 180
    assert rep.state_table["eligible"].all()
    assert list(rep.state_table.columns) == ["n_cells", "n_donors", "eligible", "reason"]


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
            donor_key="donor_id",
            state_key="cell_state",
            min_donors_per_state=3,
            min_cells_per_state=10,
        )
    assert rec[0].filename == __file__  # stacklevel points at the caller
    assert rep.state_table.loc["S2", "n_donors"] == 1
    assert not rep.state_table.loc["S2", "eligible"]
    assert rep.state_table.loc["S2", "reason"] == "few_donors"
    assert rep.state_table.loc["S0", "eligible"]
    assert len(rep.warnings) == 1
