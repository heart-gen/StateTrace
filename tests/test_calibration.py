import numpy as np

from statetrace import calibration_table
from statetrace.calibration import IsotonicCalibrator, expected_calibration_error


def test_calibration_table_and_isotonic(validated_ref):
    res = validated_ref.validation_
    t = calibration_table(res, bins=5)
    assert len(t) == 5 and list(t.columns[:2]) == ["bin_low", "bin_high"]
    assert t["n_cells"].sum() == res.overall["n_cells"]
    assert (
        t["cell_accuracy"].dropna().between(0, 1).all()
        and t["group_recovery"].dropna().between(0, 1).all()
    )
    ece = expected_calibration_error(t)
    assert 0 <= ece <= 1
    cal = IsotonicCalibrator.from_result(res)
    y = cal.transform(np.linspace(0, 1, 11))
    assert (np.diff(y) >= -1e-9).all() and y.min() >= 0 and y.max() <= 1
