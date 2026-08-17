import numpy as np

from src.phase_d.selective_prediction import risk_coverage_rows, thresholds_from_validation


def test_risk_coverage_retains_fewer_rows_at_higher_thresholds():
    p = np.array([[0.9, 0.05, 0.05], [0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.34, 0.33, 0.33]])
    y = np.array([0, 0, 1, 2])
    t = thresholds_from_validation(p, [1, 0.5])
    rows = risk_coverage_rows(y, p, t, "validation")
    assert rows[0]["coverage"] >= rows[1]["coverage"]
