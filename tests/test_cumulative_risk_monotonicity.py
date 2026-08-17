import numpy as np


def test_cumulative_risk_formula_is_monotonic():
    hazards = np.array([0.1, 0.2, 0.05, 0.4])
    risk = 1 - np.cumprod(1 - hazards)
    assert np.all(np.diff(risk) >= 0)
