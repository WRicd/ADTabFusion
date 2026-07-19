import numpy as np
def test_cumulative_risk_formula_is_monotonic():
 hazards=np.array([.1,.2,.05,.4]);risk=1-np.cumprod(1-hazards);assert np.all(np.diff(risk)>=0)
