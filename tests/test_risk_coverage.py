import numpy as np
from src.phase_d.selective_prediction import thresholds_from_validation,risk_coverage_rows
def test_risk_coverage_retains_fewer_rows_at_higher_thresholds():
 p=np.array([[.9,.05,.05],[.7,.2,.1],[.4,.3,.3],[.34,.33,.33]]);y=np.array([0,0,1,2]);t=thresholds_from_validation(p,[1,.5]);rows=risk_coverage_rows(y,p,t,"validation");assert rows[0]["coverage"]>=rows[1]["coverage"]
