from src.phase_d.calibration import ProbabilityCalibratedClassifier
def test_calibrator_has_no_test_fit_method():
 assert not hasattr(ProbabilityCalibratedClassifier,"fit_test")
