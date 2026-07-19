import inspect
from src.phase_d.calibration import ProbabilityCalibratedClassifier
def test_calibration_fit_accepts_only_supplied_validation_matrix():
 assert list(inspect.signature(ProbabilityCalibratedClassifier.fit_calibration).parameters)==["self","X_validation","y_validation"]
