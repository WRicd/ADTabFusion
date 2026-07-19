import inspect
from src.phase_d.selective_prediction import thresholds_from_validation
def test_threshold_function_uses_only_validation_probabilities_and_coverages():
 assert list(inspect.signature(thresholds_from_validation).parameters)==["probability","coverages"]
