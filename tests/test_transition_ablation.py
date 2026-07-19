from src.phase_d.transition_model import ABLATIONS

def test_required_transition_ablations_are_defined():
    assert set(ABLATIONS)=={"features_only","features_plus_forecast","features_plus_source_dx","features_plus_source_dx_forecast"}
