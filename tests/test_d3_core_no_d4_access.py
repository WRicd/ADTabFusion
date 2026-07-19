import inspect
from src.phase_d.d3_feature_profile import build_d3_feature_profiles


def test_profile_builder_has_no_d4_input():
    assert "d4" not in inspect.signature(build_d3_feature_profiles).parameters
