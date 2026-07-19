from src.phase_d.verification import verify_phase_c_artifacts


def test_phase_c_predictions_reproduce_from_frozen_artifacts():
    result = verify_phase_c_artifacts()
    assert result["passed"], [item for item in result["checks"] if not item["passed"]]
