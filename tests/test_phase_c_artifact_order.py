from pathlib import Path


def test_phase_c_models_predate_external_evaluation():
    models = list(Path("outputs/phase_c/models").glob("*.joblib"))
    evaluations = [Path("outputs/phase_c/evaluation/direct_transfer_metrics.csv"), Path("outputs/phase_c/evaluation/horizon_aware_metrics.csv")]
    assert models and all(path.exists() for path in evaluations)
    assert max(path.stat().st_mtime for path in models) <= min(path.stat().st_mtime for path in evaluations)
