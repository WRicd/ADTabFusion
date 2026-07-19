import pandas as pd
import pytest

from dashboard.artifacts import artifact, load_csv


def test_locked_transition_metrics_match_frozen_artifact() -> None:
    frame, error = load_csv(artifact("phase_d/temporal_validation/transition_model_results.csv"))
    assert error is None
    row = frame.loc[frame["split"] == "locked_temporal_test"].iloc[0]
    assert row["macro_f1"] == pytest.approx(0.8813447344437649)
    assert row["roc_auc_ovr"] == pytest.approx(0.9583458817376157)
    assert row["balanced_accuracy"] == pytest.approx(0.8780079167817361)


def test_calibration_and_selective_operating_point_match_frozen_artifacts() -> None:
    calibration, error = load_csv(artifact("phase_d/calibration/calibration_results.csv"))
    assert error is None
    isotonic = calibration.loc[(calibration["split"] == "locked_temporal_test") & (calibration["method"] == "isotonic")].iloc[0]
    assert isotonic["log_loss"] == pytest.approx(0.3518244924825336)
    assert isotonic["brier_score"] == pytest.approx(0.19056389594720433)

    selective, error = load_csv(artifact("phase_d/uncertainty/selective_prediction.csv"))
    assert error is None
    selected = selective.loc[(selective["split"] == "locked_temporal_test") & (selective["selected_validation_threshold"].astype(str).str.lower() == "true")].iloc[0]
    assert selected["coverage"] == pytest.approx(0.8107502799552072)
    assert selected["macro_f1"] == pytest.approx(0.9255959464509834)

