import json

import pandas as pd

from src.phase_d.error_analysis import generate_transition_error_analysis


def _write_fixture(root, threshold=0.8):
    (root / "temporal_validation").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "SOURCE_DX": ["MCI", "MCI", "CN", "AD", "MCI", "CN"],
            "FUTURE_DX": ["AD", "MCI", "MCI", "MCI", "AD", "CN"],
            "predicted_label": [1, 2, 0, 1, 2, 0],
            "confidence": [0.95, 0.90, 0.85, 0.70, 0.60, 0.99],
            "forecast_months": [12, 24, 36, 12, 24, 36],
        }
    )
    predictions.to_csv(root / "temporal_validation" / "transition_test_predictions.csv", index=False)
    (root / "manifests" / "selective_prediction_manifest.json").write_text(
        json.dumps({"threshold": threshold}), encoding="utf-8"
    )
    return predictions


def test_required_error_groups_are_counted(tmp_path):
    _write_fixture(tmp_path)

    counts = generate_transition_error_analysis(tmp_path)

    assert counts == {
        "MCI_to_AD_missed_conversions": 1,
        "MCI_to_MCI_false_progression": 1,
        "CN_progression_missed": 1,
        "AD_non_AD_reversions": 1,
    }


def test_high_confidence_errors_are_filtered_by_the_frozen_threshold(tmp_path):
    _write_fixture(tmp_path, threshold=0.8)

    generate_transition_error_analysis(tmp_path)

    high = pd.read_csv(tmp_path / "reports" / "high_confidence_errors.csv")
    assert high["confidence"].tolist() == [0.95, 0.9, 0.85]
    assert (high["PREDICTED_DX"] != high["FUTURE_DX"]).all()


def test_markdown_report_records_counts_and_threshold(tmp_path):
    _write_fixture(tmp_path, threshold=0.8)

    generate_transition_error_analysis(tmp_path)

    report = (tmp_path / "reports" / "transition_error_analysis.md").read_text(encoding="utf-8")
    assert "- Locked temporal-test predictions: 6" in report
    assert "- Errors: 3" in report
    assert "- Validation-frozen confidence threshold: 0.8000" in report
    assert "- High-confidence errors: 3" in report
    assert "| MCI to AD missed conversions | 1 |" in report
    assert "### By Forecast Horizon" in report
