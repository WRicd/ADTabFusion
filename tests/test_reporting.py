import json

import pandas as pd

from src.reporting import analyze_error_cases, generate_report


def test_error_analysis_writes_placeholder_when_predictions_are_absent(tmp_path):
    errors = analyze_error_cases(tmp_path / "missing_predictions.csv", tmp_path)
    summary = (tmp_path / "reports" / "error_case_summary.md").read_text(encoding="utf-8")
    assert errors.empty
    assert "No prediction file was found." in summary


def test_error_analysis_categorizes_and_ranks_by_confidence(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [0, 2, 1, 0, 0],
            "y_pred": [2, 0, 2, 0, 1],
            "max_proba": [0.91, 0.72, 0.55, 0.99, 0.60],
        }
    ).to_csv(predictions, index=False)

    errors = analyze_error_cases(predictions, tmp_path)

    assert errors["max_proba"].tolist() == [0.91, 0.72, 0.60, 0.55]
    assert errors["error_category"].tolist() == [
        "CN_predicted_as_AD",
        "AD_predicted_as_CN",
        "MCI_confusion",
        "MCI_confusion",
    ]
    written = pd.read_csv(tmp_path / "reports" / "error_cases.csv")
    assert len(written) == 4

    confusion = pd.read_csv(tmp_path / "reports" / "error_confusion_summary.csv")
    assert confusion["count"].sum() == 5
    assert confusion["count"].is_monotonic_decreasing

    summary = (tmp_path / "reports" / "error_case_summary.md").read_text(encoding="utf-8")
    assert "Total test rows in best split: 5" in summary
    assert "Misclassified rows: 4" in summary
    assert "- CN_predicted_as_AD: 1" in summary


def test_error_analysis_handles_a_perfect_prediction_file(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]}).to_csv(predictions, index=False)

    errors = analyze_error_cases(predictions, tmp_path)

    assert errors.empty
    assert "error_category" in errors.columns
    summary = (tmp_path / "reports" / "error_case_summary.md").read_text(encoding="utf-8")
    assert "No errors found in the saved best-model prediction file." in summary


def test_generate_report_degrades_gracefully_without_artifacts(tmp_path):
    report_path = tmp_path / "reports" / "final_report.md"
    generate_report(report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "# AD-TabFusion Report" in report
    assert "ADNI raw-file inventory has not been generated." in report
    assert "Leakage check has not been generated." in report
    assert "was not generated." in report


def test_generate_report_embeds_available_metrics_and_inventory(tmp_path):
    metrics = tmp_path / "metrics"
    reports = tmp_path / "reports"
    metrics.mkdir()
    reports.mkdir()
    pd.DataFrame({"model": ["random_forest"], "macro_f1": [0.812]}).to_csv(
        metrics / "baseline_results_summary_baseline_only.csv", index=False
    )
    (reports / "leakage_check.md").write_text("No future diagnosis columns were used.", encoding="utf-8")
    (metrics / "adni_file_inventory.json").write_text(
        json.dumps([{"read_status": "ok"}, {"read_status": "failed"}]), encoding="utf-8"
    )
    (metrics / "adni_modality_availability.json").write_text(
        json.dumps({"cognitive": {"available": True}, "pet": {"available": False}}), encoding="utf-8"
    )

    generate_report(reports / "final_report.md")

    report = (reports / "final_report.md").read_text(encoding="utf-8")
    assert "random_forest" in report
    assert "0.812" in report
    assert "No future diagnosis columns were used." in report
    assert "Scanned CSV files: 2 (1 readable)." in report
    assert "Available categories: cognitive." in report
    assert "Missing categories: pet." in report
