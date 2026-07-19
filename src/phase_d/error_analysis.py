from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_transition_error_analysis(output_root: str | Path = "outputs/phase_d") -> dict[str, int]:
    root = Path(output_root)
    predictions = pd.read_csv(root / "temporal_validation" / "transition_test_predictions.csv", low_memory=False)
    threshold = json.loads((root / "manifests" / "selective_prediction_manifest.json").read_text(encoding="utf-8"))["threshold"]
    label_to_dx = {0: "CN", 1: "MCI", 2: "AD"}
    predictions["PREDICTED_DX"] = predictions["predicted_label"].map(label_to_dx)
    predictions["feature_missing_rate"] = predictions.isna().mean(axis=1)
    errors = predictions[predictions["PREDICTED_DX"] != predictions["FUTURE_DX"]].copy()
    groups = {
        "MCI_to_AD_missed_conversions": ((predictions.SOURCE_DX == "MCI") & (predictions.FUTURE_DX == "AD") & (predictions.PREDICTED_DX != "AD")),
        "MCI_to_MCI_false_progression": ((predictions.SOURCE_DX == "MCI") & (predictions.FUTURE_DX == "MCI") & (predictions.PREDICTED_DX == "AD")),
        "CN_progression_missed": ((predictions.SOURCE_DX == "CN") & (predictions.FUTURE_DX.isin(["MCI", "AD"])) & (predictions.PREDICTED_DX == "CN")),
        "AD_non_AD_reversions": ((predictions.SOURCE_DX == "AD") & (predictions.FUTURE_DX != "AD")),
    }
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    high = errors[errors["confidence"] >= threshold].sort_values("confidence", ascending=False)
    reports = root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    high.to_csv(reports / "high_confidence_errors.csv", index=False)
    lines = [
        "# Transition Error Analysis", "",
        f"- Locked temporal-test predictions: {len(predictions)}",
        f"- Errors: {len(errors)}",
        f"- Validation-frozen confidence threshold: {threshold:.4f}",
        f"- High-confidence errors: {len(high)}", "",
        "## Required Error Groups", "", "| Group | Rows |", "|---|---:|",
    ]
    lines.extend(f"| {name.replace('_', ' ')} | {count} |" for name, count in counts.items())
    lines.extend(["", "## Error Distribution", "", "### By Future Class", "", errors["FUTURE_DX"].value_counts().to_string(), "", "### By Source Diagnosis", "", errors["SOURCE_DX"].value_counts().to_string(), "", "### By Forecast Horizon", "", errors["forecast_months"].describe().to_string(), "", "### Confidence", "", errors["confidence"].describe().to_string(), ""])
    (reports / "transition_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return counts
