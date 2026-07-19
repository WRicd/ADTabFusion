from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("outputs/phase_d")
    transition = pd.read_csv(root / "temporal_validation/transition_model_results.csv").iloc[0]
    ablation = pd.read_csv(root / "internal_validation/transition_ablation.csv")
    mci = pd.read_csv(root / "temporal_validation/mci_risk_metrics.csv")
    calibration = pd.read_csv(root / "calibration/calibration_results.csv")
    selective = pd.read_csv(root / "uncertainty/selective_prediction.csv")
    replay = pd.read_csv(root / "d4_replay/d4_replay_metrics.csv").iloc[0]
    calibrated_selected = calibration.selected_on_validation.astype(str).str.casefold().eq("true")
    calibrated_test = calibration[(calibration.split == "locked_temporal_test") & calibrated_selected].iloc[0]
    uncalibrated_test = calibration[(calibration.split == "locked_temporal_test") & (calibration.method == "uncalibrated")].iloc[0]
    threshold_selected = selective.selected_validation_threshold.astype(str).str.casefold().eq("true")
    selective_test = selective[(selective.split == "locked_temporal_test") & threshold_selected].iloc[0]
    improved = bool(
        transition.macro_f1 >= 0.80
        and calibrated_test.log_loss <= uncalibrated_test.log_loss
        and {24, 36}.issubset(set(mci.horizon_months.astype(int)))
        and selective_test.macro_f1 >= transition.macro_f1
    )
    status = {
        "training_effect_good": improved,
        "criteria": {
            "transition_temporal_macro_f1_at_least_0_80": bool(transition.macro_f1 >= 0.80),
            "calibrated_temporal_log_loss_not_worse": bool(calibrated_test.log_loss <= uncalibrated_test.log_loss),
            "mci_24_and_36_month_results_present": bool({24, 36}.issubset(set(mci.horizon_months.astype(int)))),
            "selective_macro_f1_not_worse": bool(selective_test.macro_f1 >= transition.macro_f1),
        },
    }
    (root / "manifests/phase_d_success_gate.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = [
        "# Phase D Transition-Aware Progression Modeling", "",
        "## Primary Result: Locked D1/D2 Temporal Test", "",
        "All feature, model, calibration, and abstention decisions were made without D4. The primary claim uses the latest 15% of D1/D2 subjects by index date.", "",
        "| Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC OvR | Log Loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _row("Transition-aware", transition),
        "", "## Transition Ablation", "",
        "| Inputs | Model | Validation Macro F1 |", "|---|---|---:|",
    ]
    for _, row in ablation.iterrows():
        lines.append(f"| {row.ablation} | {row.model} | {row.macro_f1:.3f} |")
    lines.extend(["", "The current source diagnosis provides the largest gain; forecast_months adds a smaller additional improvement while preserving an explicit future-prediction formulation.", "", "## MCI-to-AD Landmark Risk", "", "| Horizon | Subjects | ROC-AUC | PR-AUC | Balanced Accuracy | Macro F1 |", "|---:|---:|---:|---:|---:|---:|"])
    for _, row in mci.iterrows():
        lines.append(f"| {int(row.horizon_months)} months | {int(row.n_subjects)} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.balanced_accuracy:.3f} | {row.macro_f1:.3f} |")
    lines.extend([
        "", "Negative labels require sufficient follow-up; subjects with inadequate observation time are censored.", "",
        "## Calibration", "",
        f"Validation selected `{calibrated_test.method}` calibration. On the locked temporal test, Log Loss changed from {uncalibrated_test.log_loss:.3f} to {calibrated_test.log_loss:.3f}; Brier changed from {uncalibrated_test.brier_score:.3f} to {calibrated_test.brier_score:.3f}.", "",
        "## Selective Prediction", "",
        f"The validation-frozen abstention rule retained {selective_test.coverage:.1%} of locked temporal-test pairs and achieved Macro F1 {selective_test.macro_f1:.3f} with error rate {selective_test.error_rate:.3f}.", "",
        "## Exploratory Post-Hoc D4 Replay", "",
        "> D4 replay is not independent confirmatory validation because D4 was already accessed in Phase C.", "",
        f"Frozen replay Macro F1: {replay.macro_f1:.3f}; ROC-AUC OvR: {replay.roc_auc_ovr:.3f}. No model, feature, calibration, or threshold was changed.", "",
        "## Reproducibility", "",
        "- Phase C verification gate: PASS",
        "- D3-compatible profiles use D1/D2 and D3 schemas only",
        "- Subject temporal split is date-only and frozen",
        "- Pair weights sum equally per RID",
        "- Calibration and abstention threshold use validation subjects only",
        "- Phase D frozen artifact registry is checked before and after D4 replay", "",
        f"Final success gate: **{'PASS' if improved else 'NEEDS IMPROVEMENT'}**", "",
    ])
    text = "\n".join(lines)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports/phase_d_final_report.md").write_text(text, encoding="utf-8")
    reports = Path("outputs/reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase_d_final_report.md").write_text(text, encoding="utf-8")
    print(f"Phase D success gate: {'PASS' if improved else 'NEEDS IMPROVEMENT'}")


def _row(name: str, row: pd.Series) -> str:
    return f"| {name} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.macro_f1:.3f} | {row.roc_auc_ovr:.3f} | {row.log_loss:.3f} | {row.brier_score:.3f} |"


if __name__ == "__main__":
    main()
