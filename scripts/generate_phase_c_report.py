from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    root = Path("outputs/phase_c")
    direct = pd.read_csv(root / "evaluation" / "direct_transfer_metrics.csv")
    horizon = pd.read_csv(root / "evaluation" / "horizon_aware_metrics.csv")
    ci = pd.read_csv(root / "evaluation" / "bootstrap_ci.csv")
    decision = json.loads((root / "manifests" / "deployment_decision.json").read_text(encoding="utf-8"))
    primary_manifest = json.loads((root / "manifests" / "primary_model_manifest.json").read_text(encoding="utf-8"))
    horizon_manifest = json.loads((root / "manifests" / "horizon_aware_manifest.json").read_text(encoding="utf-8"))
    matched = pd.read_csv(root / "evaluation" / "d3_d4_matched_rows.csv", low_memory=False)
    _generate_figures(root, direct, horizon, matched)
    direct_main = direct[(direct["scope"] == "first_follow_up") & direct["model_id"].str.contains("primary")].iloc[0]
    horizon_main = horizon[horizon["scope"] == "first_follow_up"].iloc[0]
    lines = [
        "# Phase C External Evaluation", "",
        "## Study Design", "",
        "Phase C separates a current-diagnosis direct-transfer sanity baseline from the main horizon-aware future-diagnosis model. D4 was not used for training, preprocessing, feature selection, hyperparameter selection, or deployment ordering.", "",
        "## Frozen Deployment", "",
        f"- D3 full-whitelist coverage: {decision['full_feature_presence_ratio']:.1%}",
        f"- Primary direct-transfer model: `{primary_manifest['candidate_id']}`",
        f"- Sensitivity model: `{decision['sensitivity_candidate']}`",
        f"- Horizon-aware model selected on D1/D2 validation subjects: `{horizon_manifest['model_name']}`",
        "", "## D3/D4 Cohort", "",
        f"- Matched D4 rows: {len(matched)}",
        f"- Matched subjects: {matched['RID'].nunique()}",
        f"- Median forecast horizon: {matched['forecast_months'].median():.1f} months",
        "", "## Main External Results", "",
        "First follow-up per subject is the main independent-subject analysis.", "",
        "| Model | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC OvR | Log Loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_line("Direct-transfer primary", direct_main),
        _metric_line("Horizon-aware future diagnosis", horizon_main),
        "", "Subject-level 95% confidence intervals are stored in `outputs/phase_c/evaluation/bootstrap_ci.csv` and use 1,000 RID-level resamples with seed 42.",
        "", "## Interpretation", "",
        "The direct-transfer result is only a temporal transfer sanity baseline because it maps D3 current features to a future D4 diagnosis despite being trained on current diagnosis. The horizon-aware model explicitly includes forecast_months and is the main future-diagnosis result.",
        "", "## Audit and Shift Artifacts", "",
        "- D3 schema decision: `outputs/phase_c/audit/d3_schema_report.md`",
        "- D3/D4 matching: `outputs/phase_c/evaluation/d3_d4_matching_report.md`",
        "- Dataset shift: `outputs/phase_c/evaluation/dataset_shift_summary.csv`",
        "- Model manifests: `outputs/phase_c/manifests/`",
        "", "## Limitations", "",
        "D3 lacks genetic and PET fields and does not meet the frozen full-primary deployment threshold. Missing frozen columns are padded with NaN and handled only by training-fitted preprocessors. Horizon subgroups with fewer than 20 rows report sample counts without stable performance estimates.", "",
    ]
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (reports / "phase_c_external_evaluation.md").write_text(text, encoding="utf-8")
    global_reports = Path("outputs/reports")
    global_reports.mkdir(parents=True, exist_ok=True)
    (global_reports / "phase_c_external_evaluation.md").write_text(text, encoding="utf-8")
    print("Generated Phase C report and external-evaluation figures.")


def _metric_line(label: str, row: pd.Series) -> str:
    values = [row.get(name) for name in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr", "log_loss", "brier_score"]]
    return "| " + label + " | " + " | ".join("NA" if pd.isna(value) else f"{float(value):.3f}" for value in values) + " |"


def _generate_figures(root: Path, direct: pd.DataFrame, horizon: pd.DataFrame, matched: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / "figures" / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    true = matched["D4_label"].astype(int).to_numpy()
    pred = matched["direct_primary_predicted_class"].map({"CN": 0, "MCI": 1, "AD": 2}).to_numpy()
    cm = confusion_matrix(true, pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["CN", "MCI", "AD"]); ax.set_yticks(range(3), ["CN", "MCI", "AD"])
    ax.set_xlabel("Predicted Diagnosis"); ax.set_ylabel("D4 Diagnosis"); ax.set_title("External Direct-Transfer Confusion Matrix")
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figures / "confusion_matrix_external.png", dpi=160); plt.close(fig)
    combined = pd.concat([direct.assign(track="Direct transfer"), horizon.assign(track="Horizon aware")])
    plot = combined[(combined["scope"] == "horizon") & combined["macro_f1"].notna()]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    order = ["0-12 months", "12-24 months", "24-36 months", ">36 months"]
    for track, group in plot.groupby("track"):
        if track == "Direct transfer":
            group = group[group["model_id"].str.contains("primary")]
        values = group.set_index("horizon").reindex(order)["macro_f1"]
        ax.plot(order, values, marker="o", linewidth=2, label=track)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro F1"); ax.set_xlabel("Forecast Horizon"); ax.set_title("External Performance by Forecast Horizon"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "performance_by_horizon.png", dpi=160); plt.close(fig)
    d = direct[(direct["scope"] == "first_follow_up") & direct["model_id"].str.contains("primary")].iloc[0]
    h = horizon[horizon["scope"] == "first_follow_up"].iloc[0]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr"]
    x = np.arange(len(metrics)); width = 0.35
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.bar(x - width / 2, [d[m] for m in metrics], width, label="Direct transfer", color="#4c78a8")
    ax.bar(x + width / 2, [h[m] for m in metrics], width, label="Horizon aware", color="#59a14f")
    ax.set_xticks(x, ["Accuracy", "Balanced Accuracy", "Macro F1", "ROC-AUC OvR"]); ax.set_ylim(0, 1)
    ax.set_ylabel("Score"); ax.set_title("Direct Transfer vs Horizon-Aware Future Diagnosis"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "direct_vs_horizon_aware.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
