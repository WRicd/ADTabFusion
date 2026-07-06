from __future__ import annotations

from pathlib import Path

import pandas as pd


def analyze_error_cases(
    predictions_csv: str | Path,
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Write high-confidence error cases and confusion-category summary."""
    output = Path(output_dir)
    report_dir = output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = Path(predictions_csv)
    if not path.exists():
        (report_dir / "error_case_summary.md").write_text(
            "# Error case analysis\n\nNo prediction file was found.\n",
            encoding="utf-8",
        )
        return pd.DataFrame()
    df = pd.read_csv(path)
    errors = df[df["y_true"] != df["y_pred"]].copy()
    if "max_proba" in errors.columns:
        errors = errors.sort_values("max_proba", ascending=False)
    if errors.empty:
        errors["error_category"] = pd.Series(dtype="object")
    else:
        errors["error_category"] = errors.apply(_error_category, axis=1)
    errors.to_csv(report_dir / "error_cases.csv", index=False)
    confusion = (
        df.groupby(["y_true", "y_pred"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    confusion.to_csv(report_dir / "error_confusion_summary.csv", index=False)
    category_counts = errors["error_category"].value_counts().to_dict()
    lines = [
        "# Error case analysis",
        "",
        f"Total test rows in best split: {len(df)}",
        f"Misclassified rows: {len(errors)}",
        "",
        "## Error Categories",
        "",
    ]
    if category_counts:
        lines.extend([f"- {name}: {count}" for name, count in category_counts.items()])
    else:
        lines.append("No errors found in the saved best-model prediction file.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- High-confidence errors: `outputs/reports/error_cases.csv`",
            "- Per-class confusion summary: `outputs/reports/error_confusion_summary.csv`",
            "",
            "Only local `RID` is retained; no additional identifiable information is exported.",
        ]
    )
    (report_dir / "error_case_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return errors


def generate_report(output_path: str | Path = "outputs/reports/final_report.md") -> None:
    """Generate a compact final markdown report from available outputs."""
    output_path = Path(output_path)
    output_root = output_path.parents[1] if len(output_path.parents) > 1 else Path("outputs")
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_only = _read_csv(metrics_dir / "baseline_results_summary_baseline_only.csv")
    all_visits = _read_csv(metrics_dir / "baseline_results_summary_all_visits.csv")
    ablation = _read_csv(metrics_dir / "modality_ablation.csv")
    missing = _read_csv(metrics_dir / "missing_modality_results.csv")
    leakage = (reports_dir / "leakage_check.md").read_text(encoding="utf-8") if (reports_dir / "leakage_check.md").exists() else "Leakage check has not been generated."
    modality_importance = _read_csv(metrics_dir / "modality_importance_best_model.csv")

    lines = [
        "# AD-TabFusion Report",
        "",
        "## 1. Dataset Summary",
        "Current implementation uses the local TADPOLE_D1_D2.csv file placed under `data/raw/tadpole/`.",
        "",
        "## 2. Available and Unavailable Modalities",
        "Active modalities: demographic, cognitive, MRI-derived, genetic/APOE4.",
        "Unavailable in current D1_D2 CSV: PET (`FDG`, `AV45`, `PIB`), CSF (`ABETA`, `TAU`, `PTAU`), D3 external test data.",
        "",
        "## 3. Leakage Control",
        leakage,
        "",
        "## 4. Task Definitions",
        "Supported task modes are `baseline_only` and `all_visits`, both using subject-level splitting by `RID`.",
        "MCI conversion is skipped because the current CSV has no baseline MCI cohort in `DX_bl`.",
        "",
        "## 5. Model List",
        "Logistic Regression, Random Forest, HistGradientBoosting, and optional XGBoost/LightGBM when installed.",
        "",
        "## 6. Baseline-Only Results",
        baseline_only,
        "",
        "## 7. All-Visit Results",
        all_visits,
        "",
        "## 8. Modality Ablation",
        ablation,
        "",
        "## 9. Missing-Modality Robustness",
        missing,
        "",
        "## 10. Explainability",
        modality_importance,
        "",
        "## 11. Error-Case Analysis",
        f"See `{reports_dir / 'error_case_summary.md'}` and `error_cases.csv`.",
        "",
        "## 12. Limitations",
        "No raw MRI, PET/CSF experiments, D3 external evaluation, OASIS-3, GNN, Transformer, 3D CNN, cloud deployment, or automatic ADNI download is included in this stage.",
        "",
        "## 13. Next Steps",
        "Add full TADPOLE/ADNI variables, especially PET/CSF, optional D3 external test data, and a safe MCI conversion cohort.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> str:
    if not path.exists():
        return f"`{path}` was not generated."
    df = pd.read_csv(path)
    return "```text\n" + df.to_string(index=False) + "\n```"


def _error_category(row: pd.Series) -> str:
    true_label = int(row["y_true"])
    pred_label = int(row["y_pred"])
    if true_label == 0 and pred_label == 2:
        return "CN_predicted_as_AD"
    if true_label == 2 and pred_label == 0:
        return "AD_predicted_as_CN"
    if true_label == 1 or pred_label == 1:
        return "MCI_confusion"
    return "other_error"
