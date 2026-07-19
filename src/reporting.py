from __future__ import annotations

import json
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
    inventory = _read_json(metrics_dir / "adni_file_inventory.json", [])
    availability = _read_json(metrics_dir / "adni_modality_availability.json", {})
    inventory_summary = _format_inventory_summary(inventory, availability)

    lines = [
        "# AD-TabFusion Report",
        "",
        "## 1. Raw ADNI File Inventory",
        inventory_summary,
        "",
        "The inventory only confirms local table availability. Newly detected tables are not merged into the model pipeline at this stage.",
        "",
        "## 2. Dataset Summary",
        "Current model results use the previously prepared TADPOLE_D1_D2 table.",
        "",
        "## 3. Model-Active Modalities",
        "Active in the existing model outputs: demographic, cognitive, MRI-derived, genetic/APOE4.",
        "Raw PET and biofluid files found by the inventory are not yet model features.",
        "",
        "## 4. Leakage Control",
        leakage,
        "",
        "## 5. Task Definitions",
        "Supported task modes are `baseline_only` and `all_visits`, both using subject-level splitting by `RID`.",
        "MCI conversion is skipped because the current CSV has no baseline MCI cohort in `DX_bl`.",
        "",
        "## 6. Model List",
        "Logistic Regression, Random Forest, HistGradientBoosting, and optional XGBoost/LightGBM when installed.",
        "",
        "## 7. Baseline-Only Results",
        baseline_only,
        "",
        "## 8. All-Visit Results",
        all_visits,
        "",
        "## 9. Modality Ablation",
        ablation,
        "",
        "## 10. Missing-Modality Robustness",
        missing,
        "",
        "## 11. Explainability",
        modality_importance,
        "",
        "## 12. Error-Case Analysis",
        f"See `{reports_dir / 'error_case_summary.md'}` and `error_cases.csv`.",
        "",
        "## 13. Limitations",
        "No cross-table ADNI merge, raw-image processing, PET/CSF model experiment, D3 external evaluation, OASIS-3, GNN, Transformer, 3D CNN, cloud deployment, or automatic ADNI download is included in this stage.",
        "",
        "## 14. Next Steps",
        "Confirm the inventory and data dictionary, then design a leakage-safe subject-visit-level master table before adding new modalities to training.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> str:
    if not path.exists():
        return f"`{path}` was not generated."
    df = pd.read_csv(path)
    return "```text\n" + df.to_string(index=False) + "\n```"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _format_inventory_summary(inventory: list, availability: dict) -> str:
    if not inventory and not availability:
        return "ADNI raw-file inventory has not been generated."
    readable = sum(item.get("read_status") != "failed" for item in inventory)
    available = [key for key, value in availability.items() if value.get("available")]
    missing = [key for key, value in availability.items() if not value.get("available")]
    return "\n".join(
        [
            f"Scanned CSV files: {len(inventory)} ({readable} readable).",
            f"Available categories: {', '.join(available) if available else 'none'}.",
            f"Missing categories: {', '.join(missing) if missing else 'none'}.",
            "See `docs/adni_file_inventory.md` for the structure-only file inventory.",
        ]
    )


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
