from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

OUTPUT_DIR = Path("outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures" / "overview"
FIGSIZE = (13.333, 7.5)

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "figures" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str, str]] = []
    if plot_pipeline_overview():
        generated.append(
            (
                "pipeline_overview.png",
                "PPT Page 1: Project workflow",
                "Shows the end-to-end local analysis pipeline.",
            )
        )
    if plot_dataset_modalities_overview():
        generated.append(
            (
                "dataset_modalities_overview.png",
                "PPT Page 2: Dataset and modalities",
                "Summarizes current D1_D2 scale, active modalities, and unavailable data.",
            )
        )
    if plot_model_performance_overview():
        generated.append(
            (
                "model_performance_overview.png",
                "PPT Page 3: Baseline model comparison",
                "Compares Accuracy, Balanced Accuracy, Macro F1, and ROC-AUC OvR.",
            )
        )
    if plot_modality_ablation_overview():
        generated.append(
            (
                "modality_ablation_overview.png",
                "PPT Page 4: Modality ablation",
                "Ranks modality combinations by Macro F1 or Balanced Accuracy.",
            )
        )
    if plot_missing_modality_overview():
        generated.append(
            (
                "missing_modality_overview.png",
                "PPT Page 5: Missing-modality robustness",
                "Shows performance drop after test-time modality masking.",
            )
        )
    if plot_explainability_overview():
        generated.append(
            (
                "explainability_overview.png",
                "PPT Page 6: Explainability",
                "Shows top feature importance and modality-level importance.",
            )
        )
    write_readme(generated)
    print(f"Generated {len(generated)} overview figure(s) in {FIGURE_DIR}.")


def plot_pipeline_overview() -> bool:
    steps = [
        ("01", "D1_D2.csv"),
        ("02", "Leakage\nCheck"),
        ("03", "Subject-level\nSplit"),
        ("04", "Baselines"),
        ("05", "Modality\nAblation"),
        ("06", "Explainability"),
        ("07", "Streamlit\nDashboard"),
    ]
    colors = ["#335c67", "#4f772d", "#577590", "#b56576", "#6d597a", "#c76f36", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    ax.set_title("AD-TabFusion Pipeline Overview", fontsize=24, weight="bold", pad=26)

    card_positions = [
        (0.16, 0.63),
        (0.38, 0.63),
        (0.60, 0.63),
        (0.82, 0.63),
        (0.27, 0.36),
        (0.50, 0.36),
        (0.73, 0.36),
    ]
    card_width = 0.155
    card_height = 0.135

    for (number, title), color, (x, y) in zip(steps, colors, card_positions):
        ax.add_patch(
            plt.Rectangle(
                (x - card_width / 2, y - card_height / 2),
                card_width,
                card_height,
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor=color,
                linewidth=1.9,
                zorder=1,
            )
        )
        ax.text(
            x - card_width / 2 + 0.014,
            y + card_height / 2 - 0.032,
            number,
            ha="left",
            va="center",
            color=color,
            fontsize=11,
            weight="bold",
            transform=ax.transAxes,
            zorder=2,
        )
        ax.text(
            x,
            y - 0.006,
            title,
            ha="center",
            va="center",
            color="#222222",
            fontsize=13,
            weight="bold",
            transform=ax.transAxes,
            zorder=2,
        )

    ax.text(
        0.50,
        0.14,
        "Leakage-safe local workflow for interpretable AD diagnosis prediction",
        ha="center",
        va="center",
        color="#333333",
        fontsize=16,
        transform=ax.transAxes,
    )
    save(fig, "pipeline_overview.png")
    return True


def plot_dataset_modalities_overview() -> bool:
    audit = read_json(REPORTS_DIR / "data_audit.json")
    if not audit:
        warn_missing(REPORTS_DIR / "data_audit.json", "dataset_modalities_overview.png")
        return False

    rows = audit.get("n_rows", 5389)
    subjects = audit.get("n_subjects", 759)
    columns = audit.get("n_columns", "N/A")
    active = ["demographic", "cognitive", "MRI-derived", "APOE4"]
    unavailable = ["PET", "CSF", "D3"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()
    ax.set_title("Dataset and Modality Overview", fontsize=24, weight="bold", pad=20)
    metric_items = [("Rows", rows), ("Subjects", subjects), ("Columns", columns)]
    metric_x = [0.18, 0.5, 0.82]
    for x, (label, value) in zip(metric_x, metric_items):
        ax.text(x, 0.72, str(value), ha="center", fontsize=34, weight="bold", transform=ax.transAxes)
        ax.text(x, 0.63, label, ha="center", fontsize=15, color="#555555", transform=ax.transAxes)

    ax.text(0.08, 0.43, "Active modalities", fontsize=18, weight="bold", transform=ax.transAxes)
    draw_pills(ax, active, 0.08, 0.34, "#2a9d8f")
    ax.text(0.08, 0.20, "Unavailable in current stage", fontsize=18, weight="bold", transform=ax.transAxes)
    draw_pills(ax, unavailable, 0.08, 0.11, "#b56576")
    ax.text(
        0.08,
        0.03,
        "Source: local TADPOLE D1_D2 CSV. No PET/CSF/D3 experiment is run in this stage.",
        fontsize=12,
        color="#555555",
        transform=ax.transAxes,
    )
    save(fig, "dataset_modalities_overview.png")
    return True


def plot_model_performance_overview() -> bool:
    path = METRICS_DIR / "baseline_results_summary.csv"
    df = read_csv(path)
    if df.empty:
        warn_missing(path, "model_performance_overview.png")
        return False
    metrics = [
        ("accuracy_mean", "Accuracy"),
        ("balanced_accuracy_mean", "Balanced Accuracy"),
        ("macro_f1_mean", "Macro F1"),
        ("roc_auc_ovr_mean", "ROC-AUC OvR"),
    ]
    available = [(col, label) for col, label in metrics if col in df.columns]
    if not available or "model" not in df.columns:
        warnings.warn(f"Cannot draw model_performance_overview.png: expected summary columns not found.")
        return False

    plot_df = df.dropna(subset=[col for col, _ in available], how="all").copy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(plot_df))
    width = 0.18
    offsets = [(-1.5 + i) * width for i in range(len(available))]
    colors = ["#2a9d8f", "#577590", "#b56576", "#c76f36"]
    for offset, (col, label), color in zip(offsets, available, colors):
        ax.bar([i + offset for i in x], plot_df[col], width=width, label=label, color=color)
    ax.set_title("Baseline Model Performance", fontsize=22, weight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x), plot_df["model"].astype(str), rotation=18, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "model_performance_overview.png")
    return True


def plot_modality_ablation_overview() -> bool:
    path = METRICS_DIR / "modality_ablation_summary.csv"
    df = read_csv(path)
    if df.empty:
        warn_missing(path, "modality_ablation_overview.png")
        return False
    score_col = "macro_f1_mean" if "macro_f1_mean" in df.columns else "balanced_accuracy_mean"
    if score_col not in df.columns or "group" not in df.columns:
        warnings.warn("Cannot draw modality_ablation_overview.png: expected columns not found.")
        return False
    plot_df = df.dropna(subset=[score_col]).sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(plot_df["group"].astype(str), plot_df[score_col], color="#577590")
    ax.set_title("Modality Ablation Overview", fontsize=22, weight="bold")
    ax.set_xlabel("Macro F1" if score_col == "macro_f1_mean" else "Balanced Accuracy")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "modality_ablation_overview.png")
    return True


def plot_missing_modality_overview() -> bool:
    path = METRICS_DIR / "missing_modality_results.csv"
    df = read_csv(path)
    if df.empty:
        warn_missing(path, "missing_modality_overview.png")
        return False
    if "masked_modality" not in df.columns or "macro_f1" not in df.columns:
        warnings.warn("Cannot draw missing_modality_overview.png: expected columns not found.")
        return False
    summary = df.dropna(subset=["macro_f1"]).groupby("masked_modality", as_index=False)["macro_f1"].mean()
    baseline = summary.loc[summary["masked_modality"].eq("none"), "macro_f1"]
    if baseline.empty:
        warnings.warn("Cannot draw missing_modality_overview.png: baseline 'none' row not found.")
        return False
    summary["macro_f1_drop"] = float(baseline.iloc[0]) - summary["macro_f1"]
    wanted = ["cognitive", "mri_derived", "genetic", "demographic"]
    plot_df = summary[summary["masked_modality"].isin(wanted)].copy()
    if plot_df.empty:
        warnings.warn("Cannot draw missing_modality_overview.png: no requested mask rows found.")
        return False
    display = {
        "cognitive": "Cognitive",
        "mri_derived": "MRI-derived",
        "genetic": "Genetic",
        "demographic": "Demographic",
    }
    plot_df["label"] = plot_df["masked_modality"].map(display)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = ["#b56576" if value >= 0 else "#2a9d8f" for value in plot_df["macro_f1_drop"]]
    ax.bar(plot_df["label"], plot_df["macro_f1_drop"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Missing-Modality Robustness", fontsize=22, weight="bold")
    ax.set_ylabel("Macro F1 Drop vs. No Mask")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "missing_modality_overview.png")
    return True


def plot_explainability_overview() -> bool:
    feature_path = METRICS_DIR / "feature_importance_best_model.csv"
    modality_path = METRICS_DIR / "modality_importance_best_model.csv"
    feature_df = read_csv(feature_path)
    modality_df = read_csv(modality_path)
    if feature_df.empty or modality_df.empty:
        if feature_df.empty:
            warn_missing(feature_path, "explainability_overview.png")
        if modality_df.empty:
            warn_missing(modality_path, "explainability_overview.png")
        return False
    if not {"feature", "importance"}.issubset(feature_df.columns) or not {
        "modality",
        "importance",
    }.issubset(modality_df.columns):
        warnings.warn("Cannot draw explainability_overview.png: expected columns not found.")
        return False

    top = feature_df.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(10)
    modality = modality_df.dropna(subset=["importance"]).sort_values("importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].barh(top["feature"].iloc[::-1].astype(str), top["importance"].iloc[::-1], color="#577590")
    axes[0].set_title("Top 10 Features", fontsize=18, weight="bold")
    axes[0].set_xlabel("Importance")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(modality["modality"].astype(str), modality["importance"], color="#2a9d8f")
    axes[1].set_title("Modality Importance", fontsize=18, weight="bold")
    axes[1].set_xlabel("Aggregated Importance")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Explainability Overview", fontsize=22, weight="bold")
    fig.tight_layout()
    save(fig, "explainability_overview.png")
    return True


def write_readme(generated: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Overview Figures",
        "",
        "Generated for 16:9 PPT presentation slides.",
        "",
        "## Run Command",
        "",
        "```bash",
        "python scripts/generate_overview_figures.py",
        "```",
        "",
        "## Figure Guide",
        "",
    ]
    if generated:
        for filename, ppt_page, purpose in generated:
            lines.extend([f"### `{filename}`", "", f"- {ppt_page}", f"- Purpose: {purpose}", ""])
    else:
        lines.append("No figures were generated because required input files were missing.")
    (FIGURE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def draw_pills(ax, labels: list[str], start_x: float, y: float, color: str) -> None:
    x = start_x
    for label in labels:
        ax.text(
            x,
            y,
            label,
            ha="left",
            va="center",
            color="white",
            fontsize=14,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.18",
                "facecolor": color,
                "edgecolor": "none",
            },
            transform=ax.transAxes,
        )
        x += 0.17 + len(label) * 0.004


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.warn(f"Could not parse JSON file: {path}")
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        warnings.warn(f"CSV file is empty: {path}")
        return pd.DataFrame()


def warn_missing(path: Path, figure_name: str) -> None:
    warnings.warn(f"Skipping {figure_name}: required input not found or empty: {path}")


def save(fig, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
