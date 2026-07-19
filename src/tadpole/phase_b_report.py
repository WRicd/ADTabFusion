from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_phase_b_report(output_path: str | Path) -> None:
    phase_b = Path("outputs/phase_b")
    warning = phase_b / "EXPLAINABILITY_WARNING"
    if warning.exists():
        raise RuntimeError(
            "Final report generation stopped because explainability found suspicious fields: "
            + warning.read_text(encoding="utf-8")
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    whitelist = json.loads(
        Path("outputs/phase_a/primary_whitelist.json").read_text(encoding="utf-8")
    )
    baseline = _read_csv(phase_b / "baseline_results_summary.csv")
    comparison = _read_csv(phase_b / "compact_vs_full_summary.csv")
    ablation = _read_csv(phase_b / "modality_ablation_summary.csv")
    missing = _read_csv(phase_b / "missing_modality_summary.csv")
    importance = _read_csv(phase_b / "feature_importance.csv")
    modality_importance = _read_csv(phase_b / "modality_importance.csv")
    required = [
        phase_b / "cohort_baseline_only_summary.md",
        phase_b / "cohort_all_visits_summary.md",
        phase_b / "baseline_results_by_seed.csv",
        phase_b / "compact_vs_full_summary.csv",
        phase_b / "modality_ablation_summary.csv",
        phase_b / "missing_modality_summary.csv",
        phase_b / "feature_importance.csv",
    ]
    missing_outputs = [str(path) for path in required if not path.exists()]
    seeds = set()
    by_seed = _read_csv(phase_b / "baseline_results_by_seed.csv")
    if not by_seed.empty and "seed" in by_seed:
        seeds = set(pd.to_numeric(by_seed["seed"], errors="coerce").dropna().astype(int))
    ready = not missing_outputs and len(seeds) >= 5

    lines = [
        "# Phase B Full TADPOLE Report",
        "",
        "## 1. Phase A audit",
        "",
        "- Full columns audited: 1,907",
        "- Exact dictionary matches: 1,888",
        f"- Frozen primary whitelist: {len(whitelist)} features",
        "- Maximum missingness threshold: 0.70",
        "- DTI, CSF and tau PET remain excluded from the default primary model.",
        "",
        "## 2. Primary whitelist",
        "",
        ", ".join(f"`{feature}`" for feature in whitelist),
        "",
        "## 3. Baseline-only cohort",
        "",
        _read_text(phase_b / "cohort_baseline_only_summary.md"),
        "",
        "## 4. All-visits cohort",
        "",
        _read_text(phase_b / "cohort_all_visits_summary.md"),
        "",
        "## 5. Multi-model results and missing indicators",
        "",
        _markdown_table(baseline),
        "",
        "Missing-indicator variants are reported separately. All imputers, scalers and encoders were fit on training subjects only.",
        "",
        "## 6. Compact vs full comparison",
        "",
        _markdown_table(comparison),
        "",
        _comparison_conclusion(comparison),
        "",
        "## 7. Modality ablation",
        "",
        _markdown_table(ablation),
        "",
        "## 8. Missing-modality robustness",
        "",
        _markdown_table(missing),
        "",
        "## 9. Calibration",
        "",
        "Log loss and multiclass Brier score are included in the baseline summary; bin-level calibration data are stored in `outputs/phase_b/calibration_results.csv`.",
        "",
        "## 10. Explainability",
        "",
        "### Top features",
        "",
        _markdown_table(importance.head(20)),
        "",
        "### Modality importance",
        "",
        _markdown_table(modality_importance),
        "",
        "Per-class permutation importance is stored in `outputs/phase_b/per_class_feature_importance.csv`.",
        "",
        "## 11. Error cases",
        "",
        _read_text(phase_b / "error_case_summary.md"),
        "",
        "## 12. Sparse modality cohorts",
        "",
        _read_text(phase_b / "sparse_modalities" / "dti_cohort_summary.md"),
        "",
        _read_text(phase_b / "sparse_modalities" / "csf_cohort_summary.md"),
        "",
        _read_text(phase_b / "sparse_modalities" / "tau_pet_cohort_summary.md"),
        "",
        "## 13. Limitations",
        "",
        "- The primary cohort excludes highly sparse DTI, CSF and tau PET features.",
        "- Sparse modality cohort results are not directly comparable with the primary cohort.",
        "- Internal validation remains subject to ADNI/TADPOLE cohort and measurement-selection bias.",
        "- Dashboard work and dashboard tests were intentionally deferred for this development round.",
        "",
        "## 14. Phase C readiness",
        "",
        "Ready for model freezing and D3/D4 external evaluation."
        if ready
        else "Not ready for D3/D4 external evaluation; required Phase B outputs are incomplete.",
    ]
    if missing_outputs:
        lines.extend(["", "Missing outputs:", "", *[f"- `{item}`" for item in missing_outputs]])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_text(path: Path) -> str:
    if not path.exists():
        return f"_Pending: `{path}`_"
    text = path.read_text(encoding="utf-8").strip()
    return "\n".join(text.splitlines()[2:]) if text.startswith("# ") else text


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_Pending._"
    view = frame.head(max_rows).copy()
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in view.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _comparison_conclusion(summary: pd.DataFrame) -> str:
    if summary.empty or "macro_f1_mean" not in summary:
        return "Comparison conclusion pending."
    means = summary.groupby("data_source")["macro_f1_mean"].mean()
    if not {"compact", "full_primary"}.issubset(means.index):
        return "Comparison conclusion pending."
    difference = means["full_primary"] - means["compact"]
    direction = "improves" if difference > 0 else "does not improve"
    return (
        f"Across reported models, the full primary feature set {direction} mean Macro F1 "
        f"by {difference:+.4f} versus the compact feature set on the identical subject cohort."
    )

