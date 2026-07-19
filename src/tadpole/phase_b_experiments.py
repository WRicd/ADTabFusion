from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_schema import MULTICLASS_MAPPING, normalize_diagnosis
from src.evaluation import compute_metrics
from src.models.sklearn_models import predict_model, predict_proba_model
from src.tadpole.phase_b import (
    COMPACT_FEATURES,
    _csv_safe_metrics,
    fit_evaluate_partition,
    load_phase_b_cohort,
    make_subject_partitions,
    select_baseline_records,
    summarize_results,
)


def load_compact_cohort(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    data_cfg = config["data"]
    features = data_cfg.get("features", COMPACT_FEATURES)
    identity = [
        data_cfg.get("subject_col", "RID"),
        data_cfg.get("visit_col", "VISCODE"),
        data_cfg.get("date_col", "EXAMDATE"),
        data_cfg.get("label_col", "DX"),
    ]
    required = list(dict.fromkeys(identity + features))
    frame = pd.read_csv(
        data_cfg["raw_csv"],
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    features = [feature for feature in features if feature in frame.columns]
    frame = select_baseline_records(
        frame,
        subject_col=data_cfg.get("subject_col", "RID"),
        visit_col=data_cfg.get("visit_col", "VISCODE"),
        date_col=data_cfg.get("date_col", "EXAMDATE"),
    )
    frame["diagnosis"] = frame[data_cfg.get("label_col", "DX")].map(normalize_diagnosis)
    frame = frame.dropna(subset=["diagnosis"]).copy()
    frame["label"] = frame["diagnosis"].map(MULTICLASS_MAPPING).astype(int)
    return frame.reset_index(drop=True), features


def run_compact_vs_full(
    compact_config: dict[str, Any],
    full_config: dict[str, Any],
    quick: bool = False,
) -> pd.DataFrame:
    compact, compact_features = load_compact_cohort(compact_config)
    full, full_features, _, _ = load_phase_b_cohort(full_config, persist=False)
    subject_col = full_config["data"].get("subject_col", "RID")
    compact_labels = compact.set_index(subject_col)["label"]
    full_labels = full.set_index(subject_col)["label"]
    common = compact_labels.index.intersection(full_labels.index)
    concordant = [rid for rid in common if compact_labels.loc[rid] == full_labels.loc[rid]]
    compact = compact[compact[subject_col].isin(concordant)].copy()
    full = full[full[subject_col].isin(concordant)].copy()
    if not concordant:
        raise RuntimeError("Compact and full cohorts have no concordant subject intersection.")

    output = Path(full_config["project"].get("output_dir", "outputs/phase_b"))
    seeds = [42] if quick else full_config["project"].get("seed_list", [42])
    models = full_config.get("models", {}).get("run", ["logistic_regression"])
    if quick:
        models = [name for name in models if name in {"logistic_regression", "random_forest"}]
    rows = []
    for seed in seeds:
        partitions = make_subject_partitions(full, seed, subject_col=subject_col)
        for source, frame, features, config in (
            ("compact", compact, compact_features, compact_config),
            ("full_primary", full, full_features, full_config),
        ):
            for model_name in models:
                metrics, _, _, _ = fit_evaluate_partition(
                    frame,
                    features,
                    model_name,
                    config,
                    seed,
                    partitions,
                    add_missing_indicators=True,
                )
                metrics["data_source"] = source
                metrics["intersection_subjects"] = len(concordant)
                rows.append(_csv_safe_metrics(metrics))
    result = pd.DataFrame(rows)
    result.to_csv(output / "compact_vs_full_by_seed.csv", index=False)
    summary = summarize_results(
        result, ["data_source", "model", "task_mode", "missing_indicators"]
    )
    summary.to_csv(output / "compact_vs_full_summary.csv", index=False)
    _write_comparison_report(
        output / "compact_vs_full_report.md",
        summary,
        len(common),
        len(concordant),
        len(common) - len(concordant),
    )
    _plot_grouped(
        summary,
        "data_source",
        "macro_f1_mean",
        output / "figures" / "compact_vs_full.png",
    )
    return result


def run_phase_b_ablation(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    frame, all_features, groups, _ = load_phase_b_cohort(config, persist=False)
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    model_name = config.get("ablation", {}).get("model", "logistic_regression")
    combinations = config.get("ablation", {}).get("groups", {})
    rows = []
    for seed in seeds:
        partitions = make_subject_partitions(frame, seed)
        for group_name, modalities in combinations.items():
            features = []
            for modality in modalities:
                if modality == "pet":
                    features.extend(groups.get("pet_fdg", []))
                    features.extend(groups.get("pet_amyloid", []))
                else:
                    features.extend(groups.get(modality, []))
            features = list(dict.fromkeys(features))
            if not features:
                rows.append(
                    {
                        "group": group_name,
                        "seed": seed,
                        "skipped": "no primary features",
                    }
                )
                continue
            metrics, _, _, _ = fit_evaluate_partition(
                frame,
                features,
                model_name,
                config,
                seed,
                partitions,
                add_missing_indicators=True,
            )
            metrics.update({"group": group_name, "modalities": "|".join(modalities)})
            rows.append(_csv_safe_metrics(metrics))
    result = pd.DataFrame(rows)
    result.to_csv(output / "modality_ablation_by_seed.csv", index=False)
    summary = summarize_results(result, ["group", "modalities"])
    summary.to_csv(output / "modality_ablation_summary.csv", index=False)
    _plot_grouped(
        summary,
        "group",
        "macro_f1_mean",
        output / "figures" / "modality_ablation.png",
    )
    return result


def run_phase_b_missing_modality(
    config: dict[str, Any], quick: bool = False
) -> pd.DataFrame:
    frame, all_features, groups, _ = load_phase_b_cohort(config, persist=False)
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    model_name = config.get("missing_modality", {}).get("model", "logistic_regression")
    masks = config.get("missing_modality", {}).get("mask", [])
    rows = []
    for seed in seeds:
        partitions = make_subject_partitions(frame, seed)
        baseline, pipeline, predictions, _ = fit_evaluate_partition(
            frame,
            all_features,
            model_name,
            config,
            seed,
            partitions,
            add_missing_indicators=True,
        )
        base_f1 = float(baseline["macro_f1"])
        baseline.update(
            {
                "masked_modality": "none",
                "absolute_drop": 0.0,
                "relative_drop": 0.0,
            }
        )
        rows.append(_csv_safe_metrics(baseline))
        subject_col = config["data"].get("subject_col", "RID")
        test = frame[frame[subject_col].isin(partitions["test"])].copy()
        truth = test["label"].to_numpy()
        for mask_name in masks:
            if mask_name == "all_pet":
                mask_columns = groups.get("pet_fdg", []) + groups.get("pet_amyloid", [])
            else:
                mask_columns = groups.get(mask_name, [])
            if not mask_columns:
                rows.append(
                    {"masked_modality": mask_name, "seed": seed, "skipped": "no primary features"}
                )
                continue
            X = test[all_features].copy()
            X.loc[:, mask_columns] = np.nan
            predicted = predict_model(pipeline, X)
            probability = predict_proba_model(pipeline, X)
            metrics = compute_metrics(truth, predicted, probability, labels=[0, 1, 2])
            score = float(metrics["macro_f1"])
            metrics.update(
                {
                    "model": model_name,
                    "seed": seed,
                    "task_mode": config["data"].get("task_mode", "baseline_only"),
                    "missing_indicators": True,
                    "n_features": len(all_features),
                    "masked_modality": mask_name,
                    "absolute_drop": base_f1 - score,
                    "relative_drop": (base_f1 - score) / base_f1 if base_f1 else np.nan,
                }
            )
            rows.append(_csv_safe_metrics(metrics))
    result = pd.DataFrame(rows)
    result.to_csv(output / "missing_modality_results_by_seed.csv", index=False)
    summary = summarize_results(result, ["masked_modality"])
    drop_summary = (
        result.groupby("masked_modality", as_index=False)[["absolute_drop", "relative_drop"]]
        .mean(numeric_only=True)
    )
    summary = summary.merge(drop_summary, on="masked_modality", how="left")
    summary.to_csv(output / "missing_modality_summary.csv", index=False)
    _plot_grouped(
        summary,
        "masked_modality",
        "absolute_drop",
        output / "figures" / "missing_modality_drop.png",
    )
    return result


def build_sparse_modality_cohort(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    modality = config["sparse_cohort"]["modality"]
    catalog = pd.read_csv(data_cfg["feature_catalog"])
    features = catalog.loc[catalog["modality"] == modality, "column_name"].astype(str).tolist()
    if modality == "csf":
        features = [
            feature
            for feature in features
            if feature.upper().startswith(("ABETA", "TAU", "PTAU"))
        ]
    identity = [
        data_cfg.get("subject_col", "RID"),
        data_cfg.get("visit_col", "VISCODE"),
        data_cfg.get("date_col", "EXAMDATE"),
        data_cfg.get("label_col", "DX"),
    ]
    frame = pd.read_csv(
        data_cfg["raw_csv"],
        usecols=lambda column: column in set(identity + features),
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    frame = frame.copy()
    frame["diagnosis"] = frame[data_cfg.get("label_col", "DX")].map(normalize_diagnosis)
    valid = frame.dropna(subset=["diagnosis"]).copy()
    available = valid[features].notna().any(axis=1)
    cohort = valid[available]
    subject_col = data_cfg.get("subject_col", "RID")
    baseline = select_baseline_records(
        valid,
        subject_col=subject_col,
        visit_col=data_cfg.get("visit_col", "VISCODE"),
        date_col=data_cfg.get("date_col", "EXAMDATE"),
    )
    all_subjects = set(baseline[subject_col].unique())
    cohort_subjects = set(cohort[subject_col].unique())
    summary = {
        "modality": modality,
        "feature_count": len(features),
        "available_subjects": len(cohort_subjects),
        "available_visits": len(cohort),
        "main_cohort_subjects": len(all_subjects),
        "main_cohort_intersection": len(cohort_subjects & all_subjects),
        "diagnosis_distribution": cohort["diagnosis"].value_counts().to_dict(),
        "mean_feature_missing_rate_in_cohort": float(cohort[features].isna().mean().mean())
        if not cohort.empty
        else None,
        "modeling_value": _modeling_value(cohort, subject_col),
    }
    output_path = Path(config["paths"]["summary_report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_sparse_summary(summary), encoding="utf-8")
    return summary


def _modeling_value(frame: pd.DataFrame, subject_col: str) -> str:
    subjects = frame[subject_col].nunique()
    classes = frame["diagnosis"].nunique()
    if subjects >= 300 and classes == 3:
        return "Potentially useful as a separate cohort with dedicated validation."
    if subjects >= 100 and classes >= 2:
        return "Exploratory only; cohort size or class coverage limits robust comparison."
    return "Insufficient for a reliable standalone three-class experiment."


def _render_sparse_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['modality']} Cohort Summary",
        "",
        f"Modality fields: {summary['feature_count']}",
        f"Available subjects: {summary['available_subjects']}",
        f"Available visits: {summary['available_visits']}",
        f"Main-cohort subject intersection: {summary['main_cohort_intersection']}",
        f"Mean feature missing rate within selected visits: {summary['mean_feature_missing_rate_in_cohort']}",
        "",
        "## Diagnosis distribution",
        "",
    ]
    lines.extend(
        f"- {diagnosis}: {count} visits"
        for diagnosis, count in summary["diagnosis_distribution"].items()
    )
    lines.extend(["", "## Modeling value", "", summary["modeling_value"], ""])
    return "\n".join(lines)


def _write_comparison_report(
    path: Path,
    summary: pd.DataFrame,
    common_subjects: int,
    concordant_subjects: int,
    label_mismatches: int,
) -> None:
    lines = [
        "# Compact vs Full TADPOLE",
        "",
        f"Common subjects before label agreement filtering: {common_subjects}",
        f"Concordant subjects used: {concordant_subjects}",
        f"Label mismatches excluded: {label_mismatches}",
        "",
        "Both datasets use identical subjects, subject partitions, seeds, models and metrics.",
        "The compact arm projects the frozen 20-feature compact feature set from the full D1/D2 source because the previous 51-column local CSV is no longer present.",
        "",
        "## Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_grouped(frame: pd.DataFrame, x: str, y: str, path: Path) -> None:
    if frame.empty or y not in frame.columns:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(frame[x].astype(str), pd.to_numeric(frame[y], errors="coerce"))
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
