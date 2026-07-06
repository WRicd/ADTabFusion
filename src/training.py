from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import load_tadpole_csv
from src.data_schema import build_diagnosis_labels, select_task_rows
from src.evaluation import compute_metrics
from src.feature_groups import (
    available_groups,
    columns_for_modalities,
    infer_feature_types,
    write_used_feature_groups,
)
from src.models.sklearn_models import fit_model, predict_model, predict_proba_model
from src.leakage import build_feature_blacklist, remove_blacklisted_features
from src.preprocessing import build_preprocessor
from src.splits import make_subject_split


def load_labeled_data(config: dict[str, Any]) -> tuple[pd.DataFrame, str, dict[str, int]]:
    """Load CSV and construct labels for the configured task."""
    data_cfg = config["data"]
    df = load_tadpole_csv(data_cfg["raw_csv"])
    task_mode = data_cfg.get("task_mode")
    if not task_mode:
        task_mode = "baseline_only" if data_cfg.get("use_baseline_only", False) else "all_visits"
    df = select_task_rows(
        df,
        task_mode=task_mode,
        subject_col=data_cfg.get("subject_col", "RID"),
        visit_col=data_cfg.get("visit_col", "VISCODE"),
    )
    return build_diagnosis_labels(
        df,
        label_col=data_cfg.get("label_col", "DX"),
        task=data_cfg.get("task", "diagnosis_multiclass"),
        output_dir=config["project"].get("output_dir", "outputs"),
    )


def run_single_experiment(
    df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
    model_name: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame]:
    """Fit one model for one seed and return metrics and test-frame predictions."""
    output_dir = config["project"].get("output_dir", "outputs")
    subject_col = config["data"].get("subject_col", "RID")
    split_cfg = config.get("split", {})
    split = make_subject_split(
        df,
        subject_col=subject_col,
        label_col=label_col,
        test_size=split_cfg.get("test_size", 0.2),
        val_size=split_cfg.get("val_size", 0.1),
        seed=seed,
        output_dir=output_dir,
    )

    train_df = df.loc[split["train_idx"]]
    val_df = df.loc[split["val_idx"]]
    test_df = df.loc[split["test_idx"]]
    _write_split_distribution(
        df,
        {"train": train_df, "val": val_df, "test": test_df},
        subject_col,
        config["data"].get("visit_col", "VISCODE"),
        label_col,
        seed,
        Path(output_dir) / "metrics",
    )
    numeric_features, categorical_features = infer_feature_types(df, feature_columns)
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        config.get("preprocessing", {}).get("numeric_impute", "median"),
    )
    estimator = fit_model(
        model_name,
        preprocessor.fit_transform(train_df[feature_columns]),
        train_df[label_col].to_numpy(),
        config.get("models", {}),
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    X_test = test_df[feature_columns]
    y_true = test_df[label_col].to_numpy()
    y_pred = predict_model(pipeline, X_test)
    y_proba = predict_proba_model(pipeline, X_test)
    labels = sorted(df[label_col].unique().tolist())
    _save_eval_figures(
        y_true,
        y_pred,
        y_proba,
        labels,
        Path(output_dir) / "figures",
        f"{model_name}_seed{seed}",
    )
    _save_eval_figures(
        y_true,
        y_pred,
        y_proba,
        labels,
        Path(output_dir) / "figures",
        model_name,
    )
    metrics = compute_metrics(y_true, y_pred, y_proba, labels=labels)
    metrics.update(
        {
            "model": model_name,
            "seed": seed,
            "n_features": len(feature_columns),
            "features": "|".join(feature_columns),
            "task_mode": config.get("data", {}).get("task_mode", "all_visits"),
        }
    )
    pred_df = test_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    if y_proba is not None:
        pred_df["max_proba"] = y_proba.max(axis=1)
        for i, label in enumerate(labels):
            if i < y_proba.shape[1]:
                pred_df[f"proba_{label}"] = y_proba[:, i]
    return metrics, pipeline, pred_df


def default_feature_columns(df: pd.DataFrame, output_dir: str | Path) -> list[str]:
    """Use available configured modalities and persist the resolved groups."""
    groups = available_groups(df)
    write_used_feature_groups(groups, output_dir)
    cols: list[str] = []
    for values in groups.values():
        cols.extend(values)
    return list(dict.fromkeys([col for col in cols if col in df.columns]))


def model_feature_columns(
    df: pd.DataFrame, config: dict[str, Any], output_dir: str | Path
) -> list[str]:
    """Resolve available non-leakage model features."""
    features = default_feature_columns(df, output_dir)
    blacklist = build_feature_blacklist(config, df)
    return remove_blacklisted_features(features, blacklist)


def run_baselines(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    """Run configured baseline models over configured seeds."""
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    df, label_col, mapping = load_labeled_data(config)
    feature_columns = model_feature_columns(df, config, output_dir)
    if not feature_columns:
        raise RuntimeError("No usable feature columns are available.")

    rows = []
    best = None
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    models = config.get("models", {}).get("run", ["logistic_regression"])
    if quick:
        models = [m for m in models if m in {"logistic_regression", "random_forest"}]
    for seed in seeds:
        for model_name in models:
            try:
                metrics, pipeline, pred_df = run_single_experiment(
                    df, label_col, feature_columns, model_name, config, seed
                )
            except ImportError as exc:
                rows.append({"model": model_name, "seed": seed, "skipped": str(exc)})
                continue
            rows.append(metrics)
            score = metrics.get("macro_f1") or 0.0
            if best is None or score > best[0]:
                best = (score, model_name, seed, pipeline, pred_df)

    results = pd.DataFrame(rows)
    task_mode = config.get("data", {}).get("task_mode", "all_visits")
    results.to_csv(output_dir / "metrics" / "baseline_results.csv", index=False)
    results.to_csv(output_dir / "metrics" / "baseline_results_by_seed.csv", index=False)
    results.to_csv(output_dir / "metrics" / f"baseline_results_by_seed_{task_mode}.csv", index=False)
    summary = summarize_seed_results(results, ["model", "task_mode"])
    summary.to_csv(output_dir / "metrics" / "baseline_results_summary.csv", index=False)
    summary.to_csv(output_dir / "metrics" / f"baseline_results_summary_{task_mode}.csv", index=False)
    if best is not None:
        _, model_name, seed, pipeline, pred_df = best
        joblib.dump(pipeline, output_dir / "models" / "best_model.joblib")
        pred_df.to_csv(output_dir / "reports" / "best_model_predictions.csv", index=False)
        meta = {"model": model_name, "seed": seed, "label_mapping": mapping}
        (output_dir / "metrics" / "best_model.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return results


def run_ablation(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    """Run modality group ablations over repeated seeds."""
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df, label_col, _ = load_labeled_data(config)
    rows = []
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    model_name = config.get("ablation", {}).get("model", "logistic_regression")
    blacklist = build_feature_blacklist(config, df)
    for seed in seeds:
        for group_name, modalities in config.get("ablation", {}).get("groups", {}).items():
            cols = remove_blacklisted_features(columns_for_modalities(df, modalities), blacklist)
            if not cols:
                rows.append(
                    {
                        "group": group_name,
                        "modalities": "|".join(modalities),
                        "seed": seed,
                        "task_mode": config.get("data", {}).get("task_mode", "all_visits"),
                        "skipped": "no available columns",
                    }
                )
                continue
            metrics, _, _ = run_single_experiment(df, label_col, cols, model_name, config, seed)
            metrics.update({"group": group_name, "modalities": "|".join(modalities)})
            rows.append(metrics)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "metrics" / "modality_ablation.csv", index=False)
    result.to_csv(output_dir / "metrics" / "modality_ablation_by_seed.csv", index=False)
    summary = summarize_seed_results(result, ["group", "modalities", "task_mode"])
    summary.to_csv(output_dir / "metrics" / "modality_ablation_summary.csv", index=False)
    _plot_bar(summary, "group", "macro_f1_mean", output_dir / "figures" / "modality_ablation_bar.png")
    return result


def run_missing_modality(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    """Train on all modalities and mask one modality at test time."""
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df, label_col, _ = load_labeled_data(config)
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    model_name = config.get("ablation", {}).get("model", "logistic_regression")
    all_cols = model_feature_columns(df, config, output_dir)
    rows = []
    groups = available_groups(df)
    labels = sorted(df[label_col].unique().tolist())
    for seed in seeds:
        baseline_metrics, pipeline, pred_df = run_single_experiment(
            df, label_col, all_cols, model_name, config, seed
        )
        baseline_score = baseline_metrics.get("macro_f1")
        rows.append({"masked_modality": "none", **baseline_metrics, "macro_f1_drop": 0.0})
        test_idx = pred_df.index
        y_true = pred_df["y_true"].to_numpy()
        for modality in config.get("missing_modality", {}).get("mask", []):
            mask_cols = [col for col in groups.get(modality, []) if col in all_cols]
            if not mask_cols:
                rows.append(
                    {
                        "masked_modality": modality,
                        "seed": seed,
                        "task_mode": config.get("data", {}).get("task_mode", "all_visits"),
                        "skipped": "no available columns to mask",
                    }
                )
                continue
            X_masked = df.loc[test_idx, all_cols].copy()
            for col in mask_cols:
                X_masked[col] = np.nan
            y_pred = predict_model(pipeline, X_masked)
            y_proba = predict_proba_model(pipeline, X_masked)
            metrics = compute_metrics(y_true, y_pred, y_proba, labels=labels)
            metrics.update(
                {
                    "model": model_name,
                    "seed": seed,
                    "n_features": len(all_cols),
                    "features": "|".join(all_cols),
                    "task_mode": config.get("data", {}).get("task_mode", "all_visits"),
                }
            )
            drop = None if baseline_score is None else float(baseline_score - metrics.get("macro_f1", 0.0))
            metrics.update({"masked_modality": modality, "macro_f1_drop": drop})
            rows.append(metrics)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "metrics" / "missing_modality_results.csv", index=False)
    plot_df = summarize_seed_results(result, ["masked_modality", "task_mode"])
    if "macro_f1_mean" in plot_df.columns:
        base = plot_df.loc[plot_df["masked_modality"].eq("none"), "macro_f1_mean"]
        if not base.empty:
            plot_df["macro_f1_drop_mean"] = float(base.iloc[0]) - plot_df["macro_f1_mean"]
            _plot_bar(plot_df, "masked_modality", "macro_f1_drop_mean", output_dir / "figures" / "missing_modality_drop.png")
    else:
        _plot_bar(result, "masked_modality", "macro_f1_drop", output_dir / "figures" / "missing_modality_drop.png")
    return result


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if df.empty or y_col not in df.columns:
        return
    plot_df = df.dropna(subset=[y_col])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col].astype(float))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_seed_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize repeated-seed metrics with mean and std columns."""
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr"]
    available_metrics = [metric for metric in metrics if metric in df.columns]
    valid = df.dropna(subset=available_metrics, how="all").copy()
    if valid.empty:
        return pd.DataFrame(columns=group_cols)
    rows: list[dict[str, Any]] = []
    for keys, group in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in available_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_split_distribution(
    full_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    subject_col: str,
    visit_col: str,
    label_col: str,
    seed: int,
    metrics_dir: Path,
) -> None:
    """Save subject, visit, and label distributions for each split."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for split_name, frame in split_frames.items():
        item: dict[str, Any] = {
            "n_rows": int(len(frame)),
            "n_subjects": int(frame[subject_col].nunique()) if subject_col in frame.columns else None,
            "label_distribution": frame[label_col].value_counts().sort_index().to_dict()
            if label_col in frame.columns
            else {},
        }
        if visit_col in frame.columns:
            item["visit_distribution"] = frame[visit_col].astype(str).value_counts().to_dict()
        summary[split_name] = item
    summary["all"] = {
        "n_rows": int(len(full_df)),
        "n_subjects": int(full_df[subject_col].nunique()) if subject_col in full_df.columns else None,
    }
    (metrics_dir / f"split_distribution_seed_{seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _save_eval_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    labels: list[int],
    figure_dir: Path,
    tag: str,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(figure_dir / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, confusion_matrix, roc_curve
        from sklearn.preprocessing import label_binarize
    except ImportError:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figure_dir / f"confusion_matrix_{tag}.png", dpi=160)
    plt.close(fig)

    if y_proba is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=labels[1])
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(y_true, classes=labels)
        for i, label in enumerate(labels):
            if i >= y_proba.shape[1]:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"{label} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / f"roc_curve_{tag}.png", dpi=160)
    plt.close(fig)
