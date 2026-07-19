from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_schema import MULTICLASS_MAPPING, normalize_diagnosis
from src.evaluation import compute_metrics
from src.feature_groups import infer_feature_types
from src.models.sklearn_models import fit_model, predict_model, predict_proba_model
from src.preprocessing import build_preprocessor


FORBIDDEN_FEATURES = {"RID", "DX", "DXCHANGE", "VISCODE"}
SPARSE_MODALITIES = {"mri_dti", "csf", "pet_other"}
COMPACT_FEATURES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "MMSE",
    "ADAS11",
    "ADAS13",
    "CDRSB",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "FAQ_bl",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    "APOE4",
]
SUMMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc_ovr",
    "log_loss",
    "brier_score",
    "train_macro_f1",
    "val_macro_f1",
]


def load_primary_whitelist(path: str | Path, expected_count: int = 69) -> list[str]:
    features = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError("Primary whitelist must be a JSON list of column names.")
    features = list(dict.fromkeys(features))
    if len(features) != expected_count:
        raise ValueError(
            f"Primary whitelist must contain exactly {expected_count} features; found {len(features)}."
        )
    leaked = sorted(FORBIDDEN_FEATURES.intersection(features))
    if leaked:
        raise ValueError(f"Forbidden fields entered the primary whitelist: {', '.join(leaked)}")
    return features


def load_catalog_groups(
    catalog_path: str | Path, features: Iterable[str]
) -> dict[str, list[str]]:
    catalog = pd.read_csv(catalog_path)
    selected = set(features)
    catalog = catalog[catalog["column_name"].isin(selected)]
    groups = {
        str(modality): group["column_name"].astype(str).tolist()
        for modality, group in catalog.groupby("modality", sort=True)
    }
    mapped = {column for columns in groups.values() for column in columns}
    missing = sorted(selected - mapped)
    if missing:
        raise ValueError(f"Whitelist fields missing from feature catalog: {', '.join(missing)}")
    if SPARSE_MODALITIES.intersection(groups):
        raise ValueError("DTI, CSF, or tau PET entered the default primary feature groups.")
    return groups


def select_baseline_records(
    frame: pd.DataFrame,
    subject_col: str = "RID",
    visit_col: str = "VISCODE",
    date_col: str = "EXAMDATE",
) -> pd.DataFrame:
    """Select one baseline row per subject, falling back to earliest exam date."""
    work = frame.copy()
    work["_row_order"] = np.arange(len(work))
    work["_is_baseline"] = (
        work[visit_col].astype(str).str.strip().str.casefold().eq("bl")
        if visit_col in work.columns
        else False
    )
    work["_sort_date"] = (
        pd.to_datetime(work[date_col], errors="coerce")
        if date_col in work.columns
        else pd.NaT
    )
    work["_date_missing"] = work["_sort_date"].isna()
    work = work.sort_values(
        [subject_col, "_is_baseline", "_date_missing", "_sort_date", "_row_order"],
        ascending=[True, False, True, True, True],
    )
    selected = work.drop_duplicates(subject_col, keep="first")
    return selected.drop(
        columns=["_row_order", "_is_baseline", "_sort_date", "_date_missing"],
        errors="ignore",
    )


def load_phase_b_cohort(
    config: dict[str, Any],
    persist: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], dict[str, int]]:
    data_cfg = config["data"]
    features = load_primary_whitelist(data_cfg["feature_whitelist"])
    groups = load_catalog_groups(data_cfg["feature_catalog"], features)
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
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Full TADPOLE table is missing required fields: {', '.join(missing)}")

    subject_col = data_cfg.get("subject_col", "RID")
    task_mode = data_cfg.get("task_mode", "baseline_only")
    source_subjects = int(frame[subject_col].nunique())
    if task_mode == "baseline_only":
        frame = select_baseline_records(
            frame,
            subject_col=subject_col,
            visit_col=data_cfg.get("visit_col", "VISCODE"),
            date_col=data_cfg.get("date_col", "EXAMDATE"),
        )
    elif task_mode != "all_visits":
        raise ValueError(f"Unsupported Phase B task mode: {task_mode}")

    frame["diagnosis"] = frame[data_cfg.get("label_col", "DX")].map(normalize_diagnosis)
    invalid_label_rows = int(frame["diagnosis"].isna().sum())
    frame = frame.dropna(subset=["diagnosis"]).copy()
    frame["label"] = frame["diagnosis"].map(MULTICLASS_MAPPING).astype(int)
    frame = frame.reset_index(drop=True)
    exclusions = {
        "source_subjects": source_subjects,
        "invalid_or_missing_diagnosis_rows": invalid_label_rows,
        "included_subjects": int(frame[subject_col].nunique()),
        "included_visits": int(len(frame)),
    }
    if persist:
        write_cohort_outputs(frame, features, groups, exclusions, config)
    return frame, features, groups, exclusions


def write_cohort_outputs(
    frame: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    exclusions: dict[str, int],
    config: dict[str, Any],
) -> None:
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    output.mkdir(parents=True, exist_ok=True)
    task_mode = config["data"].get("task_mode", "baseline_only")
    subject_col = config["data"].get("subject_col", "RID")
    cohort_path = output / f"cohort_{task_mode}.csv"
    frame[[subject_col, "diagnosis", "label", *features]].to_csv(cohort_path, index=False)
    lines = [
        f"# Full TADPOLE {task_mode.replace('_', ' ').title()} Cohort",
        "",
        f"Subjects: {frame[subject_col].nunique()}",
        f"Visits: {len(frame)}",
        f"Raw whitelist features: {len(features)}",
        "",
        "## Diagnosis distribution",
        "",
        "| Diagnosis | Rows | Subjects |",
        "|---|---:|---:|",
    ]
    for diagnosis, group in frame.groupby("diagnosis", sort=True):
        lines.append(f"| {diagnosis} | {len(group)} | {group[subject_col].nunique()} |")
    lines.extend(["", "## Modality availability", "", "| Modality | Features | Available subjects |", "|---|---:|---:|"])
    for modality, columns in groups.items():
        available = frame[columns].notna().any(axis=1)
        lines.append(
            f"| {modality} | {len(columns)} | {frame.loc[available, subject_col].nunique()} |"
        )
    lines.extend(["", "## Exclusions", ""])
    lines.extend(f"- {key}: {value}" for key, value in exclusions.items())
    lines.extend(["", "## Feature missingness", "", "| Feature | Missing rate |", "|---|---:|"])
    for feature, rate in frame[features].isna().mean().sort_values(ascending=False).items():
        lines.append(f"| `{feature}` | {rate:.3f} |")
    (output / f"cohort_{task_mode}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_subject_partitions(
    frame: pd.DataFrame,
    seed: int,
    subject_col: str = "RID",
    label_col: str = "label",
    test_size: float = 0.20,
    val_size: float = 0.10,
) -> dict[str, list[Any]]:
    subject_labels = frame.groupby(subject_col)[label_col].agg(lambda values: int(values.mode().iloc[0]))
    ids = subject_labels.index.to_numpy()
    labels = subject_labels.to_numpy()
    train_val, test = _safe_split(ids, labels, test_size, seed)
    train_val_labels = subject_labels.loc[train_val].to_numpy()
    train, val = _safe_split(
        train_val,
        train_val_labels,
        val_size / (1.0 - test_size),
        seed,
    )
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def fit_evaluate_partition(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    config: dict[str, Any],
    seed: int,
    partitions: dict[str, list[Any]],
    add_missing_indicators: bool,
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame, list[dict[str, Any]]]:
    subject_col = config["data"].get("subject_col", "RID")
    train = frame[frame[subject_col].isin(partitions["train"])]
    val = frame[frame[subject_col].isin(partitions["val"])]
    test = frame[frame[subject_col].isin(partitions["test"])]
    assert set(train[subject_col]).isdisjoint(test[subject_col])

    numeric, categorical = infer_feature_types(train, features)
    preprocessor = build_preprocessor(
        numeric,
        categorical,
        config.get("preprocessing", {}).get("numeric_impute", "median"),
        add_missing_indicators=add_missing_indicators,
    )
    X_train = preprocessor.fit_transform(train[features])
    model_cfg = copy.deepcopy(config.get("models", {}))
    model_cfg.setdefault(model_name, {})["random_state"] = seed
    estimator = fit_model(model_name, X_train, train["label"].to_numpy(), model_cfg)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    labels = [0, 1, 2]
    metrics_by_split: dict[str, dict[str, Any]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for name, split_frame in (("train", train), ("val", val), ("test", test)):
        truth = split_frame["label"].to_numpy()
        predicted = predict_model(pipeline, split_frame[features])
        probability = predict_proba_model(pipeline, split_frame[features])
        metrics_by_split[name] = compute_metrics(truth, predicted, probability, labels=labels)
        predictions[name] = (truth, predicted, probability)

    result = metrics_by_split["test"]
    result.update(
        {
            "model": model_name,
            "seed": seed,
            "task_mode": config["data"].get("task_mode", "baseline_only"),
            "missing_indicators": add_missing_indicators,
            "n_features": len(features),
            "n_train_subjects": train[subject_col].nunique(),
            "n_val_subjects": val[subject_col].nunique(),
            "n_test_subjects": test[subject_col].nunique(),
            "n_train_visits": len(train),
            "n_val_visits": len(val),
            "n_test_visits": len(test),
            "train_macro_f1": metrics_by_split["train"]["macro_f1"],
            "val_macro_f1": metrics_by_split["val"]["macro_f1"],
        }
    )
    truth, predicted, probability = predictions["test"]
    pred_frame = test[[subject_col, "diagnosis", "label"]].copy()
    pred_frame["y_true"] = truth
    pred_frame["y_pred"] = predicted
    calibration = _calibration_rows(
        truth, probability, model_name, seed, result["task_mode"], add_missing_indicators
    )
    if probability is not None:
        pred_frame["max_proba"] = probability.max(axis=1)
        for index, label in enumerate(labels):
            pred_frame[f"proba_{label}"] = probability[:, index]
    return result, pipeline, pred_frame, calibration


def run_primary_baselines(config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    frame, features, groups, _ = load_phase_b_cohort(config, persist=True)
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    output.mkdir(parents=True, exist_ok=True)
    seeds = [42] if quick else config["project"].get("seed_list", [42])
    models = config.get("models", {}).get("run", ["logistic_regression"])
    if quick:
        models = [name for name in models if name in {"logistic_regression", "random_forest"}]
    indicators = config.get("preprocessing", {}).get("missing_indicator_variants", [False, True])
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    best: tuple[float, Pipeline, pd.DataFrame, dict[str, Any]] | None = None
    for seed in seeds:
        partitions = make_subject_partitions(
            frame,
            seed,
            subject_col=config["data"].get("subject_col", "RID"),
            test_size=config.get("split", {}).get("test_size", 0.2),
            val_size=config.get("split", {}).get("val_size", 0.1),
        )
        _write_partition_summary(frame, partitions, seed, config)
        for add_indicators in indicators:
            for model_name in models:
                try:
                    metrics, pipeline, predictions, calibration = fit_evaluate_partition(
                        frame,
                        features,
                        model_name,
                        config,
                        seed,
                        partitions,
                        bool(add_indicators),
                    )
                except ImportError as exc:
                    rows.append({"model": model_name, "seed": seed, "skipped": str(exc)})
                    continue
                rows.append(_csv_safe_metrics(metrics))
                calibration_rows.extend(calibration)
                score = float(metrics["val_macro_f1"])
                if best is None or score > best[0]:
                    best = (score, pipeline, predictions, metrics)

    results = pd.DataFrame(rows)
    task_mode = config["data"].get("task_mode", "baseline_only")
    _merge_task_output(output / "baseline_results_by_seed.csv", results, task_mode)
    combined = pd.read_csv(output / "baseline_results_by_seed.csv")
    summary = summarize_results(combined, ["model", "task_mode", "missing_indicators"])
    summary.to_csv(output / "baseline_results_summary.csv", index=False)
    calibration_frame = pd.DataFrame(calibration_rows)
    _merge_task_output(output / "calibration_results.csv", calibration_frame, task_mode)
    if best is not None and task_mode == "baseline_only":
        _, pipeline, predictions, metadata = best
        models_dir = Path("outputs/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, models_dir / "phase_b_full_primary_best.joblib")
        predictions.to_csv(output / "best_model_predictions.csv", index=False)
        (output / "best_model.json").write_text(
            json.dumps(_csv_safe_metrics(metadata), indent=2), encoding="utf-8"
        )
        write_phase_b_explainability(
            pipeline, frame, features, groups, predictions, config, quick=quick
        )
    return results


def summarize_results(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available = [metric for metric in SUMMARY_METRICS if metric in frame.columns]
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in available:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_phase_b_explainability(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    predictions: pd.DataFrame,
    config: dict[str, Any],
    quick: bool = False,
) -> None:
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    subject_col = config["data"].get("subject_col", "RID")
    test = frame[frame[subject_col].isin(predictions[subject_col])]
    repeats = 2 if quick else 5
    global_result = permutation_importance(
        pipeline,
        test[features],
        test["label"],
        n_repeats=repeats,
        random_state=42,
        scoring="f1_macro",
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": global_result.importances_mean,
            "importance_std": global_result.importances_std,
        }
    )
    feature_to_modality = {
        feature: modality for modality, columns in groups.items() for feature in columns
    }
    importance["modality"] = importance["feature"].map(feature_to_modality)
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    modality = (
        importance.groupby("modality", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    modality.to_csv(output / "modality_importance.csv", index=False)

    per_class_rows = []
    for class_id, class_name in ((0, "CN"), (1, "MCI"), (2, "AD")):
        def scorer(estimator, X, y, target=class_id):
            return f1_score(y, estimator.predict(X), labels=[target], average="macro", zero_division=0)

        result = permutation_importance(
            pipeline,
            test[features],
            test["label"],
            n_repeats=2 if quick else 3,
            random_state=42 + class_id,
            scoring=scorer,
        )
        for feature, value in zip(features, result.importances_mean):
            per_class_rows.append(
                {"class": class_name, "feature": feature, "importance": float(value)}
            )
    pd.DataFrame(per_class_rows).to_csv(output / "per_class_feature_importance.csv", index=False)

    suspicious = [
        feature
        for feature in importance.head(20)["feature"]
        if feature in FORBIDDEN_FEATURES
        or feature.upper().startswith(("SITE", "COLPROT", "ORIGPROT", "EXAMDATE"))
    ]
    warning_path = output / "EXPLAINABILITY_WARNING"
    if suspicious:
        warning_path.write_text("Suspicious top features: " + ", ".join(suspicious), encoding="utf-8")
    elif warning_path.exists():
        warning_path.unlink()
    errors = predictions[predictions["y_true"] != predictions["y_pred"]].copy()
    errors.to_csv(output / "error_cases.csv", index=False)
    (output / "error_case_summary.md").write_text(
        "# Phase B Error Cases\n\n"
        f"Evaluated rows: {len(predictions)}\n\n"
        f"Misclassified rows: {len(errors)}\n\n"
        f"Error rate: {len(errors) / max(len(predictions), 1):.3f}\n",
        encoding="utf-8",
    )
    _plot_importance(importance.head(20), "feature", output / "figures" / "top_features.png")
    _plot_importance(modality, "modality", output / "figures" / "modality_importance.png")


def _safe_split(ids, labels, size: float, seed: int):
    try:
        return train_test_split(ids, test_size=size, random_state=seed, stratify=labels)
    except ValueError:
        return train_test_split(ids, test_size=size, random_state=seed, stratify=None)


def _calibration_rows(
    truth: np.ndarray,
    probability: np.ndarray | None,
    model: str,
    seed: int,
    task_mode: str,
    indicators: bool,
) -> list[dict[str, Any]]:
    if probability is None:
        return []
    rows = []
    for class_id in range(probability.shape[1]):
        observed = (truth == class_id).astype(float)
        predicted = probability[:, class_id]
        bins = np.minimum((predicted * 10).astype(int), 9)
        for bin_id in range(10):
            mask = bins == bin_id
            if not mask.any():
                continue
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "task_mode": task_mode,
                    "missing_indicators": indicators,
                    "class_id": class_id,
                    "bin": bin_id,
                    "n": int(mask.sum()),
                    "mean_predicted": float(predicted[mask].mean()),
                    "observed_rate": float(observed[mask].mean()),
                }
            )
    return rows


def _csv_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in metrics.items()
    }


def _merge_task_output(path: Path, new: pd.DataFrame, task_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if "task_mode" in old.columns:
            old = old[old["task_mode"] != task_mode]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False)


def _write_partition_summary(
    frame: pd.DataFrame,
    partitions: dict[str, list[Any]],
    seed: int,
    config: dict[str, Any],
) -> None:
    output = Path(config["project"].get("output_dir", "outputs/phase_b"))
    subject_col = config["data"].get("subject_col", "RID")
    summary = {}
    for name, subjects in partitions.items():
        part = frame[frame[subject_col].isin(subjects)]
        summary[name] = {
            "subjects": int(part[subject_col].nunique()),
            "visits": int(len(part)),
            "label_distribution": part["diagnosis"].value_counts().to_dict(),
        }
    (output / f"split_{config['data'].get('task_mode', 'baseline_only')}_seed_{seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _plot_importance(frame: pd.DataFrame, label: str, path: Path) -> None:
    if frame.empty:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plot = frame.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot) * 0.25)))
    ax.barh(plot[label].astype(str), plot["importance"].astype(float))
    ax.set_xlabel("Permutation importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

