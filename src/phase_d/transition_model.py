from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.data_schema import MULTICLASS_MAPPING, normalize_diagnosis
from src.evaluation import compute_metrics
from src.external.model_freezing import sha256_file, stable_subject_hash
from src.feature_groups import infer_feature_types
from src.models.sklearn_models import fit_model
from src.preprocessing import build_preprocessor


TRANSITIONS = [f"{source} -> {future}" for source in ("CN", "MCI", "AD") for future in ("CN", "MCI", "AD")]
ABLATIONS = {
    "features_only": lambda features: list(features),
    "features_plus_forecast": lambda features: [*features, "forecast_months"],
    "features_plus_source_dx": lambda features: [*features, "SOURCE_DX"],
    "features_plus_source_dx_forecast": lambda features: [*features, "SOURCE_DX", "forecast_months"],
}


def build_transition_pairs(
    frame: pd.DataFrame,
    features: list[str],
    min_horizon_months: float = 6,
    max_horizon_months: float = 60,
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
    label_col: str = "DX",
) -> pd.DataFrame:
    required = [subject_col, date_col, label_col, *features]
    work = frame[required].copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_dx"] = work[label_col].map(normalize_diagnosis)
    work = work.dropna(subset=[subject_col, "_date", "_dx"]).sort_values([subject_col, "_date"])
    rows = []
    for rid, visits in work.groupby(subject_col, sort=False):
        records = visits.to_dict("records")
        for index, source in enumerate(records[:-1]):
            for future in records[index + 1 :]:
                months = (future["_date"] - source["_date"]).days / 30.4375
                if not min_horizon_months <= months <= max_horizon_months:
                    continue
                row = {
                    "RID": rid, "source_date": source["_date"].date().isoformat(),
                    "future_date": future["_date"].date().isoformat(),
                    "SOURCE_DX": source["_dx"], "FUTURE_DX": future["_dx"],
                    "label": MULTICLASS_MAPPING[future["_dx"]], "forecast_months": float(months),
                }
                row.update({feature: source[feature] for feature in features})
                rows.append(row)
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        counts = pairs.groupby("RID")["RID"].transform("size")
        pairs["subject_weight"] = 1.0 / counts
        pairs["transition"] = pairs["SOURCE_DX"] + " -> " + pairs["FUTURE_DX"]
    return pairs


def load_transition_data(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, list[Any]]]:
    data = config["data"]
    features = json.loads(Path(data["feature_profile"]).read_text(encoding="utf-8"))
    required = list(dict.fromkeys([data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"), *features]))
    frame = pd.read_csv(data["train_csv"], usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    split = json.loads(Path(data["temporal_split"]).read_text(encoding="utf-8"))["splits"]
    pair_cfg = config.get("pairing", {})
    pairs = build_transition_pairs(
        frame, features, pair_cfg.get("min_horizon_months", 6), pair_cfg.get("max_horizon_months", 60),
        data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"),
    )
    rid_text = pairs["RID"].astype(str)
    split_text = {name: {str(value) for value in values} for name, values in split.items()}
    pairs["split"] = "excluded"
    for name, ids in split_text.items():
        pairs.loc[rid_text.isin(ids), "split"] = name
    return pairs[pairs["split"] != "excluded"].reset_index(drop=True), features, split


def train_transition_aware(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    output = Path(config["project"]["output_dir"])
    for name in ("models", "manifests", "internal_validation", "temporal_validation", "figures", "cohorts"):
        (output / name).mkdir(parents=True, exist_ok=True)
    pairs, base_features, split = load_transition_data(config)
    train = pairs[pairs["split"] == "train"].copy()
    validation = pairs[pairs["split"] == "validation"].copy()
    temporal_test = pairs[pairs["split"] == "temporal_test"].copy()
    _write_pair_statistics(pairs, output)
    transition_counts = pairs.groupby(["split", "SOURCE_DX", "FUTURE_DX"]).size().rename("count").reset_index()
    transition_counts.to_csv(output / "internal_validation" / "transition_matrix.csv", index=False)
    ablation_rows = []
    fitted_full: dict[str, Pipeline] = {}
    for ablation, resolver in ABLATIONS.items():
        features = resolver(base_features)
        for model_name in config["models"]["run"]:
            pipeline = _fit_transition_pipeline(train, features, model_name, config)
            metrics = _evaluate(pipeline, validation, features)
            ablation_rows.append({"ablation": ablation, "model": model_name, "split": "validation", "n_rows": len(validation), **_csv(metrics)})
            if ablation == "features_plus_source_dx_forecast":
                fitted_full[model_name] = pipeline
    ablations = pd.DataFrame(ablation_rows)
    full_results = ablations[ablations["ablation"] == "features_plus_source_dx_forecast"].copy()
    complexity = {"logistic_regression": 0, "hist_gradient_boosting": 1, "random_forest": 2}
    full_results["complexity"] = full_results["model"].map(complexity)
    selected_row = full_results.sort_values(
        ["macro_f1", "log_loss", "brier_score", "complexity"], ascending=[False, True, True, True]
    ).iloc[0]
    selected_name = str(selected_row["model"])
    best_by_ablation = (
        ablations.sort_values(["ablation", "macro_f1", "log_loss"], ascending=[True, False, True])
        .drop_duplicates("ablation")
        .copy()
    )
    best_by_ablation["selected_for_ablation"] = True
    best_by_ablation.to_csv(output / "internal_validation" / "transition_ablation.csv", index=False)
    full_results.drop(columns="complexity").to_csv(output / "internal_validation" / "transition_model_results.csv", index=False)
    selected_pipeline = fitted_full[selected_name]
    full_features = ABLATIONS["features_plus_source_dx_forecast"](base_features)
    test_metrics = _evaluate(selected_pipeline, temporal_test, full_features)
    pd.DataFrame([{"model": selected_name, "split": "locked_temporal_test", "n_rows": len(temporal_test), "n_subjects": temporal_test["RID"].nunique(), **_csv(test_metrics)}]).to_csv(
        output / "temporal_validation" / "transition_model_results.csv", index=False
    )
    _prediction_frame(selected_pipeline, validation, full_features).to_csv(output / "internal_validation" / "transition_validation_predictions.csv", index=False)
    _prediction_frame(selected_pipeline, temporal_test, full_features).to_csv(output / "temporal_validation" / "transition_test_predictions.csv", index=False)
    _run_group_kfold(pd.concat([train, validation], ignore_index=True), full_features, selected_name, config, output)
    model_path = output / "models" / "transition_aware_pipeline.joblib"
    joblib.dump(selected_pipeline, model_path)
    manifest = _transition_manifest(selected_name, selected_pipeline, full_features, train, config, config_path, model_path)
    (output / "manifests" / "transition_aware_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_ablation(best_by_ablation, output / "figures" / "transition_ablation.png")
    return {"manifest": manifest, "validation": selected_row.to_dict(), "temporal_test": test_metrics}


def _fit_transition_pipeline(frame: pd.DataFrame, features: list[str], model_name: str, config: dict[str, Any]) -> Pipeline:
    numeric, categorical = infer_feature_types(frame, features)
    preprocessor = build_preprocessor(numeric, categorical, config.get("preprocessing", {}).get("numeric_impute", "median"), bool(config.get("preprocessing", {}).get("add_missing_indicators", True)))
    transformed = preprocessor.fit_transform(frame[features])
    estimator = fit_model(model_name, transformed, frame["label"].to_numpy(), config["models"], frame["subject_weight"].to_numpy())
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _evaluate(pipeline: Pipeline, frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    probability = pipeline.predict_proba(frame[features])
    return compute_metrics(frame["label"].to_numpy(), probability.argmax(axis=1), probability, labels=[0, 1, 2])


def _prediction_frame(pipeline: Pipeline, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    probability = pipeline.predict_proba(frame[features])
    result = frame[["RID", "source_date", "future_date", "SOURCE_DX", "FUTURE_DX", "forecast_months", "label", "subject_weight", *features]].copy()
    result["predicted_label"] = probability.argmax(axis=1)
    for index, name in enumerate(("CN", "MCI", "AD")): result[f"prob_{name}"] = probability[:, index]
    result["confidence"] = probability.max(axis=1)
    return result


def _run_group_kfold(frame: pd.DataFrame, features: list[str], model_name: str, config: dict[str, Any], output: Path) -> None:
    rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, frame["label"], groups=frame["RID"]), 1):
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        pipeline = _fit_transition_pipeline(train, features, model_name, config)
        rows.append({"fold": fold, "model": model_name, "n_subjects": test["RID"].nunique(), **_csv(_evaluate(pipeline, test, features))})
    pd.DataFrame(rows).to_csv(output / "internal_validation" / "transition_groupkfold.csv", index=False)


def _write_pair_statistics(pairs: pd.DataFrame, output: Path) -> None:
    counts = pairs.groupby("RID").size()
    weights = pairs.groupby("RID")["subject_weight"].sum()
    summary = {
        "subjects": int(pairs["RID"].nunique()), "total_pairs": int(len(pairs)),
        "median_pairs_per_subject": float(counts.median()), "maximum_pairs_per_subject": int(counts.max()),
        "effective_weighted_sample_size": float(pairs["subject_weight"].sum()),
        "per_subject_weight_sum_min": float(weights.min()), "per_subject_weight_sum_max": float(weights.max()),
        "future_diagnosis_distribution": pairs["FUTURE_DX"].value_counts().to_dict(),
        "forecast_months": pairs["forecast_months"].describe().to_dict(),
    }
    (output / "cohorts" / "transition_pair_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _transition_manifest(model_name: str, pipeline: Pipeline, features: list[str], train: pd.DataFrame, config: dict[str, Any], config_path: str | Path, model_path: Path) -> dict[str, Any]:
    try: commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    return {
        "model_id": f"phase_d_transition_{model_name}", "model_name": model_name,
        "feature_order": features, "source_label_name": "SOURCE_DX", "target_label_name": "FUTURE_DX",
        "subject_balanced_pair_weights": True, "selection_metric": "validation_macro_f1",
        "training_subject_count": int(train["RID"].nunique()), "training_pair_count": int(len(train)),
        "training_subject_rid_sha256": stable_subject_hash(train["RID"]),
        "feature_profile_sha256": sha256_file(config["data"]["feature_profile"]),
        "temporal_split_sha256": sha256_file(config["data"]["temporal_split"]),
        "config_sha256": sha256_file(config_path), "model_sha256": sha256_file(model_path),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "code_commit_hash": commit, "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "d4_used_for_training_selection_or_calibration": False,
    }


def _csv(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in metrics.items()}


def _plot_ablation(frame: pd.DataFrame, path: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib; matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plot = frame.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh(plot["ablation"], plot["macro_f1"], color=["#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_xlim(0, 1); ax.set_xlabel("Validation Macro F1"); ax.set_title("Transition-Aware Feature Ablation")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
