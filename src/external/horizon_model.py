from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.evaluation import compute_metrics
from src.external.model_freezing import sha256_file, stable_subject_hash
from src.feature_groups import infer_feature_types
from src.models.sklearn_models import fit_model
from src.preprocessing import build_preprocessor


def train_and_freeze_horizon_model(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    output = Path(config["project"]["output_dir"])
    pairs = pd.read_csv(output / "audit" / "future_diagnosis_pairs.csv", low_memory=False)
    source_manifest_path = Path(config["data"]["feature_source_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    base_features = source_manifest["feature_order"]
    features = [*base_features, "forecast_months"]
    train = pairs[pairs["split"] == "train"].copy()
    val = pairs[pairs["split"] == "val"].copy()
    test = pairs[pairs["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Future pair train/val/test splits must all be non-empty.")
    rows = []
    candidates: dict[str, Pipeline] = {}
    for model_name in config["models"].get("run", []):
        pipeline = _fit_pipeline(train, features, model_name, config)
        candidates[model_name] = pipeline
        for split_name, frame in (("val", val), ("test", test)):
            probability = pipeline.predict_proba(frame[features])
            predicted = pipeline.predict(frame[features])
            metrics = compute_metrics(
                frame["label"].to_numpy(), predicted, probability, labels=[0, 1, 2]
            )
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "horizon": "overall",
                    "n_rows": len(frame),
                    **_csv_metrics(metrics),
                }
            )
            rows.extend(_horizon_metric_rows(model_name, split_name, frame, pipeline, features))
    metrics_frame = pd.DataFrame(rows)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(evaluation_dir / "horizon_aware_internal_metrics.csv", index=False)
    selection = metrics_frame[
        (metrics_frame["split"] == "val") & (metrics_frame["horizon"] == "overall")
    ].sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    if selection.empty:
        raise RuntimeError("No horizon-aware candidate produced validation metrics.")
    selected_name = str(selection.iloc[0]["model"])
    final_pipeline = _fit_pipeline(pairs, features, selected_name, config)
    model_dir = output / "models"
    manifest_dir = output / "manifests"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "horizon_aware_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "model_id": f"phase_c_horizon_aware_{selected_name}",
        "model_name": selected_name,
        "hyperparameters": config["models"].get(selected_name, {}),
        "task_definition": "D1/D2 source visit features plus forecast_months to future CN/MCI/AD",
        "feature_list": features,
        "feature_order": features,
        "base_feature_order": base_features,
        "forecast_horizon_months": [
            config["pairing"].get("min_horizon_months", 6),
            config["pairing"].get("max_horizon_months", 60),
        ],
        "training_pair_count": int(len(pairs)),
        "training_subject_count": int(pairs["RID"].nunique()),
        "training_subject_rid_sha256": stable_subject_hash(pairs["RID"]),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(model_path),
        "code_commit_hash": commit,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_data": "D1/D2 validation subjects only",
        "d4_used_for_training_or_selection": False,
    }
    (manifest_dir / "horizon_aware_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _fit_pipeline(
    frame: pd.DataFrame, features: list[str], model_name: str, config: dict[str, Any]
) -> Pipeline:
    numeric, categorical = infer_feature_types(frame, features)
    preprocessor = build_preprocessor(
        numeric,
        categorical,
        config.get("preprocessing", {}).get("numeric_impute", "median"),
        bool(config.get("preprocessing", {}).get("add_missing_indicators", True)),
    )
    transformed = preprocessor.fit_transform(frame[features])
    estimator = fit_model(model_name, transformed, frame["label"].to_numpy(), config["models"])
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _horizon_metric_rows(
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    pipeline: Pipeline,
    features: list[str],
) -> list[dict[str, Any]]:
    bins = [(-float("inf"), 12, "0-12 months"), (12, 24, "12-24 months"),
            (24, 36, "24-36 months"), (36, float("inf"), ">36 months")]
    rows = []
    for low, high, name in bins:
        subset = frame[(frame["forecast_months"] > low) & (frame["forecast_months"] <= high)]
        if subset.empty:
            continue
        probability = pipeline.predict_proba(subset[features])
        metrics = compute_metrics(
            subset["label"].to_numpy(), pipeline.predict(subset[features]), probability, labels=[0, 1, 2]
        )
        rows.append(
            {"model": model_name, "split": split_name, "horizon": name,
             "n_rows": len(subset), **_csv_metrics(metrics)}
        )
    return rows


def _csv_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in metrics.items()
    }
