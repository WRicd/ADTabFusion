import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.external.horizon_model import train_and_freeze_horizon_model
from src.external.model_freezing import sha256_file

BASE_FEATURES = ["AGE", "MMSE"]


def _pairs(rows=180):
    rng = np.random.default_rng(5)
    labels = np.tile([0, 1, 2], rows // 3)
    splits = np.array(["train"] * rows, dtype=object)
    splits[::5] = "val"
    splits[1::5] = "test"
    return pd.DataFrame(
        {
            "RID": np.arange(rows),
            "AGE": rng.normal(72, 6, rows) + labels * 2,
            "MMSE": rng.normal(27, 2, rows) - labels * 3,
            "forecast_months": np.tile([6, 18, 30, 42], rows // 4),
            "label": labels,
            "split": splits,
        }
    )


def _project(tmp_path, pairs, models_run=("logistic_regression", "random_forest")):
    output = tmp_path / "outputs"
    (output / "audit").mkdir(parents=True)
    pairs.to_csv(output / "audit" / "future_diagnosis_pairs.csv", index=False)
    source_manifest = tmp_path / "primary_model_manifest.json"
    source_manifest.write_text(json.dumps({"feature_order": BASE_FEATURES}), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config = {
        "project": {"output_dir": str(output)},
        "data": {"feature_source_manifest": str(source_manifest)},
        "pairing": {"min_horizon_months": 6, "max_horizon_months": 60},
        "preprocessing": {"numeric_impute": "median", "add_missing_indicators": True},
        "models": {
            "run": list(models_run),
            "logistic_regression": {"C": 1.0, "max_iter": 500},
            "random_forest": {"n_estimators": 25, "random_state": 42},
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path, output, source_manifest


def test_frozen_manifest_records_horizon_task_and_reproducible_hashes(tmp_path):
    config, config_path, output, source_manifest = _project(tmp_path, _pairs())

    manifest = train_and_freeze_horizon_model(config, config_path)

    assert manifest["feature_order"] == [*BASE_FEATURES, "forecast_months"]
    assert manifest["base_feature_order"] == BASE_FEATURES
    assert manifest["forecast_horizon_months"] == [6, 60]
    assert manifest["model_id"] == f"phase_c_horizon_aware_{manifest['model_name']}"
    assert manifest["d4_used_for_training_or_selection"] is False
    assert manifest["selection_data"] == "D1/D2 validation subjects only"
    assert manifest["training_pair_count"] == 180
    assert manifest["training_subject_count"] == 180
    assert manifest["source_manifest_sha256"] == sha256_file(source_manifest)
    assert manifest["config_sha256"] == sha256_file(config_path)
    assert manifest["model_sha256"] == sha256_file(output / "models" / "horizon_aware_pipeline.joblib")
    assert json.loads((output / "manifests" / "horizon_aware_manifest.json").read_text(encoding="utf-8")) == manifest


def test_frozen_pipeline_predicts_with_the_forecast_horizon_feature(tmp_path):
    config, config_path, output, _ = _project(tmp_path, _pairs())

    manifest = train_and_freeze_horizon_model(config, config_path)
    pipeline = joblib.load(output / "models" / "horizon_aware_pipeline.joblib")
    frame = pd.DataFrame([{"AGE": 74.0, "MMSE": 25.0, "forecast_months": 24}])

    probability = pipeline.predict_proba(frame[manifest["feature_order"]])

    assert probability.shape == (1, 3)
    assert probability.sum(axis=1) == pytest.approx([1.0])


def test_metrics_are_written_per_split_and_horizon_stratum(tmp_path):
    config, config_path, output, _ = _project(tmp_path, _pairs())

    train_and_freeze_horizon_model(config, config_path)

    metrics = pd.read_csv(output / "evaluation" / "horizon_aware_internal_metrics.csv")
    assert set(metrics["split"]) == {"val", "test"}
    assert set(metrics["model"]) == {"logistic_regression", "random_forest"}
    assert set(metrics["horizon"]) == {"overall", "0-12 months", "12-24 months", "24-36 months", ">36 months"}
    assert metrics["macro_f1"].notna().all()
    assert "confusion_matrix" in metrics.columns
    assert metrics["confusion_matrix"].map(lambda value: len(json.loads(value)) == 3).all()


def test_empty_horizon_strata_are_skipped_instead_of_reported(tmp_path):
    pairs = _pairs()
    pairs["forecast_months"] = 9
    config, config_path, output, _ = _project(tmp_path, pairs)

    train_and_freeze_horizon_model(config, config_path)

    metrics = pd.read_csv(output / "evaluation" / "horizon_aware_internal_metrics.csv")
    assert set(metrics["horizon"]) == {"overall", "0-12 months"}


def test_model_selection_uses_validation_macro_f1_only(tmp_path):
    config, config_path, output, _ = _project(tmp_path, _pairs())

    manifest = train_and_freeze_horizon_model(config, config_path)

    metrics = pd.read_csv(output / "evaluation" / "horizon_aware_internal_metrics.csv")
    validation = metrics[(metrics["split"] == "val") & (metrics["horizon"] == "overall")]
    best = validation.sort_values(["macro_f1", "balanced_accuracy"], ascending=False).iloc[0]["model"]
    assert manifest["model_name"] == best
    assert manifest["hyperparameters"] == config["models"][best]


def test_missing_split_raises_before_any_model_is_frozen(tmp_path):
    pairs = _pairs()
    pairs["split"] = "train"
    config, config_path, output, _ = _project(tmp_path, pairs)

    with pytest.raises(ValueError, match="non-empty"):
        train_and_freeze_horizon_model(config, config_path)

    assert not (output / "models").exists()
    assert not (output / "manifests").exists()
