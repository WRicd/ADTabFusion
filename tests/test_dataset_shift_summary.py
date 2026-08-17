import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.external.distribution_shift import _population_stability_index, analyze_d1d2_d3_shift
from src.external.model_freezing import fit_frozen_pipeline

FEATURES = ["AGE", "MMSE", "PTGENDER"]


def _training_frame(rows=120):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "RID": range(rows),
            "VISCODE": ["bl"] * rows,
            "EXAMDATE": pd.date_range("2010-01-01", periods=rows, freq="D").astype(str),
            "DX": ["CN", "MCI", "AD"] * (rows // 3),
            "AGE": rng.normal(72, 6, rows),
            "MMSE": rng.normal(27, 2, rows),
            "PTGENDER": ["Male", "Female"] * (rows // 2),
        }
    )


def _frozen_project(tmp_path, d3_frame):
    output = tmp_path / "outputs"
    (output / "manifests").mkdir(parents=True)
    (output / "models").mkdir(parents=True)
    train = _training_frame()
    train_csv = tmp_path / "train.csv"
    d3_csv = tmp_path / "d3.csv"
    train.to_csv(train_csv, index=False)
    d3_frame.to_csv(d3_csv, index=False)
    (output / "manifests" / "primary_model_manifest.json").write_text(
        json.dumps({"model_id": "frozen_primary", "feature_order": FEATURES}), encoding="utf-8"
    )
    cohort = train.assign(label=[0, 1, 2] * (len(train) // 3))
    pipeline = fit_frozen_pipeline(cohort, FEATURES, "random_forest", {}, add_missing_indicators=True)
    joblib.dump(pipeline, output / "models" / "primary_pipeline.joblib")
    config = {
        "project": {"output_dir": str(output)},
        "data": {"train_csv": str(train_csv), "d3_csv": str(d3_csv)},
    }
    return config, output


def _d3_frame(rows=60):
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "RID": range(1000, 1000 + rows),
            "EXAMDATE": pd.date_range("2016-01-01", periods=rows, freq="D").astype(str),
            "AGE": rng.normal(78, 6, rows),
            "MMSE": rng.normal(27, 2, rows),
            "PTGENDER": ["Male", "Female", "Unknown"] * (rows // 3),
        }
    )
    frame.loc[frame.index[:6], "MMSE"] = np.nan
    return frame


def test_shift_summary_covers_every_frozen_feature_and_prediction_distributions(tmp_path):
    config, output = _frozen_project(tmp_path, _d3_frame())

    result = analyze_d1d2_d3_shift(config)

    features = result[result["kind"].isin(["numeric", "categorical"])]
    assert features["feature"].tolist() == FEATURES
    assert set(result.loc[result["kind"] == "prediction_distribution", "cohort"]) == {"D1_D2_training", "D3"}
    assert (output / "evaluation" / "dataset_shift_summary.csv").exists()
    assert (output / "figures" / "dataset_shift.png").exists()


def test_numeric_features_report_smd_ks_and_psi_while_categoricals_report_unseen_levels(tmp_path):
    config, _ = _frozen_project(tmp_path, _d3_frame())

    result = analyze_d1d2_d3_shift(config).set_index("feature")

    age = result.loc["AGE"]
    assert age["kind"] == "numeric"
    assert age["standardized_mean_difference"] > 0.5
    assert 0.0 <= age["ks_statistic"] <= 1.0
    assert age["psi"] > 0.0
    assert pd.isna(age["unseen_categories"])

    gender = result.loc["PTGENDER"]
    assert gender["kind"] == "categorical"
    assert gender["unseen_categories"] == "Unknown"
    assert pd.isna(gender["ks_statistic"])
    assert pd.isna(gender["psi"])


def test_missing_rate_shift_is_measured_against_the_training_cohort(tmp_path):
    d3 = _d3_frame()
    config, _ = _frozen_project(tmp_path, d3)

    result = analyze_d1d2_d3_shift(config).set_index("feature")

    mmse = result.loc["MMSE"]
    assert mmse["train_missing_rate"] == 0.0
    assert mmse["d3_missing_rate"] == pytest.approx(6 / len(d3))
    assert mmse["missing_rate_shift"] == pytest.approx(mmse["d3_missing_rate"])


def test_absent_d3_columns_are_padded_and_reported_as_fully_missing(tmp_path):
    d3 = _d3_frame().drop(columns=["MMSE"])
    config, _ = _frozen_project(tmp_path, d3)

    result = analyze_d1d2_d3_shift(config).set_index("feature")

    mmse = result.loc["MMSE"]
    assert mmse["d3_missing_rate"] == 1.0
    assert pd.isna(mmse["d3_mean"])
    assert pd.isna(mmse["ks_statistic"])


def test_population_stability_index_is_zero_for_identical_samples():
    sample = np.linspace(0, 1, 200)
    assert _population_stability_index(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)


def test_population_stability_index_grows_with_separation():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 500)
    mild = rng.normal(0.2, 1, 500)
    severe = rng.normal(3, 1, 500)
    assert _population_stability_index(reference, severe) > _population_stability_index(reference, mild) > 0


def test_population_stability_index_guards_tiny_and_constant_samples():
    assert _population_stability_index(np.array([1.0]), np.array([1.0, 2.0])) is None
    assert _population_stability_index(np.array([1.0, 2.0]), np.array([1.0])) is None
    assert _population_stability_index(np.ones(50), np.zeros(50)) == 0.0
