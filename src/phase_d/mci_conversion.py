from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_schema import normalize_diagnosis
from src.feature_groups import infer_feature_types
from src.models.sklearn_models import fit_model
from src.preprocessing import build_preprocessor
from src.external.model_freezing import sha256_file


def select_mci_index_visits(
    frame: pd.DataFrame,
    features: list[str],
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
    label_col: str = "DX",
) -> pd.DataFrame:
    work = frame[[subject_col, date_col, label_col, *features]].copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_dx"] = work[label_col].map(normalize_diagnosis)
    work = work.dropna(subset=[subject_col, "_date", "_dx"]).sort_values([subject_col, "_date"])
    rows = []
    for rid, visits in work.groupby(subject_col, sort=False):
        mci = visits[visits["_dx"] == "MCI"]
        if mci.empty:
            continue
        source = mci.iloc[0]
        future = visits[visits["_date"] > source["_date"]].copy()
        if future.empty:
            continue
        future["months"] = (future["_date"] - source["_date"]).dt.days / 30.4375
        ad = future[future["_dx"] == "AD"]
        row = {
            "RID": rid, "source_date": source["_date"].date().isoformat(),
            "max_followup_months": float(future["months"].max()),
            "first_ad_months": float(ad["months"].min()) if not ad.empty else np.nan,
        }
        row.update({feature: source[feature] for feature in features})
        rows.append(row)
    return pd.DataFrame(rows)


def build_landmark_cohort(
    index_visits: pd.DataFrame,
    horizon_months: float,
    tolerance_months: float = 6,
) -> tuple[pd.DataFrame, dict[str, int]]:
    lower = horizon_months - tolerance_months
    upper = horizon_months + tolerance_months
    work = index_visits.copy()
    converted = work["first_ad_months"].notna() & (work["first_ad_months"] <= upper)
    sufficient_negative = (~converted) & (work["max_followup_months"] >= lower)
    work["converted_to_AD_within_horizon"] = np.nan
    work.loc[converted, "converted_to_AD_within_horizon"] = 1
    work.loc[sufficient_negative, "converted_to_AD_within_horizon"] = 0
    work["horizon_months"] = horizon_months
    work["landmark_lower_months"] = lower
    work["landmark_upper_months"] = upper
    summary = {
        "eligible_subjects": int(work["converted_to_AD_within_horizon"].notna().sum()),
        "converters": int(converted.sum()),
        "non_converters": int(sufficient_negative.sum()),
        "censored_subjects": int(work["converted_to_AD_within_horizon"].isna().sum()),
    }
    eligible = work.dropna(subset=["converted_to_AD_within_horizon"]).copy()
    eligible["label"] = eligible["converted_to_AD_within_horizon"].astype(int)
    return eligible, summary


def build_all_landmarks(config: dict[str, Any]) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list[str]]:
    data = config["data"]
    features = json.loads(Path(data["feature_profile"]).read_text(encoding="utf-8"))
    required = [data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"), *features]
    frame = pd.read_csv(data["train_csv"], usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    index = select_mci_index_visits(frame, features, data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), data.get("label_col", "DX"))
    split = json.loads(Path(data["temporal_split"]).read_text(encoding="utf-8"))["splits"]
    rid_to_split = {str(rid): name for name, values in split.items() for rid in values}
    index["split"] = index["RID"].astype(str).map(rid_to_split)
    cohorts, summaries = {}, []
    tolerance = config["landmarks"].get("tolerance_months", 6)
    for horizon in config["landmarks"]["horizons"]:
        cohort, summary = build_landmark_cohort(index, horizon, tolerance)
        cohorts[int(horizon)] = cohort
        eligible = cohort
        summary.update(
            {
                "horizon_months": horizon,
                "class_prevalence": float(eligible["label"].mean()) if len(eligible) else None,
                "age_mean": float(pd.to_numeric(eligible.get("AGE"), errors="coerce").mean()) if "AGE" in eligible else None,
                "education_mean": float(pd.to_numeric(eligible.get("PTEDUCAT"), errors="coerce").mean()) if "PTEDUCAT" in eligible else None,
                "female_fraction": float(eligible.get("PTGENDER", pd.Series(dtype=object)).astype(str).str.casefold().eq("female").mean()) if "PTGENDER" in eligible else None,
                "feature_missing_rate": float(eligible[features].isna().mean().mean()) if len(eligible) else None,
            }
        )
        summaries.append(summary)
    return cohorts, pd.DataFrame(summaries), features


def train_landmark_models(config: dict[str, Any]) -> pd.DataFrame:
    output = Path(config["project"]["output_dir"])
    for name in ("cohorts", "internal_validation", "temporal_validation", "models", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cohorts, summary, features = build_all_landmarks(config)
    summary.to_csv(output / "cohorts" / "mci_landmark_summary.csv", index=False)
    rows, temporal_rows = [], []
    for horizon, cohort in cohorts.items():
        train, validation, test = (cohort[cohort["split"] == name].copy() for name in ("train", "validation", "temporal_test"))
        candidates = {}
        for model_name in config["models"]["run"]:
            if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
                continue
            pipeline = _fit_binary(train, features, model_name, config)
            candidates[model_name] = pipeline
            metrics = binary_metrics(validation["label"].to_numpy(), pipeline.predict_proba(validation[features])[:, 1])
            rows.append({"horizon_months": horizon, "model": model_name, "split": "validation", "n_subjects": len(validation), **metrics})
        if not candidates:
            continue
        candidate_rows = pd.DataFrame([row for row in rows if row["horizon_months"] == horizon])
        selected = candidate_rows.sort_values(["pr_auc", "balanced_accuracy", "brier_score"], ascending=[False, False, True]).iloc[0]
        name = str(selected["model"])
        pipeline = candidates[name]
        if test["label"].nunique() >= 2:
            metrics = binary_metrics(test["label"].to_numpy(), pipeline.predict_proba(test[features])[:, 1])
            temporal_rows.append({"horizon_months": horizon, "model": name, "split": "locked_temporal_test", "n_subjects": len(test), **metrics})
        model_path = output / "models" / f"mci_landmark_{horizon}m_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        manifest = {
            "model_id": f"phase_d_mci_{horizon}m_{name}", "model_name": name,
            "horizon_months": horizon, "tolerance_months": config["landmarks"].get("tolerance_months", 6),
            "feature_order": features, "training_subject_count": len(train),
            "censoring_rule": "negative requires follow-up through horizon minus tolerance",
            "calibration_data": None, "d4_used_for_selection": False,
            "model_sha256": sha256_file(model_path),
        }
        (output / "manifests" / f"mci_landmark_{horizon}m_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    internal = pd.DataFrame(rows); temporal = pd.DataFrame(temporal_rows)
    internal.to_csv(output / "internal_validation" / "mci_landmark_model_results.csv", index=False)
    temporal.to_csv(output / "temporal_validation" / "mci_risk_metrics.csv", index=False)
    return temporal


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "ppv": float(precision_score(y_true, predicted, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "brier_score": float(np.mean((probability - y_true) ** 2)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1])),
    }


def _fit_binary(frame: pd.DataFrame, features: list[str], model_name: str, config: dict[str, Any]) -> Pipeline:
    numeric, categorical = infer_feature_types(frame, features)
    preprocessor = build_preprocessor(numeric, categorical, config.get("preprocessing", {}).get("numeric_impute", "median"), bool(config.get("preprocessing", {}).get("add_missing_indicators", True)))
    transformed = preprocessor.fit_transform(frame[features])
    model = fit_model(model_name, transformed, frame["label"].to_numpy(), config["models"])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
