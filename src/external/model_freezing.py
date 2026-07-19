from __future__ import annotations

import hashlib
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

from src.data_schema import MULTICLASS_MAPPING, normalize_diagnosis
from src.feature_groups import infer_feature_types
from src.models.sklearn_models import fit_model
from src.preprocessing import build_preprocessor
from src.tadpole.phase_b import load_catalog_groups, select_baseline_records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_subject_hash(subjects: pd.Series) -> str:
    values = sorted({str(value) for value in subjects.dropna()})
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_baseline_training_cohort(
    train_csv: str | Path,
    features: list[str],
    subject_col: str = "RID",
    visit_col: str = "VISCODE",
    date_col: str = "EXAMDATE",
    label_col: str = "DX",
) -> pd.DataFrame:
    required = list(dict.fromkeys([subject_col, visit_col, date_col, label_col, *features]))
    frame = pd.read_csv(
        train_csv,
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Training table is missing frozen fields: {', '.join(missing)}")
    frame = select_baseline_records(frame, subject_col, visit_col, date_col)
    frame["diagnosis"] = frame[label_col].map(normalize_diagnosis)
    frame = frame.dropna(subset=["diagnosis"]).copy()
    frame["label"] = frame["diagnosis"].map(MULTICLASS_MAPPING).astype(int)
    return frame.reset_index(drop=True)


def fit_frozen_pipeline(
    cohort: pd.DataFrame,
    features: list[str],
    model_name: str,
    model_config: dict[str, Any],
    add_missing_indicators: bool,
    numeric_impute: str = "median",
) -> Pipeline:
    numeric, categorical = infer_feature_types(cohort, features)
    preprocessor = build_preprocessor(
        numeric,
        categorical,
        numeric_impute,
        add_missing_indicators=add_missing_indicators,
    )
    transformed = preprocessor.fit_transform(cohort[features])
    estimator = fit_model(model_name, transformed, cohort["label"].to_numpy(), model_config)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def freeze_direct_transfer_models(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    data = config["data"]
    output = Path(config["project"]["output_dir"])
    model_dir = output / "models"
    manifest_dir = output / "manifests"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    full_features = json.loads(Path(data["whitelist"]).read_text(encoding="utf-8"))
    compact_features = list(config["candidates"]["compact"]["features"])
    all_features = list(dict.fromkeys([*full_features, *compact_features]))
    cohort = load_baseline_training_cohort(
        data["train_csv"],
        all_features,
        data.get("subject_col", "RID"),
        data.get("visit_col", "VISCODE"),
        data.get("date_col", "EXAMDATE"),
        data.get("label_col", "DX"),
    )
    groups = load_catalog_groups(data["feature_catalog"], full_features)
    compact_map = _compact_modality_mapping(compact_features)
    decision_path = output / "audit" / "d3_modality_coverage.json"
    if not decision_path.exists():
        raise FileNotFoundError("Run audit_d3_schema.py before freezing Phase C models.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))["deployment"]
    candidates = {
        "full_hist_gradient_boosting": {
            "features": full_features,
            "groups": groups,
            "model": config["candidates"]["full"]["model"],
            "missing_indicators": bool(config["candidates"]["full"]["missing_indicators"]),
        },
        "compact_random_forest": {
            "features": compact_features,
            "groups": compact_map,
            "model": config["candidates"]["compact"]["model"],
            "missing_indicators": bool(config["candidates"]["compact"]["missing_indicators"]),
        },
    }
    assignments = {
        "primary": decision["primary_candidate"],
        "sensitivity": decision["sensitivity_candidate"],
    }
    manifests = {}
    for role, candidate_id in assignments.items():
        spec = candidates[candidate_id]
        pipeline = fit_frozen_pipeline(
            cohort,
            spec["features"],
            spec["model"],
            config.get("models", {}),
            spec["missing_indicators"],
            config.get("preprocessing", {}).get("numeric_impute", "median"),
        )
        model_path = model_dir / f"{role}_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        manifest = _build_manifest(
            role,
            candidate_id,
            spec,
            cohort,
            data,
            config,
            config_path,
            model_path,
        )
        manifest_path = manifest_dir / f"{role}_model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests[role] = manifest
    (manifest_dir / "deployment_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return manifests


def _build_manifest(
    role: str,
    candidate_id: str,
    spec: dict[str, Any],
    cohort: pd.DataFrame,
    data: dict[str, Any],
    config: dict[str, Any],
    config_path: str | Path,
    model_path: Path,
) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    feature_to_modality = {
        feature: modality for modality, features in spec["groups"].items() for feature in features
    }
    return {
        "model_id": f"phase_c_direct_{role}_{candidate_id}",
        "deployment_role": role,
        "candidate_id": candidate_id,
        "model_name": spec["model"],
        "hyperparameters": config.get("models", {}).get(spec["model"], {}),
        "task_definition": "D1/D2 baseline current diagnosis CN/MCI/AD; direct temporal-transfer baseline",
        "feature_list": spec["features"],
        "feature_order": spec["features"],
        "modality_mapping": feature_to_modality,
        "missing_indicators": spec["missing_indicators"],
        "training_subject_count": int(cohort[data.get("subject_col", "RID")].nunique()),
        "training_subject_rid_sha256": stable_subject_hash(cohort[data.get("subject_col", "RID")]),
        "whitelist_sha256": sha256_file(data["whitelist"]),
        "config_sha256": sha256_file(config_path),
        "model_sha256": sha256_file(model_path),
        "code_commit_hash": commit,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "d4_used_for_training_or_selection": False,
    }


def _compact_modality_mapping(features: list[str]) -> dict[str, list[str]]:
    groups = {
        "demographic": ["AGE", "PTGENDER", "PTEDUCAT"],
        "cognitive": [
            "MMSE", "ADAS11", "ADAS13", "CDRSB", "RAVLT_immediate",
            "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting", "FAQ_bl",
        ],
        "mri_structural": [
            "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV",
        ],
        "genetic": ["APOE4"],
    }
    return {name: [feature for feature in values if feature in features] for name, values in groups.items()}
