from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_schema import normalize_diagnosis
from src.evaluation import compute_metrics
from src.external.d3_cohort import select_d3_index_records
from src.external.model_freezing import sha256_file
from src.external.schema_alignment import align_to_frozen_schema


def replay_phase_d_on_d4(output_root: str | Path = "outputs/phase_d") -> dict[str, Any]:
    root = Path(output_root)
    registry_path = root / "manifests" / "phase_d_frozen_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("Freeze Phase D artifacts before D4 replay.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _assert_registry_hashes(registry)
    manifest = json.loads((root / "manifests" / "calibrated_transition_manifest.json").read_text(encoding="utf-8"))
    threshold = json.loads((root / "manifests" / "selective_prediction_manifest.json").read_text(encoding="utf-8"))["threshold"]
    model = joblib.load(root / "models" / "calibrated_transition_pipeline.joblib")
    matched = pd.read_csv("outputs/phase_c/evaluation/d3_d4_matched_rows.csv", low_memory=False)
    base_features = [feature for feature in manifest["feature_order"] if feature not in {"SOURCE_DX", "forecast_months"}]
    required = ["RID", "EXAMDATE", "DX", *base_features]
    d3_raw = pd.read_csv("data/tadpole_challenge/TADPOLE_D3.csv", usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    d3, _ = select_d3_index_records(d3_raw)
    d3["SOURCE_DX"] = d3["DX"].map(normalize_diagnosis)
    replay = matched.merge(d3[["RID", "SOURCE_DX", *base_features]], on="RID", how="left", suffixes=("", "_d3"))
    for feature in base_features:
        duplicate = f"{feature}_d3"
        if duplicate in replay.columns: replay[feature] = replay[duplicate]
    aligned = align_to_frozen_schema(replay, manifest["feature_order"])
    probability = model.predict_proba(aligned); predicted = probability.argmax(axis=1)
    metrics = compute_metrics(replay["D4_label"].to_numpy(dtype=int), predicted, probability, labels=[0,1,2])
    confidence = probability.max(axis=1); retained = confidence >= threshold
    selective = compute_metrics(replay.loc[retained, "D4_label"].to_numpy(dtype=int), predicted[retained], probability[retained], labels=[0,1,2]) if retained.any() else {}
    output = root / "d4_replay"; output.mkdir(parents=True, exist_ok=True)
    predictions = replay[["evaluation_row_id", "RID", "D3_EXAMDATE", "D4_SCANDATE", "forecast_months", "D4_DIAGNOSIS"]].copy()
    predictions["predicted_class"] = np.array(["CN","MCI","AD"])[predicted]
    predictions[["prob_CN","prob_MCI","prob_AD"]] = probability
    predictions["confidence"] = confidence; predictions["abstained"] = ~retained
    predictions["evaluation_label"] = "exploratory post-hoc D4 replay"
    predictions.to_csv(output / "d4_replay_predictions.csv", index=False)
    rows = [
        {"analysis": "all", "evaluation_label": "exploratory post-hoc D4 replay", "n_rows": len(replay), "coverage": 1.0, **_csv(metrics)},
        {"analysis": "validation_frozen_selective", "evaluation_label": "exploratory post-hoc D4 replay", "n_rows": int(retained.sum()), "coverage": float(retained.mean()), **_csv(selective)},
    ]
    pd.DataFrame(rows).to_csv(output / "d4_replay_metrics.csv", index=False)
    _assert_registry_hashes(registry)
    reports = root / "reports"; reports.mkdir(parents=True, exist_ok=True)
    lines = ["# Exploratory Post-Hoc D4 Replay", "", "> This is not independent confirmatory validation. D4 had already been accessed in Phase C.", "", "No model, feature, calibration, or abstention threshold was changed during replay.", "", f"Rows: {len(replay)}", f"Macro F1: {metrics['macro_f1']:.3f}", f"ROC-AUC OvR: {metrics['roc_auc_ovr']:.3f}", f"Selective coverage: {retained.mean():.1%}", ""]
    (reports / "d4_replay_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"metrics": metrics, "selective": selective, "coverage": float(retained.mean())}


def _assert_registry_hashes(registry: dict[str, Any]) -> None:
    for item in registry["artifacts"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen artifact changed: {item['path']}")


def _csv(values: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in values.items()}
