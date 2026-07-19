from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.external.schema_alignment import align_to_frozen_schema


CLASS_NAMES = np.array(["CN", "MCI", "AD"])


def probability_frame(probability: np.ndarray) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError("Expected a three-class probability matrix.")
    if np.any(probability < -1e-12) or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Predicted probabilities must be non-negative and sum to one.")
    clipped = np.clip(probability, 1e-15, 1.0)
    pred = probability.argmax(axis=1)
    return pd.DataFrame(
        {
            "predicted_class": CLASS_NAMES[pred],
            "prob_CN": probability[:, 0],
            "prob_MCI": probability[:, 1],
            "prob_AD": probability[:, 2],
            "prediction_confidence": probability.max(axis=1),
            "prediction_entropy": -(clipped * np.log(clipped)).sum(axis=1),
        }
    )


def predict_with_frozen_pipeline(
    model_path: str | Path,
    manifest_path: str | Path,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    features = manifest["feature_order"]
    missing = [feature for feature in features if feature not in frame.columns]
    aligned = align_to_frozen_schema(frame, features)
    pipeline = joblib.load(model_path)
    probabilities = pipeline.predict_proba(aligned)
    result = probability_frame(probabilities)
    result["model_id"] = manifest["model_id"]
    return result, missing
