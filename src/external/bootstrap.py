from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics


BOOTSTRAP_METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr", "log_loss", "brier_score"
]


def resample_subject_blocks(
    frame: pd.DataFrame, rng: np.random.Generator, subject_col: str = "RID"
) -> pd.DataFrame:
    subjects = frame[subject_col].dropna().unique()
    sampled = rng.choice(subjects, size=len(subjects), replace=True)
    blocks = []
    for sample_index, rid in enumerate(sampled):
        block = frame[frame[subject_col] == rid].copy()
        block["_bootstrap_subject"] = sample_index
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else frame.iloc[0:0].copy()


def subject_bootstrap_ci(
    frame: pd.DataFrame,
    truth_col: str,
    prediction_col: str,
    probability_cols: list[str],
    subject_col: str = "RID",
    repetitions: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    point = _metrics(frame, truth_col, prediction_col, probability_cols)
    rng = np.random.default_rng(seed)
    subject_values = frame[subject_col].dropna().unique()
    subject_array = frame[subject_col].to_numpy()
    subject_indices = [np.flatnonzero(subject_array == rid) for rid in subject_values]
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(repetitions):
        chosen = rng.integers(0, len(subject_indices), size=len(subject_indices))
        row_indices = np.concatenate([subject_indices[index] for index in chosen])
        sampled = frame.iloc[row_indices]
        metrics = _metrics(sampled, truth_col, prediction_col, probability_cols)
        for metric in BOOTSTRAP_METRICS:
            value = metrics.get(metric)
            if value is not None and np.isfinite(value):
                samples[metric].append(float(value))
    alpha = (1.0 - confidence_level) / 2.0
    rows: list[dict[str, Any]] = []
    for metric in BOOTSTRAP_METRICS:
        values = samples[metric]
        rows.append(
            {
                "metric": metric,
                "estimate": point.get(metric),
                "ci_lower": float(np.quantile(values, alpha)) if values else None,
                "ci_upper": float(np.quantile(values, 1.0 - alpha)) if values else None,
                "valid_repetitions": len(values),
                "bootstrap_unit": "subject",
            }
        )
    return pd.DataFrame(rows)


def _metrics(
    frame: pd.DataFrame, truth_col: str, prediction_col: str, probability_cols: list[str]
) -> dict[str, Any]:
    probability = frame[probability_cols].to_numpy(dtype=float)
    return compute_metrics(
        frame[truth_col].to_numpy(dtype=int),
        frame[prediction_col].to_numpy(dtype=int),
        probability,
        labels=[0, 1, 2],
    )
