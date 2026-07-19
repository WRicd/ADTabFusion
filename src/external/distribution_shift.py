from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.external.d3_cohort import select_d3_index_records
from src.external.inference import probability_frame
from src.external.model_freezing import load_baseline_training_cohort
from src.external.schema_alignment import align_to_frozen_schema


def analyze_d1d2_d3_shift(config: dict[str, Any]) -> pd.DataFrame:
    data = config["data"]
    output = Path(config["project"]["output_dir"])
    manifest_path = output / "manifests" / "primary_model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["feature_order"]
    train = load_baseline_training_cohort(
        data["train_csv"], features, data.get("subject_col", "RID"),
        data.get("visit_col", "VISCODE"), data.get("date_col", "EXAMDATE"),
        data.get("label_col", "DX"),
    )
    required = list(dict.fromkeys([data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"), *features]))
    d3_raw = pd.read_csv(
        data["d3_csv"], usecols=lambda column: column in required, low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    d3, _ = select_d3_index_records(d3_raw, data.get("subject_col", "RID"), data.get("date_col", "EXAMDATE"))
    d3_aligned = align_to_frozen_schema(d3, features)
    rows: list[dict[str, Any]] = []
    for feature in features:
        train_series = train[feature]
        d3_series = d3_aligned[feature]
        train_numeric = pd.to_numeric(train_series, errors="coerce")
        d3_numeric = pd.to_numeric(d3_series, errors="coerce")
        numeric = train_numeric.notna().sum() >= max(10, train_series.notna().sum() * 0.8)
        base = {
            "feature": feature,
            "train_missing_rate": float(train_series.isna().mean()),
            "d3_missing_rate": float(d3_series.isna().mean()),
            "missing_rate_shift": float(d3_series.isna().mean() - train_series.isna().mean()),
            "kind": "numeric" if numeric else "categorical",
        }
        if numeric:
            left = train_numeric.dropna().to_numpy(dtype=float)
            right = d3_numeric.dropna().to_numpy(dtype=float)
            train_mean = float(np.mean(left)) if len(left) else None
            d3_mean = float(np.mean(right)) if len(right) else None
            pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2) if len(left) > 1 and len(right) > 1 else np.nan
            base.update(
                {
                    "train_mean": train_mean,
                    "train_std": float(np.std(left, ddof=1)) if len(left) > 1 else None,
                    "d3_mean": d3_mean,
                    "d3_std": float(np.std(right, ddof=1)) if len(right) > 1 else None,
                    "standardized_mean_difference": float((d3_mean - train_mean) / pooled) if pooled and np.isfinite(pooled) else None,
                    "ks_statistic": float(ks_2samp(left, right).statistic) if len(left) and len(right) else None,
                    "psi": _population_stability_index(left, right),
                    "unseen_categories": None,
                }
            )
        else:
            known = set(train_series.dropna().astype(str))
            unseen = sorted(set(d3_series.dropna().astype(str)) - known)
            base.update(
                {
                    "train_mean": None, "train_std": None, "d3_mean": None, "d3_std": None,
                    "standardized_mean_difference": None, "ks_statistic": None, "psi": None,
                    "unseen_categories": "|".join(unseen),
                }
            )
        rows.append(base)
    pipeline = joblib.load(output / "models" / "primary_pipeline.joblib")
    for cohort_name, frame in (("D1_D2_training", train[features]), ("D3", d3_aligned)):
        prediction = probability_frame(pipeline.predict_proba(frame))
        rows.append(
            {
                "feature": "prediction_confidence", "kind": "prediction_distribution",
                "cohort": cohort_name, "train_mean": float(prediction["prediction_confidence"].mean()),
                "train_std": float(prediction["prediction_confidence"].std()),
            }
        )
        rows.append(
            {
                "feature": "prediction_entropy", "kind": "prediction_distribution",
                "cohort": cohort_name, "train_mean": float(prediction["prediction_entropy"].mean()),
                "train_std": float(prediction["prediction_entropy"].std()),
            }
        )
    result = pd.DataFrame(rows)
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    result.to_csv(evaluation / "dataset_shift_summary.csv", index=False)
    _plot_shift(result, output / "figures" / "dataset_shift.png")
    return result


def _population_stability_index(train: np.ndarray, current: np.ndarray) -> float | None:
    if len(train) < 2 or len(current) < 2:
        return None
    edges = np.unique(np.quantile(train, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_hist = np.histogram(train, bins=edges)[0] / len(train)
    current_hist = np.histogram(current, bins=edges)[0] / len(current)
    train_hist = np.clip(train_hist, 1e-6, None)
    current_hist = np.clip(current_hist, 1e-6, None)
    return float(np.sum((current_hist - train_hist) * np.log(current_hist / train_hist)))


def _plot_shift(frame: pd.DataFrame, path: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    numeric = frame[frame["kind"] == "numeric"].copy()
    numeric["abs_smd"] = numeric["standardized_mean_difference"].abs()
    plot = numeric.sort_values("abs_smd", ascending=False).head(12).sort_values("abs_smd")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh(plot["feature"], plot["abs_smd"], color="#2f6f8f")
    ax.axvline(0.2, color="#c44e52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("D1/D2 to D3 Dataset Shift")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
