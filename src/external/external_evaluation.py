from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics


CLASS_TO_ID = {"CN": 0, "MCI": 1, "AD": 2}


def evaluate_external_predictions(
    frame: pd.DataFrame,
    prediction_col: str,
    probability_cols: list[str],
    model_id: str,
    minimum_stable_rows: int = 20,
) -> pd.DataFrame:
    work = frame.copy()
    work["_predicted_label"] = work[prediction_col].map(CLASS_TO_ID)
    work["_d4_date"] = pd.to_datetime(work["D4_SCANDATE"], errors="coerce")
    first = work.sort_values(["RID", "_d4_date"]).drop_duplicates("RID", keep="first")
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("row_level", "overall", work),
        ("first_follow_up", "overall", first),
    ]
    bins = [
        (0, 12, "0-12 months"), (12, 24, "12-24 months"),
        (24, 36, "24-36 months"), (36, np.inf, ">36 months"),
    ]
    for low, high, name in bins:
        subset = work[(work["forecast_months"] > low) & (work["forecast_months"] <= high)]
        scopes.append(("horizon", name, subset))
    rows: list[dict[str, Any]] = []
    for scope, horizon, subset in scopes:
        row: dict[str, Any] = {
            "model_id": model_id,
            "scope": scope,
            "horizon": horizon,
            "n_rows": int(len(subset)),
            "n_subjects": int(subset["RID"].nunique()),
            "stable_metrics_reported": bool(len(subset) >= minimum_stable_rows or scope != "horizon"),
        }
        if subset.empty or not row["stable_metrics_reported"]:
            rows.append(row)
            continue
        metrics = compute_metrics(
            subset["D4_label"].to_numpy(dtype=int),
            subset["_predicted_label"].to_numpy(dtype=int),
            subset[probability_cols].to_numpy(dtype=float),
            labels=[0, 1, 2],
        )
        row.update(
            {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in metrics.items()}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def first_follow_up(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_date"] = pd.to_datetime(work["D4_SCANDATE"], errors="coerce")
    return work.sort_values(["RID", "_date"]).drop_duplicates("RID", keep="first")
