from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_schema import MULTICLASS_MAPPING, normalize_diagnosis


PAIR_METADATA = ["RID", "source_date", "future_date", "forecast_months", "target_diagnosis", "label"]


def split_subjects_before_pairing(
    frame: pd.DataFrame,
    subject_col: str = "RID",
    seed: int = 42,
    val_size: float = 0.10,
    test_size: float = 0.20,
) -> dict[str, set[Any]]:
    ids = frame[subject_col].dropna().drop_duplicates().to_numpy()
    train_val, test = train_test_split(ids, test_size=test_size, random_state=seed)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
    return {"train": set(train), "val": set(val), "test": set(test)}


def construct_future_diagnosis_pairs(
    frame: pd.DataFrame,
    features: list[str],
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
    label_col: str = "DX",
    min_horizon_months: float = 6.0,
    max_horizon_months: float = 60.0,
) -> pd.DataFrame:
    required = [subject_col, date_col, label_col, *features]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Pair source is missing fields: {', '.join(missing)}")
    work = frame[required].copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_future_dx"] = work[label_col].map(normalize_diagnosis)
    work = work.dropna(subset=[subject_col, "_date"]).sort_values([subject_col, "_date"])
    rows: list[dict[str, Any]] = []
    for rid, visits in work.groupby(subject_col, sort=False):
        records = visits.to_dict("records")
        for source_index, source in enumerate(records[:-1]):
            for future in records[source_index + 1 :]:
                days = (future["_date"] - source["_date"]).days
                months = days / 30.4375
                if months < min_horizon_months or months > max_horizon_months:
                    continue
                target = future["_future_dx"]
                if target not in MULTICLASS_MAPPING:
                    continue
                row = {
                    "RID": rid,
                    "source_date": source["_date"].date().isoformat(),
                    "future_date": future["_date"].date().isoformat(),
                    "forecast_months": float(months),
                    "target_diagnosis": target,
                    "label": MULTICLASS_MAPPING[target],
                }
                row.update({feature: source[feature] for feature in features})
                rows.append(row)
    return pd.DataFrame(rows, columns=[*PAIR_METADATA, *features])


def build_split_pairs(
    frame: pd.DataFrame,
    features: list[str],
    split: dict[str, set[Any]],
    **pairing: Any,
) -> dict[str, pd.DataFrame]:
    result = {}
    subject_col = pairing.get("subject_col", "RID")
    for name, ids in split.items():
        subset = frame[frame[subject_col].isin(ids)].copy()
        pairs = construct_future_diagnosis_pairs(subset, features, **pairing)
        pairs["split"] = name
        result[name] = pairs
    return result


def assert_pair_integrity(pairs: dict[str, pd.DataFrame], features: list[str]) -> None:
    subject_sets = {name: set(frame["RID"]) for name, frame in pairs.items()}
    names = list(subject_sets)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if not subject_sets[name].isdisjoint(subject_sets[other]):
                raise AssertionError("A RID crosses future-diagnosis splits.")
    for frame in pairs.values():
        if not frame.empty:
            source = pd.to_datetime(frame["source_date"])
            future = pd.to_datetime(frame["future_date"])
            if not bool((source < future).all()):
                raise AssertionError("Every source date must precede its future target date.")
            forbidden = {"target_diagnosis", "label", "future_date"}.intersection(features)
            if forbidden:
                raise AssertionError(f"Future target fields entered source features: {forbidden}")


def load_longitudinal_source(
    csv_path: str | Path,
    features: list[str],
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
    label_col: str = "DX",
) -> pd.DataFrame:
    required = list(dict.fromkeys([subject_col, date_col, label_col, *features]))
    frame = pd.read_csv(
        csv_path,
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Longitudinal source is missing fields: {', '.join(missing)}")
    return frame
