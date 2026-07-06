from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def make_subject_split(
    df: pd.DataFrame,
    subject_col: str,
    label_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    output_dir: str | Path = "outputs",
) -> dict[str, list[int]]:
    """Split dataframe indices by subject without subject leakage."""
    subjects = df[[subject_col, label_col]].groupby(subject_col)[label_col].agg(_mode)
    subject_ids = subjects.index.to_numpy()
    subject_labels = subjects.to_numpy()

    train_val_ids, test_ids = _split_ids(
        subject_ids, subject_labels, test_size, seed, "test"
    )
    train_val_labels = subjects.loc[train_val_ids].to_numpy()
    relative_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids = _split_ids(
        train_val_ids, train_val_labels, relative_val_size, seed, "val"
    )

    split = {
        "train_idx": df.index[df[subject_col].isin(train_ids)].astype(int).tolist(),
        "val_idx": df.index[df[subject_col].isin(val_ids)].astype(int).tolist(),
        "test_idx": df.index[df[subject_col].isin(test_ids)].astype(int).tolist(),
    }
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"split_seed_{seed}.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    return split


def _mode(values: pd.Series) -> int:
    return int(values.mode().iloc[0])


def _split_ids(
    ids: np.ndarray, labels: np.ndarray, test_size: float, seed: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels
    try:
        return train_test_split(
            ids,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        LOGGER.warning("Falling back to non-stratified %s split: %s", name, exc)
        return train_test_split(ids, test_size=test_size, random_state=seed, stratify=None)

