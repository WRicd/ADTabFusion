from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

MODALITY_GROUPS: dict[str, list[str]] = {
    "demographic": ["AGE", "PTGENDER", "PTEDUCAT"],
    "cognitive": [
        "MMSE",
        "ADAS11",
        "ADAS13",
        "CDRSB",
        "RAVLT_immediate",
        "RAVLT_learning",
        "RAVLT_forgetting",
        "RAVLT_perc_forgetting",
        "FAQ_bl",
    ],
    "mri_derived": [
        "Ventricles",
        "Hippocampus",
        "WholeBrain",
        "Entorhinal",
        "Fusiform",
        "MidTemp",
        "ICV",
    ],
    "genetic": ["APOE4"],
}


def available_groups(
    df: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Return modality groups restricted to columns present in the dataframe."""
    groups = groups or MODALITY_GROUPS
    available: dict[str, list[str]] = {}
    for name, columns in groups.items():
        present = [col for col in columns if col in df.columns]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            LOGGER.warning("Missing columns for %s: %s", name, missing)
        available[name] = present
    return available


def columns_for_modalities(
    df: pd.DataFrame,
    modality_names: list[str],
    groups: dict[str, list[str]] | None = None,
) -> list[str]:
    """Collect feature columns for selected modalities."""
    present_groups = available_groups(df, groups)
    columns: list[str] = []
    for name in modality_names:
        columns.extend(present_groups.get(name, []))
    return list(dict.fromkeys(columns))


def write_used_feature_groups(
    used_groups: dict[str, list[str]], output_dir: str | Path
) -> None:
    """Persist the feature groups that were actually available."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "used_feature_groups.json").write_text(
        json.dumps(used_groups, indent=2), encoding="utf-8"
    )
    used_features = {
        "groups": used_groups,
        "features": list(dict.fromkeys(col for cols in used_groups.values() for col in cols)),
        "notes": {"FAQ_bl": "baseline-only FAQ covariate"},
    }
    (metrics_dir / "used_features.json").write_text(
        json.dumps(used_features, indent=2), encoding="utf-8"
    )


def infer_feature_types(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[str], list[str]]:
    """Split selected columns into numeric and categorical features."""
    numeric = [
        col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical = [col for col in feature_columns if col in df.columns and col not in numeric]
    return numeric, categorical
