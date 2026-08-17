from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.artifact_io import write_json

LOGGER = logging.getLogger(__name__)

# Canonical compact feature set. Every module that needs these columns -- the
# modality ablations, the Phase B compact baseline, and the frozen Phase C
# manifests -- must derive them from here so the sets cannot drift apart.
DEMOGRAPHIC_FEATURES = ["AGE", "PTGENDER", "PTEDUCAT"]
COGNITIVE_FEATURES = [
    "MMSE",
    "ADAS11",
    "ADAS13",
    "CDRSB",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "FAQ_bl",
]
MRI_STRUCTURAL_FEATURES = [
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
]
GENETIC_FEATURES = ["APOE4"]

MODALITY_GROUPS: dict[str, list[str]] = {
    "demographic": DEMOGRAPHIC_FEATURES,
    "cognitive": COGNITIVE_FEATURES,
    "mri_derived": MRI_STRUCTURAL_FEATURES,
    "genetic": GENETIC_FEATURES,
}

# Same features, keyed by the audited catalog modality names used by the frozen
# Phase C manifests ("mri_structural" instead of the ablation name "mri_derived").
CATALOG_MODALITY_GROUPS: dict[str, list[str]] = {
    "demographic": DEMOGRAPHIC_FEATURES,
    "cognitive": COGNITIVE_FEATURES,
    "mri_structural": MRI_STRUCTURAL_FEATURES,
    "genetic": GENETIC_FEATURES,
}

COMPACT_FEATURES = [
    *DEMOGRAPHIC_FEATURES,
    *COGNITIVE_FEATURES,
    *MRI_STRUCTURAL_FEATURES,
    *GENETIC_FEATURES,
]


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


def write_used_feature_groups(used_groups: dict[str, list[str]], output_dir: str | Path) -> None:
    """Persist the feature groups that were actually available."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "used_feature_groups.json", used_groups)
    used_features = {
        "groups": used_groups,
        "features": list(dict.fromkeys(col for cols in used_groups.values() for col in cols)),
        "notes": {"FAQ_bl": "baseline-only FAQ covariate"},
    }
    write_json(metrics_dir / "used_features.json", used_features)


def infer_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Split selected columns into numeric and categorical features."""
    numeric = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in feature_columns if col in df.columns and col not in numeric]
    return numeric, categorical
