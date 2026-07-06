from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MULTICLASS_MAPPING = {"CN": 0, "MCI": 1, "AD": 2}
BINARY_MAPPING = {"CN": 0, "AD": 1}


def normalize_diagnosis(value: object) -> str | None:
    """Map TADPOLE diagnosis strings to CN/MCI/AD."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "nan", "None"}:
        return None
    if text in {"CN", "NL", "NL to NL", "MCI to NL", "Dementia to NL"}:
        return "CN"
    if text in {"MCI", "NL to MCI", "MCI to MCI", "Dementia to MCI"}:
        return "MCI"
    if text in {"AD", "Dementia", "NL to Dementia", "MCI to Dementia"}:
        return "AD"
    return None


def build_diagnosis_labels(
    df: pd.DataFrame,
    label_col: str = "DX",
    task: str = "diagnosis_multiclass",
    output_dir: str | Path = "outputs",
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    """Create encoded diagnostic labels and write the label mapping."""
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    work = df.copy()
    work["_diagnosis"] = work[label_col].map(normalize_diagnosis)
    work = work.dropna(subset=["_diagnosis"]).copy()

    if task == "diagnosis_binary":
        mapping = BINARY_MAPPING
        work = work[work["_diagnosis"].isin(mapping)].copy()
    elif task == "diagnosis_multiclass":
        mapping = MULTICLASS_MAPPING
        work = work[work["_diagnosis"].isin(mapping)].copy()
    else:
        raise ValueError(f"Unsupported label task for current data: {task}")

    work["label"] = work["_diagnosis"].map(mapping).astype(int)
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    return work, "label", mapping


def select_task_rows(
    df: pd.DataFrame,
    task_mode: str,
    subject_col: str = "RID",
    visit_col: str = "VISCODE",
    date_col: str = "EXAMDATE",
) -> pd.DataFrame:
    """Select rows for baseline-only or all-visits diagnosis tasks."""
    if task_mode == "all_visits":
        return df.copy()
    if task_mode != "baseline_only":
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    work = df.copy()
    if visit_col in work.columns:
        baseline = work[work[visit_col].astype(str).str.lower().eq("bl")].copy()
        if baseline.empty:
            baseline = work.copy()
    else:
        baseline = work.copy()

    sort_cols = [subject_col]
    if date_col in baseline.columns:
        baseline["_sort_date"] = pd.to_datetime(baseline[date_col], errors="coerce")
        sort_cols.append("_sort_date")
    baseline = baseline.sort_values(sort_cols)
    baseline = baseline.drop_duplicates(subject_col, keep="first").copy()
    return baseline.drop(columns=["_sort_date"], errors="ignore")


def build_mci_conversion_labels(*args, **kwargs):
    """Placeholder for MCI conversion; current CSV has no baseline MCI cohort."""
    raise NotImplementedError(
        "MCI conversion is skipped: current TADPOLE_D1_D2.csv has no baseline MCI "
        "subjects in DX_bl, so stable/progressive MCI labels cannot be built safely."
    )
