from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.external.d3_cohort import select_d3_index_records


LABEL_TO_ID = {"CN": 0, "MCI": 1, "AD": 2}


def normalize_diagnosis_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    mapping = {
        "cn": "CN", "nl": "CN", "normal": "CN",
        "mci": "MCI", "lmci": "MCI", "emci": "MCI",
        "ad": "AD", "dementia": "AD",
    }
    return mapping.get(text)


def build_d3_d4_evaluation_cohort(
    d3_csv: str | Path,
    d4_csv: str | Path,
    primary_predictions: str | Path,
    sensitivity_predictions: str | Path,
    horizon_features: list[str],
    output_dir: str | Path,
    subject_col: str = "RID",
    d3_date_col: str = "EXAMDATE",
    d4_date_col: str = "ScanDate",
    d4_label_col: str = "Diagnosis",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_d3 = list(dict.fromkeys([subject_col, d3_date_col, *horizon_features]))
    d3_raw = pd.read_csv(
        d3_csv, usecols=lambda column: column in required_d3, low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    d3, _ = select_d3_index_records(d3_raw, subject_col, d3_date_col)
    d3 = d3.rename(columns={d3_date_col: "D3_EXAMDATE"})
    d4 = pd.read_csv(
        d4_csv,
        usecols=lambda column: column in {subject_col, d4_date_col, d4_label_col},
        low_memory=False,
    ).rename(columns={d4_date_col: "D4_SCANDATE", d4_label_col: "D4_DIAGNOSIS_RAW"})
    d4["D4_DIAGNOSIS"] = d4["D4_DIAGNOSIS_RAW"].map(normalize_diagnosis_label)
    d4["D4_label"] = d4["D4_DIAGNOSIS"].map(LABEL_TO_ID)
    d4["_d4_date"] = pd.to_datetime(d4["D4_SCANDATE"], errors="coerce")
    d3["_d3_date"] = pd.to_datetime(d3["D3_EXAMDATE"], errors="coerce")
    merged = d4.merge(d3, on=subject_col, how="left", indicator=True)
    merged["forecast_days"] = (merged["_d4_date"] - merged["_d3_date"]).dt.days
    merged["forecast_months"] = merged["forecast_days"] / 30.4375
    merged["exclusion_reason"] = None
    merged.loc[merged["_merge"] != "both", "exclusion_reason"] = "RID_not_in_D3"
    merged.loc[merged["exclusion_reason"].isna() & merged["_d3_date"].isna(), "exclusion_reason"] = "invalid_D3_date"
    merged.loc[merged["exclusion_reason"].isna() & merged["_d4_date"].isna(), "exclusion_reason"] = "invalid_D4_date"
    merged.loc[merged["exclusion_reason"].isna() & (merged["forecast_days"] <= 0), "exclusion_reason"] = "nonpositive_horizon"
    merged.loc[merged["exclusion_reason"].isna() & merged["D4_DIAGNOSIS"].isna(), "exclusion_reason"] = "unresolved_D4_diagnosis"
    eligible = merged[merged["exclusion_reason"].isna()].copy()
    for role, path in (("primary", primary_predictions), ("sensitivity", sensitivity_predictions)):
        predictions = pd.read_csv(path)
        rename = {
            column: f"direct_{role}_{column}"
            for column in predictions.columns
            if column not in {"RID", "D3_EXAMDATE"}
        }
        eligible = eligible.merge(predictions.rename(columns=rename), on="RID", how="left")
    eligible = eligible.sort_values([subject_col, "_d4_date"]).reset_index(drop=True)
    eligible["evaluation_row_id"] = range(len(eligible))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_counts = (
        d4.groupby("D4_DIAGNOSIS_RAW", dropna=False)["D4_DIAGNOSIS"]
        .agg(["count", "first"])
        .reset_index()
        .rename(columns={"first": "normalized_label"})
    )
    label_counts.to_csv(output.parent / "audit" / "d4_label_mapping.csv", index=False)
    d4[d4["D4_DIAGNOSIS"].isna()][["D4_DIAGNOSIS_RAW"]].value_counts(dropna=False).rename(
        "count"
    ).reset_index().to_csv(output.parent / "audit" / "d4_unresolved_labels.csv", index=False)
    eligible.drop(columns=["_merge", "_d3_date", "_d4_date", "exclusion_reason"], errors="ignore").to_csv(
        output / "d3_d4_matched_rows.csv", index=False
    )
    reasons = merged["exclusion_reason"].fillna("included").value_counts().to_dict()
    summary = {
        "d3_subject_count": int(d3[subject_col].nunique()),
        "d4_row_count": int(len(d4)),
        "d4_unique_subject_count": int(d4[subject_col].nunique()),
        "matched_rows": int(len(eligible)),
        "matched_subjects": int(eligible[subject_col].nunique()),
        "outcomes": {str(key): int(value) for key, value in reasons.items()},
    }
    report = [
        "# D3/D4 Matching Audit", "",
        f"- D3 subjects: {summary['d3_subject_count']}",
        f"- D4 rows: {summary['d4_row_count']}",
        f"- D4 unique subjects: {summary['d4_unique_subject_count']}",
        f"- Eligible matched rows: {summary['matched_rows']}",
        f"- Eligible matched subjects: {summary['matched_subjects']}",
        "", "## Inclusion and Exclusion", "", "| Outcome | Rows |", "|---|---:|",
    ]
    report.extend(f"| {key} | {value} |" for key, value in summary["outcomes"].items())
    report.extend(["", "## Follow-up Rows per Subject", ""])
    counts = eligible.groupby(subject_col).size()
    report.append(f"Mean: {counts.mean():.2f}; median: {counts.median():.1f}; max: {counts.max() if len(counts) else 0}.")
    report.extend(["", "## Forecast Horizon (months)", ""])
    report.append(eligible["forecast_months"].describe().to_string())
    report.extend(["", "## Diagnosis Distribution", ""])
    report.append(eligible["D4_DIAGNOSIS"].value_counts().to_string())
    report.append("")
    (output / "d3_d4_matching_report.md").write_text("\n".join(report), encoding="utf-8")
    return eligible, summary
