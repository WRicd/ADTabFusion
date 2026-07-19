from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def select_d3_index_records(
    frame: pd.DataFrame,
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["_source_row"] = range(len(work))
    work["_parsed_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_date_missing"] = work["_parsed_date"].isna()
    ordered = work.sort_values(
        [subject_col, "_date_missing", "_parsed_date", "_source_row"],
        ascending=[True, True, False, False],
    )
    selected = ordered.drop_duplicates(subject_col, keep="first").copy()
    selected_rows = set(selected["_source_row"])
    audit = ordered[[subject_col, date_col, "_source_row"]].copy()
    audit["selected"] = audit["_source_row"].isin(selected_rows)
    selected = selected.sort_values(subject_col).drop(
        columns=["_source_row", "_parsed_date", "_date_missing"]
    )
    return selected.reset_index(drop=True), audit.reset_index(drop=True)


def prepare_d3_cohort(
    d3_csv: str | Path,
    output_dir: str | Path,
    features: list[str],
    modality_groups: dict[str, list[str]],
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = list(dict.fromkeys([subject_col, date_col, *features]))
    frame = pd.read_csv(
        d3_csv,
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    selected, selection_audit = select_d3_index_records(frame, subject_col, date_col)
    modality_coverage = {}
    for modality, columns in modality_groups.items():
        available = [column for column in columns if column in selected.columns]
        modality_coverage[modality] = (
            float(selected[available].notna().any(axis=1).mean()) if available else 0.0
        )
    summary = {
        "source_rows": int(len(frame)),
        "unique_subjects": int(frame[subject_col].nunique()),
        "duplicate_rid_count": int((frame[subject_col].value_counts() > 1).sum()),
        "selected_rows": int(len(selected)),
        "index_date_coverage": float(pd.to_datetime(selected[date_col], errors="coerce").notna().mean()),
        "active_feature_coverage": float(selected.reindex(columns=features).notna().mean().mean()),
        "modality_subject_coverage": modality_coverage,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "d3_cohort.csv", index=False)
    selection_audit.to_csv(output / "d3_record_selection_audit.csv", index=False)
    lines = [
        "# D3 Inference Cohort",
        "",
        f"- Source rows: {summary['source_rows']}",
        f"- Unique subjects: {summary['unique_subjects']}",
        f"- Duplicate RIDs: {summary['duplicate_rid_count']}",
        f"- Selected rows: {summary['selected_rows']}",
        f"- Index date coverage: {summary['index_date_coverage']:.1%}",
        f"- Active feature cell coverage: {summary['active_feature_coverage']:.1%}",
        "",
        "## Modality Subject Coverage",
        "",
        "| Modality | Coverage |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.1%} |" for name, value in modality_coverage.items())
    lines.extend(
        [
            "",
            "For duplicate RIDs, the latest parseable D3 EXAMDATE was selected. D4 was not consulted.",
            "",
        ]
    )
    (output / "d3_cohort_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return selected, summary
