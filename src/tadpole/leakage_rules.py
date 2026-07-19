from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BLOCKED_STATUSES = {
    "exclude_leakage",
    "exclude_identifier",
    "exclude_administrative",
    "exclude_high_missing",
    "exclude_duplicate",
    "exclude_unresolved",
}


def build_leakage_artifacts(catalog: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Build an explicit blacklist and primary feature whitelist."""
    blacklist = {
        str(row.column_name): str(row.exclusion_reason)
        for row in catalog.itertuples(index=False)
        if row.recommended_use in BLOCKED_STATUSES
    }
    whitelist = catalog.loc[
        catalog["recommended_use"] == "include_primary", "column_name"
    ].astype(str).tolist()
    return blacklist, whitelist


def write_leakage_artifacts(
    catalog: pd.DataFrame,
    blacklist_path: str | Path,
    whitelist_path: str | Path,
    report_path: str | Path,
) -> tuple[dict[str, str], list[str]]:
    blacklist, whitelist = build_leakage_artifacts(catalog)
    blacklist_file = Path(blacklist_path)
    whitelist_file = Path(whitelist_path)
    report_file = Path(report_path)
    for path in (blacklist_file, whitelist_file, report_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    blacklist_file.write_text(
        json.dumps(blacklist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    whitelist_file.write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_file.write_text(
        render_leakage_audit(catalog, blacklist, whitelist), encoding="utf-8"
    )
    return blacklist, whitelist


def render_leakage_audit(
    catalog: pd.DataFrame, blacklist: dict[str, str], whitelist: list[str]
) -> str:
    excluded = catalog[catalog["recommended_use"].isin(BLOCKED_STATUSES)]
    counts = excluded["recommended_use"].value_counts().sort_index()
    direct = catalog[catalog["suspected_leakage"]]
    lines = [
        "# Full TADPOLE Leakage Audit",
        "",
        "Task: CN / MCI / AD internal diagnosis classification",
        "",
        f"Audited columns: {len(catalog)}",
        f"Primary whitelist: {len(whitelist)}",
        f"Blacklist: {len(blacklist)}",
        "",
        "## Exclusion summary",
        "",
        "| Rule | Columns |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in counts.items())
    lines.extend(
        [
            "",
            "## Direct leakage and split-only fields",
            "",
            "| Column | Modality | Reason |",
            "|---|---|---|",
        ]
    )
    for row in direct.itertuples(index=False):
        lines.append(
            f"| `{row.column_name}` | {row.modality} | {row.exclusion_reason} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- D4 is excluded from training, preprocessing fit, feature selection and tuning.",
            "- D3 is an external transform-only input and is not used to fit the primary model.",
            "- RID is retained only for subject-level splitting and D3/D4 matching.",
            "- Visit and date fields are retained for audit or matching but excluded from diagnostic features.",
            "- Duplicate source/date variants are reduced to one representative per measurement family.",
            "",
        ]
    )
    return "\n".join(lines)

