from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.tadpole.dictionary_parser import load_tadpole_dictionary, match_dictionary_columns
from src.tadpole.modality_mapper import infer_modality


MODEL_MODALITIES = {
    "demographic",
    "cognitive",
    "mri_structural",
    "mri_dti",
    "pet_fdg",
    "pet_amyloid",
    "pet_other",
    "csf",
    "biofluid_other",
    "genetic",
}
MODALITY_PRIORITY = {
    "demographic": 0,
    "cognitive": 1,
    "mri_structural": 2,
    "mri_dti": 3,
    "pet_fdg": 4,
    "pet_amyloid": 5,
    "pet_other": 6,
    "csf": 7,
    "genetic": 8,
    "biofluid_other": 9,
}
PRIMARY_MODALITY_QUOTAS = {
    "demographic": 12,
    "cognitive": 40,
    "mri_structural": 50,
    "mri_dti": 35,
    "pet_fdg": 15,
    "pet_amyloid": 15,
    "pet_other": 15,
    "csf": 10,
    "genetic": 5,
    "biofluid_other": 3,
}
CATALOG_COLUMNS = [
    "column_name",
    "dictionary_match",
    "source_table",
    "description",
    "type",
    "units",
    "modality",
    "missing_rate",
    "non_missing_count",
    "unique_count",
    "available_in_d3",
    "suspected_label",
    "suspected_leakage",
    "duplicated_measurement_family",
    "recommended_use",
    "exclusion_reason",
]


def build_feature_catalog(
    data_path: str | Path,
    dictionary_path: str | Path,
    d3_path: str | Path | None = None,
    max_missing_rate: float = 0.70,
    min_non_missing_count: int = 100,
    max_features_per_family: int = 1,
    max_primary_features: int = 200,
) -> pd.DataFrame:
    """Build a field-level, dictionary-backed audit catalog."""
    if max_features_per_family < 1:
        raise ValueError("max_features_per_family must be at least 1")
    frame = pd.read_csv(
        data_path,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    dictionary = load_tadpole_dictionary(dictionary_path)
    matches = match_dictionary_columns(frame.columns, dictionary)
    d3_columns = _read_header(d3_path) if d3_path else set()
    missing_rates = frame.isna().mean()
    non_missing = frame.notna().sum()
    unique_counts = frame.nunique(dropna=True)

    records: list[dict[str, Any]] = []
    for column in frame.columns:
        name = str(column)
        metadata = matches[name] or {}
        modality = infer_modality(name, metadata)
        missing_rate = float(missing_rates[name])
        count = int(non_missing[name])
        status, reason = _initial_recommendation(
            modality, missing_rate, count, max_missing_rate, min_non_missing_count
        )
        records.append(
            {
                "column_name": name,
                "dictionary_match": bool(matches[name]),
                "source_table": metadata.get("TBLNAME", ""),
                "description": metadata.get("TEXT", ""),
                "type": metadata.get("TYPE", ""),
                "units": metadata.get("UNITS", ""),
                "modality": modality,
                "missing_rate": round(missing_rate, 6),
                "non_missing_count": count,
                "unique_count": int(unique_counts[name]),
                "available_in_d3": name in d3_columns,
                "suspected_label": modality in {"label", "baseline_diagnosis"},
                "suspected_leakage": modality
                in {
                    "identifier",
                    "visit_time",
                    "label",
                    "baseline_diagnosis",
                    "administrative",
                },
                "duplicated_measurement_family": measurement_family(name, metadata),
                "recommended_use": status,
                "exclusion_reason": reason,
            }
        )

    catalog = pd.DataFrame(records, columns=CATALOG_COLUMNS)
    catalog = _resolve_duplicate_families(catalog, max_features_per_family)
    catalog = _select_primary_features(catalog, max_primary_features)
    return catalog


def measurement_family(column_name: str, metadata: dict[str, str] | None = None) -> str:
    """Normalize source/date suffixes so duplicate measurements can be audited."""
    name = column_name.upper()
    name = re.sub(
        r"_(UCSFFSL|UCSFFSX|BAIPETNMRC|UCBERKELEYAV45|UCBERKELEYAV1451|DTIROI|UPENNBIOMK).*$",
        "",
        name,
    )
    name = re.sub(r"_BL$", "", name)
    return name


def write_feature_catalog(
    catalog: pd.DataFrame,
    output_csv: str | Path,
    output_md: str | Path,
    modality_map_md: str | Path | None = None,
) -> None:
    csv_path = Path(output_csv)
    md_path = Path(output_md)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(csv_path, index=False)
    md_path.write_text(render_feature_catalog_markdown(catalog), encoding="utf-8")
    if modality_map_md:
        modality_path = Path(modality_map_md)
        modality_path.parent.mkdir(parents=True, exist_ok=True)
        modality_path.write_text(render_modality_map_markdown(catalog), encoding="utf-8")


def render_feature_catalog_markdown(catalog: pd.DataFrame) -> str:
    status_counts = catalog["recommended_use"].value_counts().sort_index()
    modality_counts = catalog.groupby("modality").size().sort_values(ascending=False)
    primary = catalog[catalog["recommended_use"] == "include_primary"]
    lines = [
        "# Full TADPOLE Feature Catalog",
        "",
        f"Audited columns: {len(catalog)}",
        f"Dictionary matches: {int(catalog['dictionary_match'].sum())}",
        f"Primary whitelist features: {len(primary)}",
        "",
        "## Recommendation summary",
        "",
        "| Status | Columns |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in status_counts.items())
    lines.extend(["", "## Modality summary", "", "| Modality | Columns |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in modality_counts.items())
    lines.extend(
        [
            "",
            "## Primary whitelist",
            "",
            "| Column | Modality | Source | Missing rate | D3 available |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in primary.itertuples(index=False):
        lines.append(
            f"| `{_escape(row.column_name)}` | {row.modality} | "
            f"{_escape(row.source_table) or '-'} | {row.missing_rate:.3f} | "
            f"{'yes' if row.available_in_d3 else 'no'} |"
        )
    lines.extend(
        [
            "",
            "> The CSV output contains the complete per-column audit. This Markdown report contains aggregate results and the primary whitelist only.",
            "",
        ]
    )
    return "\n".join(lines)


def catalog_summary(catalog: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": int(len(catalog)),
        "dictionary_matches": int(catalog["dictionary_match"].sum()),
        "primary_features": int((catalog["recommended_use"] == "include_primary").sum()),
        "status_counts": catalog["recommended_use"].value_counts().to_dict(),
        "modality_counts": catalog["modality"].value_counts().to_dict(),
    }


def render_modality_map_markdown(catalog: pd.DataFrame) -> str:
    lines = [
        "# Full TADPOLE Modality Map",
        "",
        "Assignments prioritize exact dictionary metadata, especially `TBLNAME` for DTI, PET and CSF sources.",
        "",
        "| Modality | Total columns | Best missing rate | Eligible at 70% | Primary |",
        "|---|---:|---:|---:|---:|",
    ]
    for modality, group in catalog.groupby("modality", sort=True):
        eligible = group["recommended_use"].isin({"include_primary", "include_optional"}).sum()
        primary = (group["recommended_use"] == "include_primary").sum()
        lines.append(
            f"| {modality} | {len(group)} | {group['missing_rate'].min():.3f} | "
            f"{eligible} | {primary} |"
        )
    lines.extend(
        [
            "",
            "## Source rules",
            "",
            "- `DTIROI` dictionary rows map to `mri_dti`.",
            "- `UCSFFSL` and `UCSFFSX` dictionary rows map to `mri_structural`.",
            "- `BAIPETNMRC` maps to `pet_fdg`.",
            "- `UCBERKELEYAV45` maps to `pet_amyloid`.",
            "- `UCBERKELEYAV1451` maps to `pet_other`.",
            "- `UPENNBIOMK9`, `ABETA_*`, `TAU_*` and `PTAU_*` map to `csf`.",
            "",
            "> DTI and CSF are present but do not pass the default 70% missing-rate threshold in the full-row primary cohort. They require modality-specific cohort experiments rather than silent inclusion in the primary model.",
            "",
        ]
    )
    return "\n".join(lines)


def _initial_recommendation(
    modality: str,
    missing_rate: float,
    non_missing_count: int,
    max_missing_rate: float,
    min_non_missing_count: int,
) -> tuple[str, str]:
    if modality in {"label", "baseline_diagnosis", "visit_time"}:
        return "exclude_leakage", "Target-derived or temporal field is excluded from model inputs."
    if modality == "identifier":
        return "exclude_identifier", "Identifier is retained only for joins and subject-level splits."
    if modality == "administrative":
        return "exclude_administrative", "Administrative or data-construction field."
    if modality not in MODEL_MODALITIES:
        return "exclude_unresolved", "Field meaning or modeling role is not resolved for the primary set."
    if missing_rate > max_missing_rate or non_missing_count < min_non_missing_count:
        return "exclude_high_missing", "Availability is below the configured feature threshold."
    return "include_optional", "Eligible candidate pending duplicate resolution and primary-set ranking."


def _resolve_duplicate_families(
    catalog: pd.DataFrame, max_features_per_family: int
) -> pd.DataFrame:
    result = catalog.copy()
    candidates = result[result["recommended_use"] == "include_optional"]
    for _, group in candidates.groupby("duplicated_measurement_family", sort=False):
        if len(group) <= max_features_per_family:
            continue
        ranked = group.sort_values(
            ["available_in_d3", "missing_rate", "dictionary_match", "column_name"],
            ascending=[False, True, False, True],
        )
        rejected = ranked.index[max_features_per_family:]
        result.loc[rejected, "recommended_use"] = "exclude_duplicate"
        result.loc[rejected, "exclusion_reason"] = (
            "A better-covered representative from the same measurement family was retained."
        )
    return result


def _select_primary_features(catalog: pd.DataFrame, max_features: int) -> pd.DataFrame:
    result = catalog.copy()
    candidates = result[result["recommended_use"] == "include_optional"].copy()
    candidates["_modality_rank"] = candidates["modality"].map(MODALITY_PRIORITY).fillna(99)
    ranked = candidates.sort_values(
        ["_modality_rank", "available_in_d3", "missing_rate", "dictionary_match", "column_name"],
        ascending=[True, False, True, False, True],
    )
    selected: list[int] = []
    for modality, quota in PRIMARY_MODALITY_QUOTAS.items():
        group = ranked[ranked["modality"] == modality]
        selected.extend(group.index[: min(quota, max_features - len(selected))].tolist())
        if len(selected) >= max_features:
            break
    result.loc[selected, "recommended_use"] = "include_primary"
    result.loc[selected, "exclusion_reason"] = ""
    result.loc[
        (result["recommended_use"] == "include_optional"), "exclusion_reason"
    ] = "Eligible audited feature retained as optional but outside the primary feature cap."
    return result


def _read_header(path: str | Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns.astype(str))


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
