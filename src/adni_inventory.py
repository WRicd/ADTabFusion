from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ID_CANDIDATES = ["RID", "PTID", "Subject", "SUBJECT", "LONIUID"]
VISIT_CANDIDATES = [
    "VISCODE",
    "VISCODE2",
    "VISDATE",
    "EXAMDATE",
    "SCANDATE",
    "USERDATE",
    "USERDATE2",
    "COLPROT",
]
DIAGNOSIS_CANDIDATES = [
    "DX",
    "DX_bl",
    "DXCHANGE",
    "DIAGNOSIS",
    "DIAGNOSISCN",
    "Phase",
    "COLPROT",
]
DEMOGRAPHIC_CANDIDATES = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTETHCAT",
    "PTRACCAT",
    "PTDOBYY",
    "PTMARRY",
]
APOE_CANDIDATES = ["APOE4", "APGEN1", "APGEN2", "GENOTYPE", "APOE_GENOTYPE"]
COGNITIVE_CANDIDATES = [
    "MMSE",
    "MMSCORE",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "CDRSB",
    "CDGLOBAL",
    "FAQ",
    "FAQ_bl",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "LDELTOTAL",
    "TRABSCOR",
    "CLOCKSCOR",
]
MRI_CANDIDATES = [
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
    "ST101SV",
    "ST102CV",
    "ST29SV",
    "ST88SV",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "CorticalThickness",
    "FreeSurfer",
]
PET_CANDIDATES = [
    "FDG",
    "AV45",
    "PIB",
    "AV1451",
    "FBB",
    "SUMMARYSUVR",
    "SUVR",
    "CEREBELLUMGREYMATTER",
    "AMYLOID",
    "TAU",
]
CSF_CANDIDATES = [
    "ABETA",
    "ABETA42",
    "TAU",
    "PTAU",
    "PTAU181",
    "A\u03b242",
    "tTau",
    "pTau",
    "ELECSYS",
    "UPENNBIOMK",
]
CSF_PATTERNS = [
    r"^ABETA($|_)",
    r"^ABETA42($|_)",
    r"^TAU($|_)",
    r"^PTAU($|_)",
    r"^PTAU181($|_)",
    r"UPENNBIOMK",
    r"ELECSYS",
]
PET_PATTERNS = [
    r"^FDG($|_)",
    r"^PIB($|_)",
    r"^AV45($|_)",
    r"AV1451",
    r"FLORBETAPIR",
    r"FLORTAUCIPIR",
    r"SUVR",
    r"CENTILOID",
]
DICTIONARY_CANDIDATES = ["FLDNAME", "TBLNAME", "TEXT", "CODE", "TYPE", "UNITS"]

MRI_KEYWORDS = [
    "MRI",
    "FREESURFER",
    "UCSF",
    "VOLUME",
    "VOLUMETRIC",
    "CORTICAL",
    "THICKNESS",
    "HIPPOCAMPUS",
    "VENTRICLE",
]
PET_KEYWORDS = [
    "PET",
    "FDG",
    "AV45",
    "AV1451",
    "PIB",
    "FBB",
    "AMYLOID",
    "TAU",
    "SUVR",
    "FLORBETAPIR",
    "FLORTAUCIPIR",
]
CSF_KEYWORDS = [
    "CSF",
    "BIOFLUID",
    "BIOSPECIMEN",
    "BIOMARKER",
    "PLASMA",
    "ABETA",
    "TAU",
    "PTAU",
    "ELECSYS",
    "UPENN",
]
DICTIONARY_KEYWORDS = ["DICTIONARY", "DATADIC", "DEFINE", "VARIABLE"]

COGNITIVE_PREFIXES = (
    "ADAS",
    "MMSE",
    "RAVLT",
    "CDR",
    "FAQ",
    "LDELT",
    "TRAB",
    "CLOCK",
    "LOGMEM",
    "CATFLU",
    "WAISR",
)
IMAGE_METADATA_COLUMNS = {
    "IMAGEUID",
    "SERIESID",
    "STUDYID",
    "IMGFILE",
    "DOWNLOAD",
    "MODALITY",
}
OUTPUT_MODALITIES = {
    "data_dictionary": ("data_dictionary", "data_dictionary"),
    "diagnosis": ("diagnosis", "diagnosis"),
    "demographic": ("demographics", "demographic"),
    "apoe": ("apoe", "apoe"),
    "cognitive": ("cognitive", "cognitive"),
    "mri_measurement": ("mri_measurement", "mri"),
    "pet_measurement": ("pet_measurement", "pet"),
    "biofluid_csf": ("biofluid_csf", "csf"),
}


def scan_raw_directory(raw_dir: str | Path) -> list[dict]:
    """Inspect every CSV below raw_dir without modifying source files."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {root}")

    csv_paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
        key=lambda path: str(path).lower(),
    )
    return [inspect_csv(path) for path in csv_paths]


def inspect_csv(path: str | Path) -> dict:
    """Inspect one CSV and return structure-only metadata."""
    csv_path = Path(path)
    try:
        sample, encoding = _read_sample(csv_path)
    except Exception as exc:  # A bad file must not stop the inventory.
        return _failed_record(csv_path, exc)

    columns = [str(column) for column in sample.columns]
    matched = _matched_features(columns)
    measurement_columns = _numeric_measurement_columns(sample)
    categories, primary = _classify(csv_path, columns, matched, measurement_columns)

    if primary in {"mri_measurement", "pet_measurement", "biofluid_csf"}:
        feature_key = {
            "mri_measurement": "mri",
            "pet_measurement": "pet",
            "biofluid_csf": "csf",
        }[primary]
        matched[feature_key] = _unique(matched[feature_key] + measurement_columns[:12])

    try:
        rows = _count_csv_rows(csv_path, encoding)
        read_status = "ok"
        read_error = None
    except Exception as exc:
        rows = None
        read_status = "partial"
        read_error = type(exc).__name__

    missing_rates = sample.isna().mean().sort_values(ascending=False).head(10)
    return {
        "file": _display_path(csv_path),
        "rows": rows,
        "columns": len(columns),
        "read_status": read_status,
        "read_error": read_error,
        "primary_category": primary,
        "matched_categories": categories,
        "id_columns": _match_candidates(columns, ID_CANDIDATES),
        "time_columns": _match_candidates(columns, VISIT_CANDIDATES),
        "diagnosis_columns": _match_candidates(columns, DIAGNOSIS_CANDIDATES),
        "matched_features": matched,
        "missing_rate_top10": {
            str(column): round(float(rate), 4) for column, rate in missing_rates.items()
        },
        "recommended_use": _recommended_use(primary),
    }


def build_modality_availability(inventory: Iterable[dict]) -> dict:
    """Aggregate file-level classifications into modality availability."""
    availability = {
        key: {"available": False, "files": [], "matched_columns": []}
        for key in OUTPUT_MODALITIES
    }
    for record in inventory:
        if record.get("read_status") == "failed":
            continue
        categories = set(record.get("matched_categories", []))
        for output_key, (category, feature_key) in OUTPUT_MODALITIES.items():
            if category not in categories:
                continue
            entry = availability[output_key]
            entry["available"] = True
            entry["files"].append(record["file"])
            if feature_key == "diagnosis":
                columns = record.get("diagnosis_columns", [])
            else:
                columns = record.get("matched_features", {}).get(feature_key, [])
            entry["matched_columns"].extend(columns)

    for entry in availability.values():
        entry["files"] = _unique(entry["files"])
        entry["matched_columns"] = _unique(entry["matched_columns"])
    return availability


def write_inventory_outputs(
    inventory: list[dict],
    raw_dir: str | Path,
    output_md: str | Path,
    output_json: str | Path = "outputs/metrics/adni_file_inventory.json",
    availability_json: str | Path = "outputs/metrics/adni_modality_availability.json",
) -> dict:
    """Write Markdown and JSON structure-only inventory outputs."""
    availability = build_modality_availability(inventory)
    output_md_path = Path(output_md)
    output_json_path = Path(output_json)
    availability_path = Path(availability_json)
    for path in (output_md_path, output_json_path, availability_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    availability_path.write_text(
        json.dumps(availability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_md_path.write_text(
        render_inventory_markdown(inventory, availability, raw_dir), encoding="utf-8"
    )
    return availability


def render_inventory_markdown(
    inventory: list[dict], availability: dict, raw_dir: str | Path
) -> str:
    """Render a compact report that contains no row-level source values."""
    readable = sum(item.get("read_status") != "failed" for item in inventory)
    failed = len(inventory) - readable
    lines = [
        "# ADNI File Inventory",
        "",
        f"Scanned directory: `{raw_dir}`",
        f"CSV files found: {len(inventory)}",
        f"Readable CSV files: {readable}",
        f"Unreadable files: {failed}",
        "",
        "## Available modalities",
        "",
        "| Modality | Available | Files | Key columns |",
        "|---|---|---|---|",
    ]
    for modality, entry in availability.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    modality,
                    "yes" if entry["available"] else "no",
                    _compact(entry["files"], 4),
                    _compact(entry["matched_columns"], 8),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## File-level inventory",
            "",
            "| File | Category | Rows | Columns | ID cols | Time cols | Key matched columns | Recommended use |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for item in inventory:
        matched_columns = []
        for values in item.get("matched_features", {}).values():
            matched_columns.extend(values)
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(item["file"]),
                    item["primary_category"],
                    str(item["rows"]) if item["rows"] is not None else "N/A",
                    str(item["columns"]),
                    _compact(item.get("id_columns", []), 4),
                    _compact(item.get("time_columns", []), 5),
                    _compact(_unique(matched_columns), 8),
                    item["recommended_use"],
                ]
            )
            + " |"
        )

    recommended = [
        ("PET measurements", "pet_measurement"),
        ("Biofluid / CSF measurements", "biofluid_csf"),
        ("MRI measurements", "mri_measurement"),
        ("Cognitive assessments", "cognitive"),
    ]
    lines.extend(["", "## Missing recommended categories", ""])
    for label, key in recommended:
        status = "found" if availability[key]["available"] else "not found"
        lines.append(f"- {label}: {status}")

    next_steps = next_download_recommendations(availability)
    lines.extend(["", "## Next recommended downloads", ""])
    if next_steps:
        lines.extend(f"- {step}" for step in next_steps)
    else:
        lines.append("- No missing second-batch modality category was detected.")
    lines.extend(
        [
            "",
            "> This inventory contains structural metadata only. No row-level source data is included.",
            "",
        ]
    )
    return "\n".join(lines)


def next_download_recommendations(availability: dict) -> list[str]:
    routes = {
        "pet_measurement": "Please check: Study Files -> Imaging -> PET Image Analysis.",
        "biofluid_csf": "Please check: Biospecimen -> Biospecimen Results.",
        "mri_measurement": "Please check: Study Files -> Imaging -> MR Image Analysis.",
        "cognitive": "Please check: Study Files -> Assessments -> Neuropsychological.",
    }
    return [
        route
        for key, route in routes.items()
        if not availability.get(key, {}).get("available", False)
    ]


def _read_sample(path: Path) -> tuple[pd.DataFrame, str]:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, nrows=50, encoding=encoding, low_memory=False), encoding
        except Exception as exc:
            errors.append(exc)
    raise errors[-1]


def _count_csv_rows(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for _ in reader)


def _matched_features(columns: list[str]) -> dict[str, list[str]]:
    cognitive = _match_candidates(columns, COGNITIVE_CANDIDATES)
    cognitive.extend(_prefix_matches(columns, COGNITIVE_PREFIXES, limit=24))
    return {
        "demographic": _match_candidates(columns, DEMOGRAPHIC_CANDIDATES),
        "cognitive": _unique(cognitive),
        "mri": _match_candidates(columns, MRI_CANDIDATES),
        "pet": _unique(
            _match_candidates(columns, PET_CANDIDATES)
            + _pattern_matches(columns, PET_PATTERNS)
        ),
        "csf": _unique(
            _match_candidates(columns, CSF_CANDIDATES)
            + _pattern_matches(columns, CSF_PATTERNS)
        ),
        "apoe": _match_candidates(columns, APOE_CANDIDATES),
        "data_dictionary": _match_candidates(columns, DICTIONARY_CANDIDATES),
    }


def _classify(
    path: Path,
    columns: list[str],
    matched: dict[str, list[str]],
    measurement_columns: list[str],
) -> tuple[list[str], str]:
    normalized = {_normalize(column) for column in columns}
    text = " ".join([str(path), *columns]).upper()
    path_text = str(path).upper()
    strong_diagnosis = bool(
        normalized
        & {
            _normalize(name)
            for name in ["DX", "DX_bl", "DXCHANGE", "DIAGNOSIS", "DIAGNOSISCN"]
        }
    )
    has_id = bool(_match_candidates(columns, ID_CANDIDATES))
    has_visit = bool(_match_candidates(columns, VISIT_CANDIDATES))
    dictionary = len(matched["data_dictionary"]) >= 3 or _contains_any(
        path_text, DICTIONARY_KEYWORDS
    )
    diagnosis = has_id and has_visit and strong_diagnosis
    demographics = len(matched["demographic"]) >= 2
    apoe = bool(matched["apoe"])
    cognitive_signal_count = sum(
        1 for column in columns if str(column).upper().startswith(COGNITIVE_PREFIXES)
    )
    cognitive = bool(matched["cognitive"]) or cognitive_signal_count >= 2

    pet_specific = [name for name in PET_KEYWORDS if name != "TAU"]
    fluid_specific = [name for name in CSF_KEYWORDS if name != "TAU"]
    pet = bool(matched["pet"]) or _contains_any(text, pet_specific)
    non_tau_csf_matches = [
        column for column in matched["csf"] if _normalize(column) != _normalize("TAU")
    ]
    csf = bool(non_tau_csf_matches) or _contains_any(path_text, fluid_specific)
    if _contains_keyword(text, "TAU") and not (pet or csf):
        csf = _contains_any(path_text, ["PLASMA", "CSF", "BIOFLUID", "BIOMARKER"])
        pet = not csf and _contains_any(path_text, ["PET", "AV1451", "FLORTAUCIPIR"])

    mri_exact = bool(matched["mri"])
    mri_path = _contains_any(path_text, MRI_KEYWORDS)
    mri_column_terms = sum(
        _contains_any(str(column).upper(), [*MRI_KEYWORDS[1:], "HIPPO"])
        for column in columns
    )
    mri = mri_exact or mri_path or (
        mri_column_terms >= 2 and not pet and not cognitive
    )

    biomarker_count = sum(
        len(matched[key]) for key in ("mri", "pet", "csf", "apoe")
    )
    merged = (
        _normalize("RID") in normalized
        and bool(normalized & {_normalize("VISCODE"), _normalize("VISCODE2")})
        and _normalize("DX") in normalized
        and len(_match_candidates(columns, COGNITIVE_CANDIDATES)) >= 2
        and biomarker_count >= 2
    )

    image_metadata = bool(normalized & {_normalize(name) for name in IMAGE_METADATA_COLUMNS})
    has_measurements = bool(measurement_columns)
    mri_category = "mri_measurement" if has_measurements else "mri_manifest_or_image_metadata"
    pet_category = "pet_measurement" if has_measurements else "pet_manifest_or_image_metadata"

    categories = []
    if merged:
        categories.append("merged_adnimerge")
    if dictionary:
        categories.append("data_dictionary")
    if diagnosis:
        categories.append("diagnosis")
    if demographics:
        categories.append("demographics")
    if apoe:
        categories.append("apoe")
    if cognitive:
        categories.append("cognitive")
    if mri:
        categories.append(mri_category if image_metadata or has_measurements else "mri_measurement")
    if pet:
        categories.append(pet_category if image_metadata or has_measurements else "pet_measurement")
    if csf:
        categories.append("biofluid_csf")
    categories = _unique(categories)

    if merged:
        primary = "merged_adnimerge"
    elif dictionary:
        primary = "data_dictionary"
    elif csf and (not pet or _contains_any(path_text, fluid_specific)):
        primary = "biofluid_csf"
    elif pet:
        primary = pet_category
    elif mri:
        primary = mri_category
    elif diagnosis:
        primary = "diagnosis"
    elif demographics:
        primary = "demographics"
    elif apoe:
        primary = "apoe"
    elif cognitive:
        primary = "cognitive"
    else:
        primary = "unknown"
    if not categories:
        categories = ["unknown"]
    return categories, primary


def _numeric_measurement_columns(sample: pd.DataFrame) -> list[str]:
    excluded = {
        _normalize(name)
        for name in ID_CANDIDATES
        + VISIT_CANDIDATES
        + DIAGNOSIS_CANDIDATES
        + list(IMAGE_METADATA_COLUMNS)
        + [
            "SITEID",
            "ID",
            "LONISID",
            "RECNO",
            "RUNDATE",
            "PROCESSDATE",
            "VERSION",
            "STATUS",
            "update_stamp",
        ]
    }
    result = []
    for column in sample.columns:
        normalized_column = _normalize(str(column))
        if normalized_column in excluded or any(
            token in normalized_column
            for token in ("STATUS", "WARNING", "QCFLAG", "PROCESSDATE", "RUNDATE")
        ):
            continue
        series = sample[column].dropna()
        if series.empty:
            continue
        numeric_fraction = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_fraction >= 0.8:
            result.append(str(column))
    return result


def _match_candidates(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    candidate_keys = {_normalize(candidate) for candidate in candidates}
    return [str(column) for column in columns if _normalize(str(column)) in candidate_keys]


def _prefix_matches(
    columns: Iterable[str], prefixes: tuple[str, ...], limit: int
) -> list[str]:
    matched = [str(column) for column in columns if str(column).upper().startswith(prefixes)]
    return matched[:limit]


def _pattern_matches(columns: Iterable[str], patterns: Iterable[str]) -> list[str]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [
        str(column)
        for column in columns
        if any(pattern.search(str(column)) for pattern in compiled)
    ]


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in keywords)


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword.upper() in text.upper()


def _normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9\u0370-\u03ff]", "", value.upper())


def _recommended_use(primary: str) -> str:
    return {
        "merged_adnimerge": "primary_multimodal_table",
        "data_dictionary": "data_dictionary_reference",
        "diagnosis": "diagnosis_table",
        "demographics": "demographic_table",
        "apoe": "genetic_table",
        "cognitive": "cognitive_assessment_table",
        "mri_measurement": "mri_measurement_table",
        "pet_measurement": "pet_measurement_table",
        "biofluid_csf": "biofluid_biomarker_table",
        "mri_manifest_or_image_metadata": "image_metadata_only",
        "pet_manifest_or_image_metadata": "image_metadata_only",
        "unknown": "manual_review",
    }.get(primary, "manual_review")


def _failed_record(path: Path, exc: Exception) -> dict:
    return {
        "file": _display_path(path),
        "rows": None,
        "columns": 0,
        "read_status": "failed",
        "read_error": type(exc).__name__,
        "primary_category": "unknown",
        "matched_categories": ["unknown"],
        "id_columns": [],
        "time_columns": [],
        "diagnosis_columns": [],
        "matched_features": {
            "demographic": [],
            "cognitive": [],
            "mri": [],
            "pet": [],
            "csf": [],
            "apoe": [],
            "data_dictionary": [],
        },
        "missing_rate_top10": {},
        "recommended_use": "manual_review",
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _compact(values: Iterable[str], limit: int) -> str:
    items = list(values)
    if not items:
        return "-"
    visible = [_escape_md(Path(item).name if "/" in item or "\\" in item else item) for item in items[:limit]]
    if len(items) > limit:
        visible.append(f"+{len(items) - limit} more")
    return ", ".join(visible)


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
