from __future__ import annotations

import re
from typing import Mapping


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

IDENTIFIER_NAMES = {"RID", "PTID", "LONIUID", "LONISID", "IMAGEUID"}
VISIT_TIME_NAMES = {
    "VISCODE",
    "VISCODE2",
    "EXAMDATE",
    "SCANDATE",
    "VISDATE",
    "MONTH_BL",
    "YEARS_BL",
}
LABEL_NAMES = {"DX", "DXCHANGE", "DIAGNOSIS"}
ADMINISTRATIVE_NAMES = {
    "D1",
    "D2",
    "SITE",
    "COLPROT",
    "ORIGPROT",
    "PHASE",
    "RUNDATE",
    "STATUS",
    "VERSION",
    "UPDATE_STAMP",
}
DEMOGRAPHIC_PREFIXES = (
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTETHCAT",
    "PTRACCAT",
    "PTMARRY",
    "PTDOB",
)
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
    "MOCA",
)


def infer_modality(
    column_name: str, metadata: Mapping[str, str] | None = None
) -> str:
    """Infer one audited modality, prioritizing dictionary source metadata."""
    metadata = metadata or {}
    name = column_name.upper()
    source = str(metadata.get("TBLNAME", "")).upper()
    description = str(metadata.get("TEXT", "")).upper()
    combined = " ".join((name, source, description))
    base = _base_name(name)

    if base in LABEL_NAMES:
        return "label"
    if base == "DX_BL":
        return "baseline_diagnosis"
    if base in IDENTIFIER_NAMES or any(name.startswith(f"{item}_") for item in IDENTIFIER_NAMES):
        return "identifier"
    if name in VISIT_TIME_NAMES or base in VISIT_TIME_NAMES or any(
        name.startswith(f"{item}_") for item in VISIT_TIME_NAMES
    ):
        return "visit_time"
    if base in ADMINISTRATIVE_NAMES or _contains_any(
        name, ("RUNDATE", "STATUS", "VERSION", "UPDATE_STAMP", "QC", "COMMENT")
    ):
        return "administrative"
    if source == "UPENNBIOMK9" or _matches(name, CSF_PATTERNS) or "ELECSYS" in combined:
        return "csf"
    if source == "DTIROI" or "DIFFUSION" in combined or "DIFFUSIVITY" in combined:
        return "mri_dti"
    if source == "BAIPETNMRC" or name.startswith("FDG"):
        return "pet_fdg"
    if source == "UCBERKELEYAV45" or _contains_any(name, ("AV45", "PIB", "FLORBETAPIR", "CENTILOID")):
        return "pet_amyloid"
    if source == "UCBERKELEYAV1451" or _matches(name, PET_PATTERNS):
        return "pet_other"
    if source in {"UCSFFSL", "UCSFFSX"} or _contains_any(
        combined, ("FREESURFER", "CORTICAL THICKNESS", "BRAIN VOLUME", "HIPPOCAMP")
    ):
        return "mri_structural"
    if base.startswith(DEMOGRAPHIC_PREFIXES):
        return "demographic"
    if base.startswith(COGNITIVE_PREFIXES):
        return "cognitive"
    if _contains_any(base, ("APOE", "APGEN", "GENOTYPE")):
        return "genetic"
    if _contains_any(combined, ("MEDICATION", "DRUG", "PRESCRIPTION")):
        return "medication"
    if _contains_any(combined, ("MEDICAL HISTORY", "FAMILY HISTORY", "HISTORY")):
        return "medical_history"
    if _contains_any(combined, ("PLASMA", "SERUM", "BIOFLUID", "BIOMARKER")):
        return "biofluid_other"
    if source == "ADNIMERGE" and _contains_any(
        base, ("VENTRICLES", "HIPPOCAMPUS", "WHOLEBRAIN", "ENTORHINAL", "FUSIFORM", "MIDTEMP", "ICV")
    ):
        return "mri_structural"
    return "unknown"


def _base_name(name: str) -> str:
    for suffix in ("_BL",):
        if name.endswith(suffix) and name != "DX_BL":
            return name[: -len(suffix)]
    return name


def _matches(value: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)
