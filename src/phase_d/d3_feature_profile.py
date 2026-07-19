from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CLINICAL_CANDIDATES = [
    "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY",
    "MMSE", "ADAS13", "Ventricles", "Hippocampus", "WholeBrain",
    "Entorhinal", "Fusiform", "MidTemp", "ICV",
]


def build_d3_feature_profiles(
    train_csv: str | Path,
    d3_csv: str | Path,
    whitelist_path: str | Path,
    compact_features: list[str],
    catalog_path: str | Path,
    output_dir: str | Path,
) -> dict[str, list[str]]:
    whitelist = json.loads(Path(whitelist_path).read_text(encoding="utf-8"))
    train_columns = pd.read_csv(train_csv, nrows=0).columns.tolist()
    d3_columns = pd.read_csv(d3_csv, nrows=0).columns.tolist()
    d3_set = set(d3_columns)
    train_set = set(train_columns)
    profiles = {
        "d3_core": [feature for feature in whitelist if feature in d3_set],
        "compact_d3_core": [feature for feature in compact_features if feature in d3_set],
        "clinical_d3_core": [feature for feature in CLINICAL_CANDIDATES if feature in d3_set and feature in train_set],
    }
    required = list(dict.fromkeys([*whitelist, *compact_features, *CLINICAL_CANDIDATES]))
    train = pd.read_csv(train_csv, usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    d3 = pd.read_csv(d3_csv, usecols=lambda column: column in required, low_memory=False, na_values=["", " ", "-4", "-4.0"])
    audit_rows = []
    for feature in list(dict.fromkeys([*whitelist, *compact_features, *CLINICAL_CANDIDATES])):
        train_present = feature in train.columns
        d3_present = feature in d3.columns
        train_missing = float(train[feature].isna().mean()) if train_present else 1.0
        d3_missing = float(d3[feature].isna().mean()) if d3_present else 1.0
        train_numeric = train_present and pd.api.types.is_numeric_dtype(train[feature])
        d3_numeric = d3_present and pd.api.types.is_numeric_dtype(d3[feature])
        unseen = ""
        if train_present and d3_present and not train_numeric:
            unseen = "|".join(sorted(set(d3[feature].dropna().astype(str)) - set(train[feature].dropna().astype(str))))
        audit_rows.append(
            {
                "feature": feature, "train_present": train_present, "d3_present": d3_present,
                "train_missing_rate": train_missing, "d3_missing_rate": d3_missing,
                "missing_rate_shift": d3_missing - train_missing,
                "dtype_compatible": bool(train_present and d3_present and train_numeric == d3_numeric),
                "unseen_d3_levels": unseen,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, features in profiles.items():
        (output / f"{name}_features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    pd.DataFrame(audit_rows).to_csv(output / "d3_feature_profile_compatibility.csv", index=False)
    catalog = pd.read_csv(catalog_path)
    modality = dict(zip(catalog["column_name"].astype(str), catalog["modality"].astype(str)))
    excluded = [feature for feature in whitelist if feature not in d3_set]
    excluded_modalities = sorted({modality.get(feature, "unmapped") for feature in excluded})
    lines = [
        "# D3-Compatible Feature Profiles", "",
        "> Built from D1/D2 and D3 schemas only. D4 was not accessed.", "",
        f"- Phase A whitelist: {len(whitelist)}",
        f"- d3_core: {len(profiles['d3_core'])}",
        f"- compact_d3_core: {len(profiles['compact_d3_core'])}",
        f"- clinical_d3_core: {len(profiles['clinical_d3_core'])}",
        f"- Excluded whitelist fields: {len(excluded)}",
        f"- Excluded modalities: {', '.join(excluded_modalities)}", "",
        "## Excluded Whitelist Fields", "", *(f"- `{feature}`" for feature in excluded), "",
        "## Frozen Profiles", "",
    ]
    for name, features in profiles.items():
        lines.extend([f"### {name}", "", ", ".join(f"`{feature}`" for feature in features), ""])
    lines.extend(["## Compatibility Detail", "", "See `d3_feature_profile_compatibility.csv` for missingness shifts, dtypes, and unseen categorical levels.", ""])
    (output / "d3_feature_profile_report.md").write_text("\n".join(lines), encoding="utf-8")
    return profiles
