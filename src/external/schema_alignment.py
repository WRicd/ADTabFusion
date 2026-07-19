from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_groups import infer_feature_types


def align_to_frozen_schema(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return exactly the frozen feature columns, padding absent columns with NaN."""
    aligned = frame.copy()
    for feature in features:
        if feature not in aligned.columns:
            aligned[feature] = np.nan
    return aligned.loc[:, features]


def audit_d3_schema(
    train_csv: str | Path,
    d3_csv: str | Path,
    whitelist_path: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    full_feature_threshold: float = 0.95,
    required_modalities: list[str] | None = None,
) -> dict[str, Any]:
    features = json.loads(Path(whitelist_path).read_text(encoding="utf-8"))
    catalog = pd.read_csv(catalog_path)
    modality_map = dict(
        zip(catalog["column_name"].astype(str), catalog["modality"].astype(str))
    )
    train = pd.read_csv(
        train_csv,
        usecols=lambda column: column in features,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    d3 = pd.read_csv(
        d3_csv,
        usecols=lambda column: column in features,
        low_memory=False,
        na_values=["", " ", "-4", "-4.0"],
    )
    present = [feature for feature in features if feature in d3.columns]
    absent = [feature for feature in features if feature not in d3.columns]
    numeric, categorical = infer_feature_types(train, features)
    rows: list[dict[str, Any]] = []
    unseen: dict[str, list[str]] = {}
    for feature in features:
        train_rate = float(train[feature].isna().mean())
        is_present = feature in d3.columns
        d3_rate = float(d3[feature].isna().mean()) if is_present else 1.0
        train_kind = "categorical" if feature in categorical else "numeric"
        mismatch = False
        if is_present:
            if train_kind == "numeric":
                converted = pd.to_numeric(d3[feature], errors="coerce")
                source_non_missing = int(d3[feature].notna().sum())
                mismatch = source_non_missing > 0 and int(converted.notna().sum()) == 0
            else:
                known = set(train[feature].dropna().astype(str))
                new_levels = sorted(set(d3[feature].dropna().astype(str)) - known)
                if new_levels:
                    unseen[feature] = new_levels
        rows.append(
            {
                "feature": feature,
                "modality": modality_map.get(feature, "unmapped"),
                "present_in_d3": is_present,
                "train_kind": train_kind,
                "d3_dtype": str(d3[feature].dtype) if is_present else "absent",
                "severe_dtype_mismatch": mismatch,
                "train_missing_rate": train_rate,
                "d3_missing_rate": d3_rate,
                "missing_rate_shift": d3_rate - train_rate,
            }
        )
    compatibility = pd.DataFrame(rows)
    coverage: dict[str, dict[str, Any]] = {}
    for modality, group in compatibility.groupby("modality", sort=True):
        coverage[str(modality)] = {
            "feature_count": int(len(group)),
            "present_feature_count": int(group["present_in_d3"].sum()),
            "has_usable_feature": bool(group["present_in_d3"].any()),
        }
    required = required_modalities or sorted(coverage)
    modality_ok = all(coverage.get(name, {}).get("has_usable_feature", False) for name in required)
    presence_ratio = len(present) / max(len(features), 1)
    dtype_ok = not bool(compatibility["severe_dtype_mismatch"].any())
    full_compatible = presence_ratio >= full_feature_threshold and modality_ok and dtype_ok
    decision = {
        "primary_candidate": "full_hist_gradient_boosting" if full_compatible else "compact_random_forest",
        "sensitivity_candidate": "compact_random_forest" if full_compatible else "full_hist_gradient_boosting",
        "full_compatible": full_compatible,
        "full_feature_presence_ratio": presence_ratio,
        "feature_threshold": full_feature_threshold,
        "required_modalities_available": modality_ok,
        "severe_dtype_mismatch": not dtype_ok,
        "decision_frozen_before_d4": True,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    compatibility.to_csv(output / "d3_feature_compatibility.csv", index=False)
    compatibility[
        ["feature", "modality", "train_missing_rate", "d3_missing_rate", "missing_rate_shift"]
    ].to_csv(output / "d3_missingness_shift.csv", index=False)
    (output / "d3_modality_coverage.json").write_text(
        json.dumps({"modalities": coverage, "deployment": decision}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# D3 Schema Compatibility Audit",
        "",
        "> This audit was completed without reading D4 labels.",
        "",
        f"- Frozen whitelist features: {len(features)}",
        f"- Present in D3: {len(present)} ({presence_ratio:.1%})",
        f"- Absent in D3: {len(absent)}",
        f"- Severe dtype mismatches: {int(compatibility['severe_dtype_mismatch'].sum())}",
        f"- Primary deployment: `{decision['primary_candidate']}`",
        f"- Sensitivity deployment: `{decision['sensitivity_candidate']}`",
        "",
        "## Decision",
        "",
        "Full-primary compatibility requires at least 95% feature presence, every required modality, and no severe dtype mismatch.",
        "The deployment order above is frozen before any D4 evaluation.",
        "",
        "## Absent Features",
        "",
        *(f"- `{feature}`" for feature in absent),
        "",
        "## Modality Coverage",
        "",
        "| Modality | Present | Total | Usable |",
        "|---|---:|---:|---|",
    ]
    for modality, values in coverage.items():
        report.append(
            f"| {modality} | {values['present_feature_count']} | {values['feature_count']} | "
            f"{'yes' if values['has_usable_feature'] else 'no'} |"
        )
    report.extend(["", "## Unseen Categorical Levels", ""])
    if unseen:
        report.extend(f"- `{key}`: {', '.join(values)}" for key, values in unseen.items())
    else:
        report.append("None detected.")
    report.extend(
        [
            "",
            "## Frozen Alignment Policy",
            "",
            "Absent columns are inserted in frozen training order with all values set to NaN. The fitted training imputer handles them; no D3 transformer is fitted or updated.",
            "",
        ]
    )
    (output / "d3_schema_report.md").write_text("\n".join(report), encoding="utf-8")
    return {"compatibility": compatibility, "coverage": coverage, "decision": decision}
