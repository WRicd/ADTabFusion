from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ALWAYS_EXCLUDE = {
    "DX": "Target diagnosis label; including it would directly leak the answer.",
    "DXCHANGE": "Diagnosis transition code derived from diagnosis state; leaks target information.",
}


def build_feature_blacklist(config: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, str]:
    """Return leakage-prone columns to remove from model features."""
    task_mode = config.get("data", {}).get("task_mode", "all_visits")
    blacklist = dict(ALWAYS_EXCLUDE)
    if task_mode == "baseline_only":
        blacklist["DX_bl"] = "Baseline diagnosis is diagnosis-derived and overlaps with the target."
        blacklist["VISCODE"] = "Baseline-only task filters to baseline visits, so visit code is not a model feature."
    else:
        blacklist["DX_bl"] = "Baseline diagnosis is diagnosis-derived; excluded from main all-visits model."
        if config.get("leakage", {}).get("exclude_visit_code", True):
            blacklist["VISCODE"] = "Visit code may encode follow-up timing and is excluded by default."

    if df is not None:
        blacklist = {col: reason for col, reason in blacklist.items() if col in df.columns}
    return blacklist


def remove_blacklisted_features(
    features: list[str], blacklist: dict[str, str]
) -> list[str]:
    """Remove leakage columns from a feature list while preserving order."""
    blocked = set(blacklist)
    return [feature for feature in features if feature not in blocked]


def write_leakage_report(
    config: dict[str, Any], df: pd.DataFrame, output_dir: str | Path
) -> dict[str, str]:
    """Persist leakage blacklist JSON and a markdown explanation."""
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    reports_dir = output / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    blacklist = build_feature_blacklist(config, df)
    (metrics_dir / "feature_blacklist.json").write_text(
        json.dumps(blacklist, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    task_mode = config.get("data", {}).get("task_mode", "all_visits")
    lines = [
        "# Leakage Check",
        "",
        f"Task mode: `{task_mode}`",
        "",
        "## Excluded Columns",
        "",
    ]
    if blacklist:
        for col, reason in blacklist.items():
            lines.append(f"- `{col}`: {reason}")
    else:
        lines.append("No configured leakage columns were present in this CSV.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "`FAQ_bl` is treated as a baseline covariate, not as current-visit FAQ.",
            "Subject-level splitting is required for every task mode.",
        ]
    )
    (reports_dir / "leakage_check.md").write_text("\n".join(lines), encoding="utf-8")
    return blacklist

