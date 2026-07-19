from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_temporal_subject_split(
    frame: pd.DataFrame,
    subject_col: str = "RID",
    date_col: str = "EXAMDATE",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, list[Any]], pd.DataFrame]:
    work = frame[[subject_col, date_col]].copy()
    work["index_date"] = pd.to_datetime(work[date_col], errors="coerce")
    first = work.dropna(subset=[subject_col, "index_date"]).groupby(subject_col, as_index=False)["index_date"].min()
    first["_rid_sort"] = first[subject_col].astype(str)
    first = first.sort_values(["index_date", "_rid_sort"]).drop(columns="_rid_sort").reset_index(drop=True)
    n = len(first)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    first["split"] = "temporal_test"
    first.loc[: train_end - 1, "split"] = "train"
    first.loc[train_end : validation_end - 1, "split"] = "validation"
    split = {
        name: first.loc[first["split"] == name, subject_col].tolist()
        for name in ("train", "validation", "temporal_test")
    }
    return split, first


def write_temporal_split(
    split: dict[str, list[Any]], assignment: pd.DataFrame, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_ids = [str(value) for values in split.values() for value in values]
    payload = {
        "method": "first eligible source visit date, earliest 70% / next 15% / latest 15%",
        "label_free_split": True,
        "d4_accessed": False,
        "splits": split,
        "subject_count": len(all_ids),
        "subject_assignment_sha256": hashlib.sha256("\n".join(sorted(all_ids)).encode()).hexdigest(),
        "cut_dates": {
            name: {
                "min": assignment.loc[assignment["split"] == name, "index_date"].min().date().isoformat(),
                "max": assignment.loc[assignment["split"] == name, "index_date"].max().date().isoformat(),
            }
            for name in split
        },
    }
    (output / "temporal_split_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assignment.to_csv(output / "temporal_split_assignments.csv", index=False)
    lines = ["# Locked D1/D2 Temporal Subject Split", "", "> Split cutoffs use dates only; labels and D4 were not accessed.", "", "| Split | Subjects | Earliest index | Latest index |", "|---|---:|---|---|"]
    for name, ids in split.items():
        dates = payload["cut_dates"][name]
        lines.append(f"| {name} | {len(ids)} | {dates['min']} | {dates['max']} |")
    lines.extend(["", f"Assignment SHA256: `{payload['subject_assignment_sha256']}`", ""])
    (output / "temporal_split_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def assert_subject_split_disjoint(split: dict[str, list[Any]]) -> None:
    sets = {name: set(values) for name, values in split.items()}
    names = list(sets)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if not sets[name].isdisjoint(sets[other]):
                raise AssertionError(f"RID overlap between {name} and {other}")
