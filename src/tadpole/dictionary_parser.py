from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DICTIONARY_FIELDS = ("FLDNAME", "TBLNAME", "TEXT", "TYPE", "UNITS")
EMPTY_VALUES = {"", "-4", "-4.0", "nan", "none"}


def load_tadpole_dictionary(path: str | Path) -> pd.DataFrame:
    """Load and normalize the TADPOLE field dictionary."""
    frame = pd.read_csv(path, low_memory=False, dtype=str).fillna("")
    missing = [column for column in DICTIONARY_FIELDS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dictionary is missing required columns: {', '.join(missing)}")
    for column in frame.columns:
        frame[column] = frame[column].map(_clean_value)
    return frame


def build_dictionary_index(dictionary: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Index dictionary rows by exact, case-insensitive field name."""
    index: dict[str, dict[str, str]] = {}
    scores: dict[str, int] = {}
    for row in dictionary.to_dict(orient="records"):
        field_name = str(row.get("FLDNAME", "")).strip()
        if not field_name or field_name.upper() == "FLDNAME":
            continue
        key = field_name.casefold()
        score = sum(bool(row.get(name, "")) for name in DICTIONARY_FIELDS[1:])
        if key not in index or score > scores[key]:
            index[key] = {name: str(row.get(name, "")) for name in DICTIONARY_FIELDS}
            scores[key] = score
    return index


def match_dictionary_columns(
    columns: Iterable[str], dictionary: pd.DataFrame
) -> dict[str, dict[str, str] | None]:
    """Return exact dictionary metadata for each data column."""
    index = build_dictionary_index(dictionary)
    return {str(column): index.get(str(column).casefold()) for column in columns}


def _clean_value(value: object) -> str:
    text = str(value).strip()
    return "" if text.casefold() in EMPTY_VALUES else text

