from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# ADNI/TADPOLE tables encode "not applicable" as -4 and also ship blank and
# single-space cells. Every reader in the project must treat all four as missing,
# otherwise numeric columns silently degrade to object dtype.
TADPOLE_NA_VALUES = ["", " ", "-4", "-4.0"]


def read_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, ensure_ascii: bool = True) -> Path:
    """Write a JSON artifact with the project's canonical two-space indent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=ensure_ascii), encoding="utf-8")
    return target


def read_tadpole_table(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a TADPOLE-derived CSV, optionally restricted to *columns*.

    The source tables are ~1900 columns wide, so callers pass the identity and
    feature columns they need instead of materializing the full frame.
    """
    if columns is None:
        return pd.read_csv(path, low_memory=False, na_values=TADPOLE_NA_VALUES)
    required = set(columns)
    return pd.read_csv(
        path,
        usecols=lambda column: column in required,
        low_memory=False,
        na_values=TADPOLE_NA_VALUES,
    )
