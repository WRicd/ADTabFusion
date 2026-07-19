from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@st.cache_data(show_spinner=False)
def load_csv(path: str | Path, required_columns: tuple[str, ...] = ()) -> tuple[pd.DataFrame, str | None]:
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = PROJECT_ROOT / artifact
    if not artifact.exists():
        return pd.DataFrame(), f"Missing artifact: {artifact.name}"
    try:
        frame = pd.read_csv(artifact)
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {artifact.name}: {exc}"
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        return pd.DataFrame(), f"{artifact.name} is missing columns: {', '.join(missing)}"
    return frame, None


@st.cache_data(show_spinner=False)
def load_json(path: str | Path) -> tuple[dict[str, Any], str | None]:
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = PROJECT_ROOT / artifact
    if not artifact.exists():
        return {}, f"Missing artifact: {artifact.name}"
    try:
        return json.loads(artifact.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, f"Could not read {artifact.name}: {exc}"


def artifact(relative_path: str) -> Path:
    return OUTPUT_ROOT / relative_path


def safe_artifact_name(path: str | Path) -> str:
    candidate = Path(str(path))
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.name


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].copy()

