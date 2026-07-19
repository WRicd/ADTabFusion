from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.artifacts import safe_artifact_name


def artifact_table(records: list[dict[str, object]]) -> None:
    sanitized: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        if "path" in row:
            row["path"] = safe_artifact_name(str(row["path"]))
        if "sha256" in row and row["sha256"]:
            row["sha256"] = str(row["sha256"])[:12]
        sanitized.append(row)
    st.dataframe(pd.DataFrame(sanitized), use_container_width=True, hide_index=True)

