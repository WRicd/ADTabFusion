from __future__ import annotations

import streamlit as st

from dashboard_utils import (
    OUTPUT_DIR,
    available_modalities,
    display_modalities,
    language_selector,
    read_json,
    tr,
)


st.set_page_config(page_title="AD-TabFusion", layout="wide")
lang = language_selector()
st.title("AD-TabFusion")

audit = read_json(OUTPUT_DIR / "reports" / "data_audit.json")
active, unavailable = available_modalities()

st.subheader(tr("overview", lang))
col1, col2, col3 = st.columns(3)
col1.metric(tr("rows", lang), audit.get("n_rows", "N/A"))
col2.metric(tr("subjects", lang), audit.get("n_subjects", "N/A"))
col3.metric(tr("columns", lang), audit.get("n_columns", "N/A"))

st.write(tr("active_modalities", lang), display_modalities(active, lang) if active else tr("not_generated", lang))
st.write(
    tr("unavailable_modalities", lang),
    display_modalities(unavailable, lang) if unavailable else tr("none_detected", lang),
)
st.write(tr("task_modes", lang))
st.info(tr("dashboard_info", lang))

st.subheader(tr("limitations", lang))
st.markdown(tr("limitations_body", lang))
