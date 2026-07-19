from __future__ import annotations

import streamlit as st

from dashboard.dashboard_utils import OUTPUT_DIR, image_path, language_selector, read_json, tr


lang = language_selector()
st.title(tr("data_audit", lang))
audit = read_json(OUTPUT_DIR / "reports" / "data_audit.json")
if not audit:
    st.warning(tr("audit_missing", lang))
else:
    st.json(
        {
            tr("rows", lang): audit.get("n_rows"),
            tr("subjects", lang): audit.get("n_subjects"),
            tr("columns", lang): audit.get("n_columns"),
            "label_distribution": audit.get("label_distribution"),
        }
    )
    st.subheader(tr("missingness", lang))
    st.dataframe(audit.get("missing_rate_by_column", {}), use_container_width=True)
    fig = image_path("missing_rate_by_column.png")
    if fig.exists():
        st.image(str(fig))
