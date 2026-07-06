from __future__ import annotations

import streamlit as st

from dashboard_utils import OUTPUT_DIR, language_selector, read_csv, show_dataframe_or_warning, tr


lang = language_selector()
st.title(tr("error_cases", lang))
errors = read_csv(OUTPUT_DIR / "reports" / "error_cases.csv")
summary = read_csv(OUTPUT_DIR / "reports" / "error_confusion_summary.csv")
st.subheader(tr("high_confidence_errors", lang))
show_dataframe_or_warning(errors, lang)
if not errors.empty:
    st.download_button(
        tr("download_error_cases", lang),
        errors.to_csv(index=False),
        "error_cases.csv",
        "text/csv",
    )
st.subheader(tr("per_class_confusion", lang))
show_dataframe_or_warning(summary, lang)
