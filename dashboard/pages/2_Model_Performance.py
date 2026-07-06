from __future__ import annotations

import streamlit as st

from dashboard_utils import (
    OUTPUT_DIR,
    image_path,
    language_selector,
    read_csv,
    show_dataframe_or_warning,
    tr,
)


lang = language_selector()
st.title(tr("model_performance", lang))
summary = read_csv(OUTPUT_DIR / "metrics" / "baseline_results_summary.csv")
by_seed = read_csv(OUTPUT_DIR / "metrics" / "baseline_results_by_seed.csv")
st.subheader(tr("seed_summary", lang))
show_dataframe_or_warning(summary, lang)
if not summary.empty and "macro_f1_mean" in summary.columns:
    st.bar_chart(summary.set_index("model")["macro_f1_mean"])
st.subheader(tr("by_seed", lang))
show_dataframe_or_warning(by_seed, lang)
cm = image_path("confusion_matrix_random_forest.png")
if cm.exists():
    st.image(str(cm), caption=tr("confusion_matrix", lang))
