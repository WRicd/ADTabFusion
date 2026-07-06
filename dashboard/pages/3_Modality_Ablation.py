from __future__ import annotations

import streamlit as st

from dashboard_utils import OUTPUT_DIR, image_path, language_selector, read_csv, show_dataframe_or_warning, tr


lang = language_selector()
st.title(tr("modality_ablation", lang))
summary = read_csv(OUTPUT_DIR / "metrics" / "modality_ablation_summary.csv")
show_dataframe_or_warning(summary, lang)
if not summary.empty and "macro_f1_mean" in summary.columns:
    st.bar_chart(summary.set_index("group")["macro_f1_mean"])
fig = image_path("modality_ablation_bar.png")
if fig.exists():
    st.image(str(fig))
