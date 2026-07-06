from __future__ import annotations

import streamlit as st

from dashboard_utils import OUTPUT_DIR, image_path, language_selector, read_csv, show_dataframe_or_warning, tr


lang = language_selector()
st.title(tr("missing_modality", lang))
df = read_csv(OUTPUT_DIR / "metrics" / "missing_modality_results.csv")
show_dataframe_or_warning(df, lang)
if not df.empty and "macro_f1_drop" in df.columns:
    st.bar_chart(df.dropna(subset=["macro_f1_drop"]).set_index("masked_modality")["macro_f1_drop"])
fig = image_path("missing_modality_drop.png")
if fig.exists():
    st.image(str(fig))
