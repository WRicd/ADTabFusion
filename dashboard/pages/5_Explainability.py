from __future__ import annotations

import streamlit as st

from dashboard_utils import OUTPUT_DIR, image_path, language_selector, read_csv, show_dataframe_or_warning, tr


lang = language_selector()
st.title(tr("explainability", lang))
features = read_csv(OUTPUT_DIR / "metrics" / "feature_importance_best_model.csv")
modalities = read_csv(OUTPUT_DIR / "metrics" / "modality_importance_best_model.csv")
st.subheader(tr("top_features", lang))
show_dataframe_or_warning(features.head(30), lang)
fig = image_path("top_features_best_model.png")
if fig.exists():
    st.image(str(fig))
st.subheader(tr("modality_importance", lang))
show_dataframe_or_warning(modalities, lang)
mod_fig = image_path("modality_importance_best_model.png")
if mod_fig.exists():
    st.image(str(mod_fig))
