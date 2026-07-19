from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("temporal_validation",lang));st.info(tr("primary_notice",lang))
result=read_csv(OUTPUT_DIR/"phase_d"/"temporal_validation"/"transition_model_results.csv")
ablation=read_csv(OUTPUT_DIR/"phase_d"/"internal_validation"/"transition_ablation.csv")
st.dataframe(result,use_container_width=True);st.dataframe(ablation,use_container_width=True)
figure=OUTPUT_DIR/"phase_d"/"figures"/"transition_ablation.png"
if figure.exists():st.image(str(figure),use_container_width=True)
