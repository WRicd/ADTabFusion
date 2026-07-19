from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("risk_coverage",lang));st.info(tr("primary_notice",lang))
frame=read_csv(OUTPUT_DIR/"phase_d"/"uncertainty"/"selective_prediction.csv");st.dataframe(frame,use_container_width=True)
if not frame.empty:
 test=frame[frame.split=="locked_temporal_test"]
 st.line_chart(test,x="coverage",y=["macro_f1","accuracy","balanced_accuracy"])
figure=OUTPUT_DIR/"phase_d"/"figures"/"risk_coverage_curve.png"
if figure.exists():st.image(str(figure),use_container_width=True)
