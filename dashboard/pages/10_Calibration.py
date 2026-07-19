from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("calibration",lang));st.info(tr("primary_notice",lang))
frame=read_csv(OUTPUT_DIR/"phase_d"/"calibration"/"calibration_results.csv");st.dataframe(frame,use_container_width=True)
if not frame.empty:
 test=frame[frame.split=="locked_temporal_test"]
 st.bar_chart(test,x="method",y=["log_loss","brier_score","ece"])
figure=OUTPUT_DIR/"phase_d"/"figures"/"reliability_diagram.png"
if figure.exists():st.image(str(figure),use_container_width=True)
