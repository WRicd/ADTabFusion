from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("d4_replay",lang));st.warning(tr("replay_notice",lang))
frame=read_csv(OUTPUT_DIR/"phase_d"/"d4_replay"/"d4_replay_metrics.csv");st.dataframe(frame,use_container_width=True)
