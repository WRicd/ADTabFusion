from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("mci_conversion",lang));st.info(tr("primary_notice",lang))
metrics=read_csv(OUTPUT_DIR/"phase_d"/"temporal_validation"/"mci_risk_metrics.csv")
summary=read_csv(OUTPUT_DIR/"phase_d"/"cohorts"/"mci_landmark_summary.csv")
st.dataframe(metrics,use_container_width=True);st.dataframe(summary,use_container_width=True)
if not metrics.empty:st.line_chart(metrics,x="horizon_months",y=["roc_auc","pr_auc","balanced_accuracy"])
figure=OUTPUT_DIR/"phase_d"/"figures"/"mci_risk_by_horizon.png"
if figure.exists():st.image(str(figure),use_container_width=True)
