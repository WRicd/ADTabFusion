from __future__ import annotations
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR,language_selector,read_csv,tr
lang=language_selector();st.title(tr("transition_matrix",lang))
frame=read_csv(OUTPUT_DIR/"phase_d"/"internal_validation"/"transition_matrix.csv")
if frame.empty:st.warning(tr("empty_table",lang))
else:
 split=st.segmented_control("Split",frame["split"].unique().tolist(),default=frame["split"].unique().tolist()[0])
 table=frame[frame.split==split].pivot(index="SOURCE_DX",columns="FUTURE_DX",values="count").fillna(0).astype(int)
 st.dataframe(table,use_container_width=True);st.bar_chart(frame[frame.split==split],x="SOURCE_DX",y="count",color="FUTURE_DX")
