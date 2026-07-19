from __future__ import annotations
import json
import streamlit as st
from dashboard.dashboard_utils import OUTPUT_DIR, language_selector, read_csv, tr

lang=language_selector();st.title(tr("d3_features",lang))
audit=OUTPUT_DIR/"phase_d"/"audit"
profiles={}
for name in ("d3_core","compact_d3_core","clinical_d3_core"):
 path=audit/f"{name}_features.json";profiles[name]=json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
cols=st.columns(3)
for column,(name,features) in zip(cols,profiles.items()):column.metric(name,len(features))
selected=st.segmented_control("Profile",list(profiles),default="clinical_d3_core")
st.dataframe({"feature":profiles.get(selected,[])},use_container_width=True)
compatibility=read_csv(audit/"d3_feature_profile_compatibility.csv")
if not compatibility.empty:st.dataframe(compatibility,use_container_width=True)
