from __future__ import annotations

import streamlit as st

from dashboard.i18n import bilingual, language_selector
from dashboard.theme import apply_theme


st.set_page_config(page_title="AD-TabFusion", page_icon=None, layout="wide")
apply_theme()
lang = language_selector()

st.sidebar.markdown("### AD-TabFusion")
st.sidebar.caption(
    bilingual(
        "冻结证据的只读科研看板",
        "Read-only dashboard for frozen evidence",
        lang,
    )
)

pages = [
    st.Page("views/executive_summary.py", title=bilingual("执行摘要", "Executive Summary", lang), url_path="executive", default=True),
    st.Page("views/data_cohort.py", title=bilingual("数据与队列", "Data & Cohort", lang), url_path="data-cohort"),
    st.Page("views/scientific_guardrails.py", title=bilingual("科研护栏", "Scientific Guardrails", lang), url_path="guardrails"),
    st.Page("views/transition_aware.py", title=bilingual("转归感知模型", "Transition-Aware Model", lang), url_path="transition"),
    st.Page("views/mci_progression.py", title=bilingual("MCI 进展风险", "MCI Progression Risk", lang), url_path="mci-risk"),
    st.Page("views/calibration_uncertainty.py", title=bilingual("校准与不确定性", "Calibration & Uncertainty", lang), url_path="uncertainty"),
    st.Page("views/external_replay.py", title=bilingual("外部回放", "External Replay", lang), url_path="external-replay"),
    st.Page("views/reproducibility.py", title=bilingual("可复现性", "Reproducibility", lang), url_path="reproducibility"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
