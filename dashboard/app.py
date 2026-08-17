from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run` puts this file's own folder on sys.path, not the project
# root, so `import dashboard.*` cannot resolve without help. Adding the root
# here lets the app be launched from any working directory, and matches the
# bootstrap the scripts/ entry points already use.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.i18n import bilingual, language_selector  # noqa: E402
from dashboard.theme import appearance_hint, apply_theme  # noqa: E402

st.set_page_config(page_title="AD-TabFusion", page_icon=None, layout="wide")

st.sidebar.markdown("### AD-TabFusion")
lang = language_selector()
# Colour mode follows Streamlit's own theme (top-right menu). That is the only
# switch that can repaint canvas-rendered widgets such as st.dataframe, so the
# custom layer reads it rather than maintaining a competing control.
apply_theme()
appearance_hint(lang)

st.sidebar.caption(
    bilingual(
        "AD-TabFusion Dashboard",
        lang,
    )
)

pages = [
    st.Page(
        "views/executive_summary.py",
        title=bilingual("执行摘要", "Executive Summary", lang),
        url_path="executive",
        default=True,
    ),
    st.Page("views/data_cohort.py", title=bilingual("数据与队列", "Data & Cohort", lang), url_path="data-cohort"),
    st.Page(
        "views/scientific_guardrails.py",
        title=bilingual("科研护栏", "Scientific Guardrails", lang),
        url_path="guardrails",
    ),
    st.Page(
        "views/transition_aware.py",
        title=bilingual("转归感知模型", "Transition-Aware Model", lang),
        url_path="transition",
    ),
    st.Page(
        "views/mci_progression.py", title=bilingual("MCI 进展风险", "MCI Progression Risk", lang), url_path="mci-risk"
    ),
    st.Page(
        "views/calibration_uncertainty.py",
        title=bilingual("校准与不确定性", "Calibration & Uncertainty", lang),
        url_path="uncertainty",
    ),
    st.Page(
        "views/external_replay.py", title=bilingual("外部回放", "External Replay", lang), url_path="external-replay"
    ),
    st.Page(
        "views/reproducibility.py", title=bilingual("可复现性", "Reproducibility", lang), url_path="reproducibility"
    ),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
