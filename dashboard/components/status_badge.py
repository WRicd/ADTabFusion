from __future__ import annotations

from html import escape

import streamlit as st


def status_badge(label: str, tone: str = "info") -> None:
    tone = tone if tone in {"success", "warning", "info"} else "info"
    st.markdown(
        f'<span class="ad-badge ad-badge-{tone}">{escape(label)}</span>',
        unsafe_allow_html=True,
    )

