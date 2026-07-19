from __future__ import annotations

from html import escape

import streamlit as st


def metric_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="ad-metric-note">{escape(note)}</div>' if note else ""
    st.markdown(
        '<div class="ad-metric-card">'
        f'<div class="ad-metric-label">{escape(label)}</div>'
        f'<div class="ad-metric-value">{escape(value)}</div>'
        f"{note_html}</div>",
        unsafe_allow_html=True,
    )

