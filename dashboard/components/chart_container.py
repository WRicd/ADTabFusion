from __future__ import annotations

from html import escape

import streamlit as st


def chart_heading(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="ad-chart-title">{escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="ad-chart-caption">{escape(caption)}</div>', unsafe_allow_html=True)

