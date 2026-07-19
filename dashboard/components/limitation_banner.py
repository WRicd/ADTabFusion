from __future__ import annotations

from html import escape

import streamlit as st


def limitation_banner(text: str, danger: bool = False) -> None:
    modifier = " ad-banner-danger" if danger else ""
    st.markdown(f'<div class="ad-banner{modifier}">{escape(text)}</div>', unsafe_allow_html=True)

