from __future__ import annotations

from html import escape

import streamlit as st


def interpretation_box(text: str) -> None:
    st.markdown(f'<div class="ad-interpretation">{escape(text)}</div>', unsafe_allow_html=True)

