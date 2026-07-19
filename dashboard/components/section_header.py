from __future__ import annotations

from html import escape

import streamlit as st


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="ad-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="ad-subtitle">{escape(subtitle)}</p>', unsafe_allow_html=True)


def section_header(title: str, description: str = "") -> None:
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="ad-section"><h2>{escape(title)}</h2>{description_html}</div>',
        unsafe_allow_html=True,
    )

