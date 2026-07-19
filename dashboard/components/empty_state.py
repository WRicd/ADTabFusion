from __future__ import annotations

from html import escape

import streamlit as st


def empty_state(message: str, container=st) -> None:
    container.markdown(f'<div class="ad-empty">{escape(message)}</div>', unsafe_allow_html=True)

