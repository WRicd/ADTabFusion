from __future__ import annotations

from pathlib import Path

import streamlit as st


COLORS = {
    "navy": "#16324F",
    "blue": "#2F6B9A",
    "teal": "#168C8C",
    "green": "#2E8B67",
    "amber": "#C78321",
    "red": "#B84A4A",
    "ink": "#17212B",
    "muted": "#5F6B76",
    "border": "#D9E2EA",
    "surface": "#F6F8FA",
    "white": "#FFFFFF",
    "cn": "#2F6B9A",
    "mci": "#C78321",
    "ad": "#B84A4A",
}

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
RADIUS = {"sm": 6, "md": 8, "lg": 12}
CLASS_ORDER = ["CN", "MCI", "AD"]
CLASS_COLORS = {label: COLORS[label.lower()] for label in CLASS_ORDER}


def apply_theme() -> None:
    css_path = Path(__file__).with_name("styles.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

