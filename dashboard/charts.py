from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard.theme import COLORS


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(
            background=COLORS["white"],
            font="Segoe UI",
        )
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=COLORS["border"],
            gridColor="#E9EEF2",
            labelColor=COLORS["muted"],
            titleColor=COLORS["ink"],
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=COLORS["muted"],
            titleColor=COLORS["ink"],
            labelFontSize=11,
            titleFontSize=11,
        )
    )


def show_chart(chart: alt.Chart) -> None:
    st.altair_chart(style_chart(chart), use_container_width=True, theme=None)
