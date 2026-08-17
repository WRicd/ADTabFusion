"""Altair rendering helpers.

Two rules are enforced here rather than left to each view:

* Charts follow the active colour mode. The Vega config is rebuilt per render
  from the theme tokens, so the canvas never fights the page surface.
* Diagnosis identity is never colour-alone. The validated palette carries an
  advisory in each mode (light: MCI contrast 2.11:1; dark: MCI/AD CVD dE 6.2),
  and ``value_labels`` plus an always-present legend discharge both.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.theme import (
    CLASS_ORDER,
    class_colors,
    palette,
    sequential_range,
    series_colors,
)


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Apply mode-aware typography, grid, and surface settings."""
    tokens = palette()
    return (
        chart.configure(
            background="transparent",
            font="Inter, Segoe UI, Arial, sans-serif",
        )
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=tokens["border"],
            gridColor=tokens["grid"],
            tickColor=tokens["border"],
            labelColor=tokens["muted"],
            titleColor=tokens["ink"],
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=tokens["muted"],
            titleColor=tokens["ink"],
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_title(color=tokens["ink"])
    )


def show_chart(chart: alt.Chart) -> None:
    """Render a chart at container width using the active theme."""
    st.altair_chart(style_chart(chart), width="stretch", theme=None)


def class_color_scale(legend_title: str | None = None) -> alt.Color:
    """Colour encoding for CN/MCI/AD with a legend always present.

    A legend is not optional here: it is the standing half of the secondary
    encoding the validated palette requires.
    """
    colors = class_colors()
    return alt.Color(
        "Class:N",
        scale=alt.Scale(domain=CLASS_ORDER, range=[colors[name] for name in CLASS_ORDER]),
        legend=alt.Legend(orient="top", title=legend_title, symbolType="square"),
    )


def sequential_color(field: str, legend: alt.Legend | None = None) -> alt.Color:
    """Single-hue magnitude encoding (light -> dark), never a rainbow."""
    return alt.Color(field, scale=alt.Scale(range=sequential_range()), legend=legend)


def series_color(field: str, domain: list[str], legend_title: str | None = None) -> alt.Color:
    """Ordered categorical encoding for non-diagnosis series."""
    hues = series_colors()
    if len(domain) > len(hues):
        raise ValueError(
            f"{len(domain)} series exceeds the {len(hues)} defined slots; "
            "fold the remainder into 'Other' or facet instead of cycling hues."
        )
    return alt.Color(
        field,
        scale=alt.Scale(domain=domain, range=hues[: len(domain)]),
        legend=alt.Legend(orient="top", title=legend_title),
    )


def value_labels(
    chart: alt.Chart,
    text_field: str,
    fmt: str = ".3f",
    dx: int = 0,
    dy: int = 0,
    align: str = "left",
) -> alt.Chart:
    """Direct value labels in text ink, not the series colour.

    These are the secondary encoding that makes the validated palette legal in
    both modes -- dropping them would leave identity resting on colour alone.
    """
    return chart.mark_text(
        align=align,
        baseline="middle",
        dx=dx,
        dy=dy,
        fontSize=11,
        color=palette()["muted"],
    ).encode(text=alt.Text(f"{text_field}:Q", format=fmt))


def table_view(frame: pd.DataFrame, label: str = "View as table") -> None:
    """Collapsed table beneath a chart.

    The light-mode MCI amber falls below 3:1 against the surface, which the
    palette rules allow only with relief; this is that relief, and it doubles
    as the screen-reader path for every chart.
    """
    with st.expander(label):
        st.dataframe(frame, width="stretch", hide_index=True)
