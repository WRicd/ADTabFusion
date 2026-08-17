"""Design tokens for the dashboard.

The diagnosis palette is **validated, not chosen by eye**. Both modes were
checked with the data-viz palette validator (lightness band, chroma floor,
colour-vision-deficiency separation, normal-vision floor, surface contrast):

    light  CN #2a78d6  MCI #eda100  AD #e34948   surface #ffffff
    dark   CN #3987e5  MCI #c98500  AD #e34948   surface #1a1a19

Both modes clear every hard gate but carry one advisory each, and both
advisories are discharged the same way -- with **direct labels on the marks**:

* light: MCI amber sits at 2.11:1 against the light surface (below 3:1), so
  identity must not rest on the fill alone.
* dark: MCI/AD separate by CVD dE 6.2, inside the 6-8 band that is only
  permitted alongside a secondary encoding.

``dashboard.charts`` provides ``class_color_scale`` and ``value_labels`` so
views satisfy this by construction. Re-run the validator before editing any
hex value here.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# -- Diagnosis classes ---------------------------------------------------
CLASS_ORDER = ["CN", "MCI", "AD"]

CLASS_COLORS_LIGHT = {"CN": "#2a78d6", "MCI": "#eda100", "AD": "#e34948"}
CLASS_COLORS_DARK = {"CN": "#3987e5", "MCI": "#c98500", "AD": "#e34948"}

# Sequential ramp for magnitude (confusion counts): one hue, light -> dark.
SEQUENTIAL_LIGHT = ["#eaf1fb", "#2a78d6"]
SEQUENTIAL_DARK = ["#16233a", "#3987e5"]

# Non-diagnosis series (metric comparisons). Assigned in order, never cycled.
SERIES_LIGHT = ["#2a78d6", "#1baf7a", "#4a3aa7", "#eb6834"]
SERIES_DARK = ["#3987e5", "#199e70", "#9085e9", "#d95926"]

LIGHT = {
    "surface": "#ffffff",
    "surface_alt": "#f6f8fa",
    "ink": "#17212b",
    "muted": "#5f6b76",
    "border": "#d9e2ea",
    "heading": "#16324f",
    "accent": "#168c8c",
    "grid": "#e9eef2",
}

DARK = {
    "surface": "#1a1a19",
    "surface_alt": "#232322",
    "ink": "#f2f3f4",
    "muted": "#a8b0b8",
    "border": "#34383c",
    "heading": "#cfe0f0",
    "accent": "#4ec9c9",
    "grid": "#2c3034",
}

# Legacy flat token map. Views and tests import these names directly, so the
# keys stay stable; the values track light mode.
COLORS = {
    "navy": LIGHT["heading"],
    "blue": CLASS_COLORS_LIGHT["CN"],
    "teal": LIGHT["accent"],
    "green": "#1baf7a",
    "amber": CLASS_COLORS_LIGHT["MCI"],
    "red": CLASS_COLORS_LIGHT["AD"],
    "ink": LIGHT["ink"],
    "muted": LIGHT["muted"],
    "border": LIGHT["border"],
    "surface": LIGHT["surface_alt"],
    "white": LIGHT["surface"],
    "cn": CLASS_COLORS_LIGHT["CN"],
    "mci": CLASS_COLORS_LIGHT["MCI"],
    "ad": CLASS_COLORS_LIGHT["AD"],
}

CLASS_COLORS = CLASS_COLORS_LIGHT

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
RADIUS = {"sm": 6, "md": 8, "lg": 12}

_MODE_KEY = "color_mode"


def get_mode() -> str:
    """Return the active colour mode ("light" or "dark").

    Streamlit's own theme is the source of truth, because it is the only thing
    that can colour canvas-rendered widgets such as ``st.dataframe``. This
    reads it rather than maintaining a second, desyncable switch.

    ``st.context.theme`` is read-only and lags a theme flip by exactly one
    rerun (Streamlit sends no rerun when the theme changes -- see the note in
    ``streamlit/runtime/context.py``). The stylesheet mirrors the dark tokens
    under ``prefers-color-scheme`` so the custom layer keeps up in the common
    "System" case.

    The explicit session override is checked first and exists for tests and
    headless screenshot capture, where no browser theme is negotiated.
    """
    override = st.session_state.get(_MODE_KEY)
    if override in ("light", "dark"):
        return override
    try:
        native = st.context.theme.type
    except Exception:  # no script run context (bare mode, import time)
        native = None
    return native if native in ("light", "dark") else "light"


def is_dark() -> bool:
    return get_mode() == "dark"


def palette() -> dict[str, str]:
    """Surface and ink tokens for the active mode."""
    return DARK if is_dark() else LIGHT


def class_colors() -> dict[str, str]:
    """Validated diagnosis colours for the active mode."""
    return CLASS_COLORS_DARK if is_dark() else CLASS_COLORS_LIGHT


def sequential_range() -> list[str]:
    """Two-stop single-hue ramp for magnitude encodings."""
    return SEQUENTIAL_DARK if is_dark() else SEQUENTIAL_LIGHT


def series_colors() -> list[str]:
    """Ordered categorical hues for non-diagnosis series."""
    return SERIES_DARK if is_dark() else SERIES_LIGHT


def appearance_hint(lang: str) -> None:
    """Point at Streamlit's native theme switch.

    A sidebar radio used to live here. It was removed because it could not do
    the job: Streamlit's theme options are non-scriptable, so no Python control
    can repaint the canvas-rendered dataframe. Keeping it meant two switches
    for one concept, which is exactly how the tables fell out of step with the
    rest of the page.
    """
    st.sidebar.caption(
        "外观：右上角菜单 → Light / Dark" if lang == "zh" else "Appearance: top-right menu → Light / Dark"
    )


def apply_theme() -> None:
    """Inject the stylesheet plus the custom properties for the active mode."""
    css_path = Path(__file__).with_name("styles.css")
    tokens = palette()
    classes = class_colors()
    overrides = "\n".join(
        [
            f"  --ad-surface-1: {tokens['surface']};",
            f"  --ad-surface-2: {tokens['surface_alt']};",
            f"  --ad-ink: {tokens['ink']};",
            f"  --ad-muted: {tokens['muted']};",
            f"  --ad-border: {tokens['border']};",
            f"  --ad-heading: {tokens['heading']};",
            f"  --ad-accent: {tokens['accent']};",
            f"  --ad-grid: {tokens['grid']};",
            f"  --ad-cn: {classes['CN']};",
            f"  --ad-mci: {classes['MCI']};",
            f"  --ad-ad: {classes['AD']};",
        ]
    )
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}\n:root {{\n{overrides}\n}}</style>",
        unsafe_allow_html=True,
    )
