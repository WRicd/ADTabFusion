"""Guardrail: Streamlit's own theme must stay the single source of truth.

st.dataframe is glide-data-grid -- its cells are painted onto a <canvas> from
Streamlit's internal theme object, so injected CSS can never colour them. That
theme comes from .streamlit/config.toml and nowhere else (theme options are
non-scriptable: st.set_option("theme.primaryColor", ...) raises "cannot be set
on the fly"). If this file and dashboard/theme.py drift apart, the tables fall
out of step with the rest of the page -- the originally reported bug.
"""

import re
import tomllib
from pathlib import Path

from dashboard.theme import CLASS_COLORS_DARK, DARK, LIGHT

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".streamlit" / "config.toml"
CSS = ROOT / "dashboard" / "styles.css"


def _config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_defines_both_modes() -> None:
    """A flat [theme] alone collapses the switcher: it hides itself below two themes."""
    theme = _config()["theme"]
    assert "light" in theme, "[theme.light] missing -- native switcher would disappear"
    assert "dark" in theme, "[theme.dark] missing -- native switcher would disappear"
    assert theme["base"] == "light", "light must remain the reference presentation"


def test_streamlit_theme_matches_the_python_tokens() -> None:
    """The canvas and the .ad-* layer must be painted the same colours."""
    theme = _config()["theme"]
    for mode, tokens in (("light", LIGHT), ("dark", DARK)):
        block = theme[mode]
        assert block["backgroundColor"] == tokens["surface"], mode
        assert block["secondaryBackgroundColor"] == tokens["surface_alt"], mode
        assert block["textColor"] == tokens["ink"], mode
        assert block["borderColor"] == tokens["border"], mode
        assert block["primaryColor"] == tokens["accent"], mode


def test_dataframe_colours_are_themed_per_mode() -> None:
    """These specific keys are what the canvas grid reads for header and border."""
    theme = _config()["theme"]
    for mode, tokens in (("light", LIGHT), ("dark", DARK)):
        assert theme[mode]["dataframeHeaderBackgroundColor"] == tokens["surface_alt"], mode
        assert theme[mode]["dataframeBorderColor"] == tokens["border"], mode


def test_toolbar_stays_reachable_for_the_theme_switch() -> None:
    """toolbarMode must not hide the toolbar that hosts the theme switcher."""
    assert _config()["client"]["toolbarMode"] == "viewer"


def test_stylesheet_does_not_hide_the_header() -> None:
    """display:none on stHeader deletes the only runtime theme switch."""
    css = CSS.read_text(encoding="utf-8")
    header_rule = re.search(r'\[data-testid="stHeader"\]\s*\{([^}]*)\}', css, re.MULTILINE)
    assert header_rule, 'no [data-testid="stHeader"] rule found'
    assert "display: none" not in header_rule.group(1).replace("display:none", "display: none")


def _css_without_comments() -> str:
    """Strip /* ... */ so prose explaining the banned selector is not scanned."""
    return re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.DOTALL)


def test_stylesheet_never_overrides_the_icon_font() -> None:
    """`[class*="st-"]` matches Streamlit's emotion classes, including the icon
    span whose glyph is a font ligature. Overriding its font-family makes icons
    render as their literal names."""
    for selector, body in re.findall(r"([^{}]+)\{([^}]*)\}", _css_without_comments()):
        if "font-family" in body and 'class*="st-"' in selector:
            raise AssertionError(
                f"selector {selector.strip()!r} sets font-family and matches every "
                "Streamlit element, which breaks Material icon ligatures"
            )


def test_stylesheet_restores_the_icon_font_explicitly() -> None:
    """A belt-and-braces guard so a future override cannot silently win again."""
    css = _css_without_comments()
    assert '[data-testid="stIconMaterial"]' in css
    assert "Material Symbols Rounded" in css


def test_dark_media_query_mirrors_the_python_dark_tokens() -> None:
    """The prefers-color-scheme block covers the one-rerun lag; it must not drift."""
    css = CSS.read_text(encoding="utf-8")
    block = re.search(r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*:root\s*\{([^}]*)\}", css)
    assert block, "dark media query missing"
    declared = dict(re.findall(r"--ad-([\w-]+)\s*:\s*([^;]+);", block.group(1)))
    assert declared["surface-1"].strip() == DARK["surface"]
    assert declared["ink"].strip() == DARK["ink"]
    assert declared["cn"].strip() == CLASS_COLORS_DARK["CN"]
    assert declared["mci"].strip() == CLASS_COLORS_DARK["MCI"]
