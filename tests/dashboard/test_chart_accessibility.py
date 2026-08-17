"""Guardrail: diagnosis identity is never encoded by colour alone.

The validated palette clears every hard gate but carries one advisory per mode
(light: MCI amber at 2.11:1 contrast; dark: MCI/AD CVD dE 6.2). Both are only
permissible alongside a secondary encoding, so the relief is a requirement of
the palette rather than a nicety -- these tests keep it from being dropped.
"""

import ast
import inspect
from pathlib import Path

import dashboard.charts as charts

ROOT = Path(__file__).resolve().parents[2]
VIEW_DIR = ROOT / "dashboard" / "views"

# Views that plot diagnosis classes or multi-series metrics.
CHART_VIEWS = {
    "executive_summary.py",
    "transition_aware.py",
    "mci_progression.py",
    "calibration_uncertainty.py",
    "data_cohort.py",
    "external_replay.py",
}


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_class_color_scale_always_builds_a_legend() -> None:
    """The legend is the standing half of the secondary encoding."""
    source = inspect.getsource(charts.class_color_scale)
    assert "alt.Legend" in source
    assert "legend=None" not in source


def test_every_chart_view_ships_labels_or_a_table() -> None:
    for name in sorted(CHART_VIEWS):
        called = _called_names(VIEW_DIR / name)
        assert called & {
            "value_labels",
            "table_view",
            "mark_text",
        }, f"{name} renders charts with no direct labels and no table view"


def test_no_view_hardcodes_a_diagnosis_hex() -> None:
    """Colours must come from the mode-aware helpers, not literals."""
    banned = {"#2a78d6", "#eda100", "#e34948", "#3987e5", "#c98500", "#2F6B9A", "#C78321", "#B84A4A"}
    for path in VIEW_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for hex_value in banned:
            assert hex_value not in source, f"{path.name} hardcodes {hex_value}"


def test_series_color_refuses_to_cycle_hues() -> None:
    """A 5th series must fold into 'Other' or facet, never reuse a hue."""
    source = inspect.getsource(charts.series_color)
    assert "raise ValueError" in source


def test_charts_render_on_a_transparent_surface() -> None:
    """A pinned white canvas would fight the dark surface."""
    source = inspect.getsource(charts.style_chart)
    assert 'background="transparent"' in source
