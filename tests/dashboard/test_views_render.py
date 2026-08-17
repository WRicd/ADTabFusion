"""Smoke test: every view renders, in both colour modes, without raising.

The other dashboard tests read source text. These actually execute each view
against the frozen artifacts, which is what catches a renamed column, a stale
variable after a refactor, or a token that only one mode defines.
"""

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

# executive_summary is exercised through app.py instead: it calls
# st.page_link, which needs the navigation context that only the entry point
# establishes, and raises 'url_pathname' when run standalone.
STANDALONE_VIEWS = [
    "data_cohort",
    "scientific_guardrails",
    "transition_aware",
    "mci_progression",
    "calibration_uncertainty",
    "external_replay",
    "reproducibility",
]


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("view", STANDALONE_VIEWS)
def test_view_renders_without_error(view: str, mode: str) -> None:
    app = AppTest.from_file(f"dashboard/views/{view}.py", default_timeout=120)
    app.session_state["color_mode"] = mode
    app.session_state["language"] = "en"
    app.run()

    assert not app.exception, f"{view} [{mode}]: {app.exception[0].value if app.exception else ''}"
    assert len(app.markdown) > 0, f"{view} [{mode}] rendered nothing"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_entry_point_renders_with_navigation(mode: str) -> None:
    app = AppTest.from_file("dashboard/app.py", default_timeout=120)
    app.session_state["color_mode"] = mode
    app.session_state["language"] = "en"
    app.run()

    assert not app.exception, f"app.py [{mode}]: {app.exception[0].value if app.exception else ''}"


def test_charts_carry_a_table_view_for_screen_readers() -> None:
    """The table view is the palette's relief path; it must actually render."""
    app = AppTest.from_file("dashboard/views/transition_aware.py", default_timeout=120)
    app.session_state["language"] = "en"
    app.run()

    assert not app.exception
    assert len(app.dataframe) >= 1
