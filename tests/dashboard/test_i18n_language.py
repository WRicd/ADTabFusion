"""The language layer is only reachable through a Streamlit run context, so
these tests execute it inside AppTest rather than calling it directly."""

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

from dashboard.i18n import LANGUAGES, bilingual


def _script() -> None:
    import streamlit as st

    from dashboard.i18n import bilingual, get_language, language_selector

    selected = language_selector()
    st.text(f"selector:{selected}")
    st.text(f"state:{st.session_state['language']}")
    st.text(f"get_language:{get_language()}")
    st.text(f"bilingual:{bilingual('中文标题', 'English title')}")


def test_language_map_exposes_exactly_chinese_and_english():
    assert LANGUAGES == {"中文": "zh", "English": "en"}


def test_selector_defaults_to_chinese_and_seeds_session_state():
    app = AppTest.from_function(_script, default_timeout=60).run()

    assert not app.exception
    assert app.sidebar.selectbox[0].index == 0
    assert [element.value for element in app.text] == [
        "selector:zh",
        "state:zh",
        "get_language:zh",
        "bilingual:中文标题",
    ]


def test_selector_honours_a_preexisting_english_session_state():
    app = AppTest.from_function(_script, default_timeout=60)
    app.session_state["language"] = "en"
    app.run()

    assert not app.exception
    assert app.sidebar.selectbox[0].index == 1
    assert [element.value for element in app.text] == [
        "selector:en",
        "state:en",
        "get_language:en",
        "bilingual:English title",
    ]


def test_selecting_english_switches_the_stored_language():
    app = AppTest.from_function(_script, default_timeout=60).run()

    app.sidebar.selectbox[0].select("English").run()

    assert not app.exception
    assert app.session_state["language"] == "en"
    assert [element.value for element in app.text] == [
        "selector:en",
        "state:en",
        "get_language:en",
        "bilingual:English title",
    ]


@pytest.mark.parametrize(("lang", "expected"), [("zh", "中文"), ("en", "English")])
def test_bilingual_uses_an_explicit_language_without_session_state(lang: str, expected: str):
    assert bilingual("中文", "English", lang) == expected
