from __future__ import annotations

import streamlit as st


LANGUAGES = {"中文": "zh", "English": "en"}


def language_selector() -> str:
    current = st.session_state.get("language", "zh")
    labels = list(LANGUAGES)
    default_index = 0 if current == "zh" else 1
    selected = st.sidebar.selectbox("Language / 语言", labels, index=default_index, key="language_picker")
    lang = LANGUAGES[selected]
    st.session_state["language"] = lang
    return lang


def get_language() -> str:
    return st.session_state.get("language", "zh")


def bilingual(zh: str, en: str, lang: str | None = None) -> str:
    return zh if (lang or get_language()) == "zh" else en

