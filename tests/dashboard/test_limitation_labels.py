from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_D4_LABEL = "Exploratory post-hoc replay — not independent confirmatory validation"


def test_d4_page_always_contains_required_label() -> None:
    source = (ROOT / "dashboard" / "views" / "external_replay.py").read_text(encoding="utf-8")
    assert REQUIRED_D4_LABEL in source
    assert source.count("EXPLORATORY_LABEL") >= 3


def test_executive_page_calls_out_d4_as_exploratory() -> None:
    source = (ROOT / "dashboard" / "views" / "executive_summary.py").read_text(encoding="utf-8").lower()
    assert "exploratory post-hoc replay" in source
    assert "not independent confirmatory external validation" in source

