from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_VIEWS = {
    "executive_summary.py",
    "data_cohort.py",
    "scientific_guardrails.py",
    "transition_aware.py",
    "mci_progression.py",
    "calibration_uncertainty.py",
    "external_replay.py",
    "reproducibility.py",
}


def test_eight_required_views_exist_and_are_registered() -> None:
    view_dir = ROOT / "dashboard" / "views"
    assert REQUIRED_VIEWS <= {path.name for path in view_dir.glob("*.py")}
    app_source = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    for view in REQUIRED_VIEWS:
        assert f"views/{view}" in app_source
    assert "st.navigation" in app_source

