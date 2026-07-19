from pathlib import Path
def test_d4_replay_report_is_explicitly_exploratory():
 text=Path("outputs/phase_d/reports/d4_replay_report.md").read_text(encoding="utf-8").lower();assert "exploratory post-hoc" in text and "not independent confirmatory" in text
