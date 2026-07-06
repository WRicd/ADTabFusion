from dashboard.dashboard_utils import read_csv, read_json


def test_dashboard_readers_missing_files(tmp_path):
    assert read_json(tmp_path / "missing.json") == {}
    assert read_csv(tmp_path / "missing.csv").empty
