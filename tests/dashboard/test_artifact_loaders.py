import json

import pandas as pd

from dashboard.artifacts import load_csv, load_json


def test_csv_loader_returns_frame_without_mutating_source(tmp_path) -> None:
    path = tmp_path / "artifact.csv"
    pd.DataFrame({"metric": [0.5]}).to_csv(path, index=False)
    frame, error = load_csv(path, ("metric",))
    assert error is None
    assert frame.to_dict("records") == [{"metric": 0.5}]
    assert path.read_text(encoding="utf-8").startswith("metric")


def test_json_loader_and_missing_artifact_are_safe(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    payload, error = load_json(path)
    assert error is None
    assert payload == {"frozen": True}

    missing, missing_error = load_json(tmp_path / "missing.json")
    assert missing == {}
    assert "Missing artifact" in str(missing_error)


def test_required_columns_are_validated(tmp_path) -> None:
    path = tmp_path / "artifact.csv"
    pd.DataFrame({"other": [1]}).to_csv(path, index=False)
    frame, error = load_csv(path, ("metric",))
    assert frame.empty
    assert "missing columns" in str(error)

