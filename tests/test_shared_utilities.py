import hashlib
import json

import pandas as pd

from src.artifact_io import TADPOLE_NA_VALUES, read_json, read_tadpole_table, write_json
from src.evaluation import csv_safe_metrics
from src.provenance import sha256_file, stable_subject_hash


def test_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "artifact.json"
    payload = {"model": "random_forest", "features": ["AGE", "MMSE"]}
    assert write_json(path, payload) == path
    assert path.read_text(encoding="utf-8") == json.dumps(payload, indent=2)
    assert read_json(path) == payload


def test_read_tadpole_table_treats_placeholders_as_missing(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("RID,DX,MMSE,EXTRA\n1,MCI,-4,keep\n2,AD, ,keep\n", encoding="utf-8")
    frame = read_tadpole_table(path, ["RID", "DX", "MMSE"])
    assert list(frame.columns) == ["RID", "DX", "MMSE"]
    assert frame["MMSE"].isna().all()
    assert TADPOLE_NA_VALUES == ["", " ", "-4", "-4.0"]


def test_read_tadpole_table_without_columns_keeps_every_field(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("RID,MMSE\n1,-4.0\n", encoding="utf-8")
    frame = read_tadpole_table(path)
    assert list(frame.columns) == ["RID", "MMSE"]
    assert frame["MMSE"].isna().all()


def test_csv_safe_metrics_serializes_nested_values():
    row = csv_safe_metrics({"macro_f1": 0.5, "confusion_matrix": [[1, 0], [0, 1]]})
    assert row["macro_f1"] == 0.5
    assert row["confusion_matrix"] == "[[1, 0], [0, 1]]"


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"frozen artifact")
    assert sha256_file(path) == hashlib.sha256(b"frozen artifact").hexdigest()


def test_stable_subject_hash_ignores_row_order():
    first = stable_subject_hash(pd.Series(["2", "1", None]))
    second = stable_subject_hash(pd.Series(["1", "2"]))
    assert first == second
