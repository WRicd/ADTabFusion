import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import audit_dataframe, load_tadpole_csv


def _frame():
    return pd.DataFrame(
        {
            "RID": [1, 1, 2, 3],
            "VISCODE": ["bl", "bl", "bl", "m12"],
            "DX": ["CN", "MCI", None, "AD"],
            "AGE": [70.0, 71.0, 72.0, 73.0],
            "SPARSE": [np.nan, np.nan, np.nan, 1.0],
        }
    )


def test_loader_strips_column_whitespace(tmp_path):
    path = tmp_path / "tadpole.csv"
    path.write_text(" RID , DX \n1,CN\n", encoding="utf-8")

    frame = load_tadpole_csv(path)

    assert frame.columns.tolist() == ["RID", "DX"]


def test_loader_explains_the_data_access_requirement_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="data access agreement"):
        load_tadpole_csv(tmp_path / "absent.csv")


def test_audit_summarizes_shape_labels_and_duplicates(tmp_path):
    audit = audit_dataframe(_frame(), tmp_path, high_missing_threshold=0.6)

    assert audit["n_rows"] == 4
    assert audit["n_columns"] == 5
    assert audit["n_subjects"] == 3
    assert audit["label_distribution"] == {"CN": 1, "MCI": 1, "<missing>": 1, "AD": 1}
    assert audit["numeric_columns"] == 3
    assert audit["categorical_columns"] == 2
    assert audit["high_missing_columns"] == ["SPARSE"]
    assert audit["missing_rate_by_column"]["SPARSE"] == pytest.approx(0.75)
    assert audit["duplicate_subject_visit_rows"] == 1


def test_audit_writes_json_and_diagnostic_figures(tmp_path):
    audit = audit_dataframe(_frame(), tmp_path)

    written = json.loads((tmp_path / "reports" / "data_audit.json").read_text(encoding="utf-8"))
    assert written == audit
    assert (tmp_path / "figures" / "missing_rate_by_column.png").exists()
    assert (tmp_path / "figures" / "label_distribution.png").exists()


def test_audit_tolerates_tables_without_the_expected_key_columns(tmp_path):
    audit = audit_dataframe(pd.DataFrame({"AGE": [70.0, 71.0]}), tmp_path)

    assert audit["n_subjects"] is None
    assert audit["label_distribution"] == {}
    assert audit["duplicate_subject_visit_rows"] == 0
    assert not (tmp_path / "figures" / "label_distribution.png").exists()
