from __future__ import annotations

import pandas as pd
import pytest

from src.adni_inventory import inspect_csv, scan_raw_directory


@pytest.mark.parametrize(
    ("columns", "expected_category"),
    [
        (
            {"RID": [1], "VISCODE": ["bl"], "EXAMDATE": ["2020-01-01"], "DX": ["CN"]},
            "diagnosis",
        ),
        (
            {
                "RID": [1],
                "PTGENDER": ["Female"],
                "PTEDUCAT": [16],
                "PTETHCAT": ["Not Hisp/Latino"],
            },
            "demographics",
        ),
        ({"RID": [1], "APGEN1": [3], "APGEN2": [4]}, "apoe"),
        ({"RID": [1], "VISCODE": ["bl"], "MMSCORE": [29]}, "cognitive"),
        ({"RID": [1], "VISCODE": ["bl"], "ST101SV": [3200.0]}, "mri_measurement"),
        (
            {"RID": [1], "VISCODE": ["bl"], "LEFTHIPPO": [3100.0], "RIGHTHIPPO": [3200.0]},
            "mri_measurement",
        ),
        ({"RID": [1], "VISCODE": ["bl"], "SUMMARY_SUVR": [1.2]}, "pet_measurement"),
        ({"RID": [1], "VISCODE": ["bl"], "ABETA42": [900.0], "PTAU": [22.0]}, "biofluid_csf"),
        (
            {
                "RID": [1],
                "VISCODE": ["bl"],
                "ABETA_UPENNBIOMK9_04_19_17": [900.0],
                "PTAU_UPENNBIOMK9_04_19_17": [22.0],
            },
            "biofluid_csf",
        ),
    ],
)
def test_classifies_tables_from_columns(tmp_path, columns, expected_category):
    path = tmp_path / "UNINFORMATIVE.CSV"
    pd.DataFrame(columns).to_csv(path, index=False)

    record = inspect_csv(path)

    assert record["read_status"] == "ok"
    assert record["primary_category"] == expected_category


def test_scans_upper_and_lowercase_csv_extensions(tmp_path):
    pd.DataFrame({"RID": [1], "APOE4": [1]}).to_csv(tmp_path / "A.CSV", index=False)
    pd.DataFrame({"RID": [2], "MMSCORE": [28]}).to_csv(tmp_path / "b.csv", index=False)

    inventory = scan_raw_directory(tmp_path)

    assert len(inventory) == 2


def test_cognitive_item_columns_are_not_misclassified_as_mri(tmp_path):
    path = tmp_path / "ITEM.CSV"
    pd.DataFrame(
        {
            "RID": [1],
            "VISCODE": ["bl"],
            "ADAS_Q8_W17_volume": [1],
            "ADAS_Q1_word": [0],
        }
    ).to_csv(path, index=False)

    record = inspect_csv(path)

    assert record["primary_category"] == "cognitive"


def test_unreadable_csv_does_not_stop_scan(tmp_path):
    (tmp_path / "EMPTY.CSV").write_bytes(b"")
    pd.DataFrame({"RID": [1], "APOE4": [1]}).to_csv(tmp_path / "GOOD.csv", index=False)

    inventory = scan_raw_directory(tmp_path)

    statuses = {record["read_status"] for record in inventory}
    assert statuses == {"failed", "ok"}
    assert any(record["primary_category"] == "apoe" for record in inventory)
