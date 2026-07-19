import json

from dashboard.dashboard_utils import raw_inventory_summary, read_csv, read_json


def test_dashboard_readers_missing_files(tmp_path):
    assert read_json(tmp_path / "missing.json") == {}
    assert read_csv(tmp_path / "missing.csv").empty


def test_raw_inventory_summary(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "adni_file_inventory.json").write_text(
        json.dumps([{"file": "A.CSV"}, {"file": "B.CSV"}]), encoding="utf-8"
    )
    (metrics / "adni_modality_availability.json").write_text(
        json.dumps(
            {
                "mri_measurement": {"available": True},
                "pet_measurement": {"available": False},
            }
        ),
        encoding="utf-8",
    )

    summary = raw_inventory_summary(tmp_path)

    assert summary["scanned_csvs"] == 2
    assert summary["available"] == ["mri_measurement"]
    assert summary["missing"] == ["pet_measurement"]
