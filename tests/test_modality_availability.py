from __future__ import annotations

from src.adni_inventory import build_modality_availability


def _record(category: str, feature_key: str, columns: list[str]) -> dict:
    return {
        "file": f"data/raw/{category}.CSV",
        "read_status": "ok",
        "matched_categories": [category],
        "diagnosis_columns": columns if category == "diagnosis" else [],
        "matched_features": {feature_key: columns},
    }


def test_missing_pet_is_reported_without_affecting_other_modalities():
    inventory = [
        _record("mri_measurement", "mri", ["ST101SV"]),
        _record("biofluid_csf", "csf", ["ABETA42", "PTAU"]),
        _record("cognitive", "cognitive", ["MMSCORE"]),
    ]

    availability = build_modality_availability(inventory)

    assert availability["mri_measurement"]["available"] is True
    assert availability["biofluid_csf"]["available"] is True
    assert availability["cognitive"]["available"] is True
    assert availability["pet_measurement"] == {
        "available": False,
        "files": [],
        "matched_columns": [],
    }
