from __future__ import annotations

import pandas as pd

from src.tadpole.feature_catalog import build_feature_catalog
from src.tadpole.leakage_rules import build_leakage_artifacts


def test_labels_identifiers_and_dates_are_not_whitelisted(tmp_path):
    data_path = tmp_path / "data.csv"
    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(
        {
            "RID": [1, 2, 3],
            "VISCODE": ["bl", "bl", "bl"],
            "DX": ["CN", "MCI", "AD"],
            "MMSE": [29, 25, 18],
            "ABETA_UPENNBIOMK9_04_19_17": [1000.0, 800.0, 500.0],
        }
    ).to_csv(data_path, index=False)
    pd.DataFrame(
        {
            "FLDNAME": ["RID", "VISCODE", "DX", "MMSE", "ABETA_UPENNBIOMK9_04_19_17"],
            "TBLNAME": ["ADNIMERGE", "ADNIMERGE", "ADNIMERGE", "ADNIMERGE", "UPENNBIOMK9"],
            "TEXT": ["Roster ID", "Visit code", "Diagnosis", "MMSE", "CSF amyloid beta"],
            "TYPE": ["", "", "", "numeric", "numeric"],
            "UNITS": ["", "", "", "score", "pg/mL"],
        }
    ).to_csv(dictionary_path, index=False)

    catalog = build_feature_catalog(
        data_path,
        dictionary_path,
        max_missing_rate=1.0,
        min_non_missing_count=1,
        max_primary_features=10,
    )
    blacklist, whitelist = build_leakage_artifacts(catalog)

    assert {"RID", "VISCODE", "DX"}.issubset(blacklist)
    assert "RID" not in whitelist
    assert "DX" not in whitelist
    assert "MMSE" in whitelist
    assert "ABETA_UPENNBIOMK9_04_19_17" in whitelist

