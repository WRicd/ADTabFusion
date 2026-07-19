from __future__ import annotations

import pytest

from src.tadpole.modality_mapper import infer_modality


@pytest.mark.parametrize(
    ("column", "metadata", "expected"),
    [
        ("ABETA_UPENNBIOMK9_04_19_17", {"TBLNAME": "UPENNBIOMK9"}, "csf"),
        ("TAU_UPENNBIOMK9_04_19_17", {"TBLNAME": "UPENNBIOMK9"}, "csf"),
        ("AV45", {"TBLNAME": "ADNIMERGE"}, "pet_amyloid"),
        ("ROI_SUVR_UCBERKELEYAV1451", {"TBLNAME": "UCBERKELEYAV1451"}, "pet_other"),
        ("FA_CST_L_DTIROI", {"TBLNAME": "DTIROI", "TEXT": "Fractional anisotropy"}, "mri_dti"),
        ("RID", {"TBLNAME": "ADNIMERGE"}, "identifier"),
        ("DX", {"TBLNAME": "ADNIMERGE"}, "label"),
        ("Years_bl", {"TBLNAME": "ADNIMERGE"}, "visit_time"),
        ("Month_bl", {"TBLNAME": "ADNIMERGE"}, "visit_time"),
    ],
)
def test_dictionary_backed_modality_mapping(column, metadata, expected):
    assert infer_modality(column, metadata) == expected
