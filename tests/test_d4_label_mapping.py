import pytest

from src.external.d4_matching import normalize_diagnosis_label


@pytest.mark.parametrize("value,expected", [("CN", "CN"), ("NL", "CN"), ("Normal", "CN"), ("LMCI", "MCI"), ("EMCI", "MCI"), ("Dementia", "AD")])
def test_d4_label_mapping(value, expected):
    assert normalize_diagnosis_label(value) == expected


def test_ambiguous_d4_label_is_unresolved():
    assert normalize_diagnosis_label("possible AD") is None
