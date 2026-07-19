from src.tadpole.phase_b import FORBIDDEN_FEATURES, load_catalog_groups, load_primary_whitelist


def test_frozen_primary_whitelist_has_exactly_69_safe_features():
    features = load_primary_whitelist("outputs/phase_a/primary_whitelist.json")
    assert len(features) == 69
    assert FORBIDDEN_FEATURES.isdisjoint(features)


def test_sparse_modalities_are_not_in_primary_groups():
    features = load_primary_whitelist("outputs/phase_a/primary_whitelist.json")
    groups = load_catalog_groups("outputs/phase_a/feature_catalog.csv", features)
    assert {"mri_dti", "csf", "pet_other"}.isdisjoint(groups)

