from src.config import load_config
from src.tadpole.phase_b import load_catalog_groups, load_primary_whitelist


def test_primary_modality_groups_are_disjoint():
    features = load_primary_whitelist("outputs/phase_a/primary_whitelist.json")
    groups = load_catalog_groups("outputs/phase_a/feature_catalog.csv", features)
    seen = set()
    for columns in groups.values():
        assert seen.isdisjoint(columns)
        seen.update(columns)
    assert seen == set(features)


def test_ablation_config_does_not_reference_sparse_modalities():
    config = load_config("configs/tadpole_full_modality_ablation.yaml")
    modalities = {
        modality
        for combination in config["ablation"]["groups"].values()
        for modality in combination
    }
    assert {"mri_dti", "csf", "pet_other"}.isdisjoint(modalities)

