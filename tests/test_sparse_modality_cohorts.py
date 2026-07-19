from src.config import load_config


def test_sparse_modalities_have_separate_configs_and_reports():
    expected = {
        "configs/tadpole_dti_cohort.yaml": "mri_dti",
        "configs/tadpole_csf_cohort.yaml": "csf",
        "configs/tadpole_tau_pet_cohort.yaml": "pet_other",
    }
    for path, modality in expected.items():
        config = load_config(path)
        assert config["sparse_cohort"]["modality"] == modality
        assert "sparse_modalities" in config["paths"]["summary_report"]

