from pathlib import Path

import pytest

from src.config import load_config


@pytest.mark.parametrize(
    "name",
    [
        "tadpole_full_primary_baseline.yaml",
        "tadpole_full_primary_all_visits.yaml",
        "tadpole_compact_comparison.yaml",
        "tadpole_full_modality_ablation.yaml",
        "tadpole_full_missing_modality.yaml",
    ],
)
def test_phase_b_configs_are_isolated(name):
    config = load_config(Path("configs") / name)
    assert config["project"]["phase"] == "phase_b"
    assert config["project"]["output_dir"] == "outputs/phase_b"


def test_full_primary_config_uses_frozen_phase_a_whitelist():
    config = load_config("configs/tadpole_full_primary_baseline.yaml")
    assert config["data"]["feature_whitelist"] == "outputs/phase_a/primary_whitelist.json"
    assert config["data"]["max_missing_rate"] == 0.70

