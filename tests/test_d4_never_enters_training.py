import inspect

from src.external.model_freezing import freeze_direct_transfer_models, load_baseline_training_cohort


def test_direct_training_interfaces_do_not_accept_d4():
    assert "d4" not in inspect.signature(load_baseline_training_cohort).parameters
    assert "d4" not in inspect.signature(freeze_direct_transfer_models).parameters
