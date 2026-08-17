"""Guardrail: an early-stopping set may inform *when* to stop, never *what* to score.

Passing the held-out fold into a fold's own fit would let the model stop at the
point that flatters that fold, turning cross-validation into a selection metric.
"""

import inspect

from src.phase_d import transition_model


def test_cross_validation_folds_get_no_early_stopping_frame():
    source = inspect.getsource(transition_model._run_group_kfold)
    assert "early_stopping_frame" not in source


def test_early_stopping_frame_is_optional_and_defaults_to_none():
    parameter = inspect.signature(transition_model._fit_transition_pipeline).parameters["early_stopping_frame"]
    assert parameter.default is None


def test_ablation_loop_uses_the_validation_split_for_early_stopping():
    source = inspect.getsource(transition_model.train_transition_aware)
    assert "early_stopping_frame=validation" in source
    assert "early_stopping_frame=temporal_test" not in source
