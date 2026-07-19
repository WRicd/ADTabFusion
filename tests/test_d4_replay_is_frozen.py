import inspect
from src.phase_d.d4_replay import replay_phase_d_on_d4
def test_d4_replay_contains_no_fit_or_selection_operation():
 source=inspect.getsource(replay_phase_d_on_d4);assert ".fit(" not in source and "sort_values" not in source
