import pandas as pd
from src.phase_d.temporal_split import build_temporal_subject_split


def test_temporal_cutoffs_are_reproducible():
    frame = pd.DataFrame({"RID": [3, 1, 2, 4, 5, 6, 7], "EXAMDATE": ["2020-01-01"] * 7})
    first, _ = build_temporal_subject_split(frame)
    second, _ = build_temporal_subject_split(frame.sample(frac=1, random_state=9))
    assert first == second
