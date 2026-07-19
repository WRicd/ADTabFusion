import pandas as pd

from src.tasks.future_diagnosis import construct_future_diagnosis_pairs


def test_future_measurements_never_enter_source_features():
    frame = pd.DataFrame(
        {"RID": [1, 1], "EXAMDATE": ["2020-01-01", "2021-01-01"], "DX": ["CN", "AD"], "x": [1, 99]}
    )
    pair = construct_future_diagnosis_pairs(frame, ["x"]).iloc[0]
    assert pair["x"] == 1
    assert pair["label"] == 2
