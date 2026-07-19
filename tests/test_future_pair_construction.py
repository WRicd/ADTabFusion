import pandas as pd

from src.tasks.future_diagnosis import construct_future_diagnosis_pairs


def test_future_pairs_respect_date_and_horizon():
    frame = pd.DataFrame(
        {
            "RID": [1, 1, 1],
            "EXAMDATE": ["2020-01-01", "2021-01-01", "2027-01-01"],
            "DX": ["CN", "MCI", "AD"],
            "x": [10, 20, 30],
        }
    )
    pairs = construct_future_diagnosis_pairs(frame, ["x"], min_horizon_months=6, max_horizon_months=60)
    assert len(pairs) == 1
    assert pairs.iloc[0]["x"] == 10
    assert pairs.iloc[0]["target_diagnosis"] == "MCI"
