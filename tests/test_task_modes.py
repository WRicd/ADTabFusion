import pandas as pd

from src.data_schema import select_task_rows


def test_baseline_only_keeps_one_row_per_subject():
    df = pd.DataFrame(
        {
            "RID": [1, 1, 2, 2],
            "VISCODE": ["bl", "m06", "bl", "m06"],
            "EXAMDATE": ["2020-01-01", "2020-06-01", "2020-01-02", "2020-06-02"],
        }
    )
    selected = select_task_rows(df, "baseline_only")
    assert len(selected) == 2
    assert selected["RID"].nunique() == 2
    assert set(selected["VISCODE"]) == {"bl"}

