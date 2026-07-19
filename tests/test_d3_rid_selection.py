import pandas as pd

from src.external.d3_cohort import select_d3_index_records


def test_latest_d3_record_is_selected_without_future_data():
    frame = pd.DataFrame(
        {"RID": [1, 1, 2], "EXAMDATE": ["2015-01-01", "2016-01-01", "bad"], "x": [1, 2, 3]}
    )
    selected, audit = select_d3_index_records(frame)
    assert selected.set_index("RID").loc[1, "x"] == 2
    assert len(selected) == 2
    assert audit.groupby("RID")["selected"].sum().eq(1).all()
