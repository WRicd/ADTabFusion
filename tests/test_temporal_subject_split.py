import pandas as pd
from src.phase_d.temporal_split import build_temporal_subject_split


def test_temporal_split_is_subject_disjoint_and_date_ordered():
    frame = pd.DataFrame({"RID": list(range(20)), "EXAMDATE": pd.date_range("2020-01-01", periods=20)})
    split, assignment = build_temporal_subject_split(frame)
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["temporal_test"])
    assert assignment.loc[assignment.split == "train", "index_date"].max() <= assignment.loc[assignment.split == "validation", "index_date"].min()
