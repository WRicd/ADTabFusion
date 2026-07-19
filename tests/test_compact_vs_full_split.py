import pandas as pd

from src.tadpole.phase_b import make_subject_partitions, select_baseline_records


def test_shared_subject_partition_can_be_applied_to_compact_and_full():
    subjects = list(range(60))
    full = pd.DataFrame({"RID": subjects, "label": [value % 3 for value in subjects]})
    compact = full.copy()
    partitions = make_subject_partitions(full, seed=42)
    for split, ids in partitions.items():
        assert set(full[full["RID"].isin(ids)]["RID"]) == set(
            compact[compact["RID"].isin(ids)]["RID"]
        )
    assert set(partitions["train"]).isdisjoint(partitions["test"])


def test_baseline_selection_prefers_bl_then_earliest_date():
    frame = pd.DataFrame(
        {
            "RID": [1, 1, 2, 2],
            "VISCODE": ["m06", "bl", "m12", "m06"],
            "EXAMDATE": ["2020-06-01", "2020-02-01", "2021-01-01", "2020-07-01"],
            "value": [1, 2, 3, 4],
        }
    )
    selected = select_baseline_records(frame).set_index("RID")
    assert selected.loc[1, "value"] == 2
    assert selected.loc[2, "value"] == 4

