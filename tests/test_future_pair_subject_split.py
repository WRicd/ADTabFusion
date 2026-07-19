import pandas as pd

from src.tasks.future_diagnosis import build_split_pairs, split_subjects_before_pairing


def test_subjects_do_not_cross_pair_splits():
    rows = []
    for rid in range(30):
        rows.extend(
            [
                {"RID": rid, "EXAMDATE": "2020-01-01", "DX": "CN", "x": rid},
                {"RID": rid, "EXAMDATE": "2021-01-01", "DX": "MCI", "x": rid + 1},
            ]
        )
    frame = pd.DataFrame(rows)
    split = split_subjects_before_pairing(frame, seed=42)
    pairs = build_split_pairs(frame, ["x"], split)
    subject_sets = [set(value["RID"]) for value in pairs.values()]
    assert subject_sets[0].isdisjoint(subject_sets[1])
    assert subject_sets[0].isdisjoint(subject_sets[2])
    assert subject_sets[1].isdisjoint(subject_sets[2])
