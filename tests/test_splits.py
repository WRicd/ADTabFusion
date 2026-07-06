import pandas as pd

from src.splits import make_subject_split


def test_subject_split_has_no_leakage(tmp_path):
    df = pd.DataFrame(
        {
            "RID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "label": [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2],
        }
    )
    split = make_subject_split(df, "RID", "label", 0.2, 0.2, 42, tmp_path)
    sets = []
    for key in ["train_idx", "val_idx", "test_idx"]:
        sets.append(set(df.loc[split[key], "RID"]))
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])

