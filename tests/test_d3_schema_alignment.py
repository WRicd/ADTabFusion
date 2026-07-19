import pandas as pd

from src.external.schema_alignment import align_to_frozen_schema


def test_schema_alignment_preserves_frozen_order():
    frame = pd.DataFrame({"b": [2], "a": [1], "extra": [9]})
    aligned = align_to_frozen_schema(frame, ["a", "b"])
    assert aligned.columns.tolist() == ["a", "b"]
    assert aligned.iloc[0].tolist() == [1, 2]
