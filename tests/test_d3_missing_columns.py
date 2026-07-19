import pandas as pd

from src.external.schema_alignment import align_to_frozen_schema


def test_missing_d3_columns_are_padded_with_nan():
    aligned = align_to_frozen_schema(pd.DataFrame({"a": [1.0]}), ["a", "missing"])
    assert aligned["missing"].isna().all()
    assert aligned.columns.tolist() == ["a", "missing"]
