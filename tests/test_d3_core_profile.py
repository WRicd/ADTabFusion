import json
from pathlib import Path


def test_d3_core_is_exact_whitelist_schema_intersection():
    whitelist = json.loads(Path("outputs/phase_a/primary_whitelist.json").read_text())
    profile = json.loads(Path("outputs/phase_d/audit/d3_core_features.json").read_text())
    import pandas as pd
    d3 = set(pd.read_csv("data/tadpole_challenge/TADPOLE_D3.csv", nrows=0).columns)
    assert profile == [feature for feature in whitelist if feature in d3]
