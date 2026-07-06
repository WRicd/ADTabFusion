import pandas as pd

from src.leakage import build_feature_blacklist, remove_blacklisted_features


def test_baseline_leakage_blacklist():
    cfg = {"data": {"task_mode": "baseline_only"}}
    df = pd.DataFrame(columns=["DX", "DXCHANGE", "DX_bl", "VISCODE", "AGE"])
    blacklist = build_feature_blacklist(cfg, df)
    assert {"DX", "DXCHANGE", "DX_bl", "VISCODE"}.issubset(blacklist)
    assert remove_blacklisted_features(["AGE", "DX", "VISCODE"], blacklist) == ["AGE"]


def test_all_visit_leakage_blacklist():
    cfg = {"data": {"task_mode": "all_visits"}}
    df = pd.DataFrame(columns=["DX", "DXCHANGE", "DX_bl", "VISCODE", "AGE"])
    blacklist = build_feature_blacklist(cfg, df)
    assert {"DX", "DXCHANGE", "DX_bl"}.issubset(blacklist)

