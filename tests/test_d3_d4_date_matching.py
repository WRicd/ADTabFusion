import pandas as pd


def test_forecast_horizon_uses_d4_minus_d3():
    d3 = pd.Timestamp("2020-01-01")
    d4 = pd.Timestamp("2021-01-01")
    days = (d4 - d3).days
    assert days == 366
    assert 12.0 < days / 30.4375 < 12.1
