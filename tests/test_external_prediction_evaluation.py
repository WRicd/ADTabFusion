import numpy as np
import pandas as pd

from src.external.external_evaluation import evaluate_external_predictions, first_follow_up

PROBABILITY_COLS = ["prob_CN", "prob_MCI", "prob_AD"]


def _frame(n_per_class=10):
    classes = ["CN", "MCI", "AD"]
    rows = []
    for index in range(n_per_class * 3):
        label = index % 3
        rows.append(
            {
                "RID": index,
                "D4_SCANDATE": f"2016-0{1 + label}-01",
                "forecast_months": 6 + label * 12,
                "D4_label": label,
                "predicted_class": classes[label],
                "prob_CN": [0.8, 0.1, 0.1][label],
                "prob_MCI": [0.1, 0.8, 0.1][label],
                "prob_AD": [0.1, 0.1, 0.8][label],
            }
        )
    return pd.DataFrame(rows)


def test_evaluation_reports_row_level_first_follow_up_and_horizon_scopes():
    result = evaluate_external_predictions(_frame(), "predicted_class", PROBABILITY_COLS, "phase_c_direct_transfer")

    assert result["model_id"].unique().tolist() == ["phase_c_direct_transfer"]
    assert result.loc[result["scope"] == "row_level", "n_rows"].item() == 30
    assert result.loc[result["scope"] == "first_follow_up", "n_rows"].item() == 30
    assert result.loc[result["scope"] == "horizon", "horizon"].tolist() == [
        "0-12 months",
        "12-24 months",
        "24-36 months",
        ">36 months",
    ]
    row_level = result[result["scope"] == "row_level"].iloc[0]
    assert row_level["macro_f1"] == 1.0
    assert row_level["n_subjects"] == 30


def test_small_horizon_strata_are_flagged_unstable_and_carry_no_metrics():
    frame = _frame(n_per_class=2)

    result = evaluate_external_predictions(
        frame, "predicted_class", PROBABILITY_COLS, "phase_c_direct_transfer", minimum_stable_rows=20
    )

    horizons = result[result["scope"] == "horizon"]
    assert not horizons["stable_metrics_reported"].any()
    assert "macro_f1" not in horizons.columns or horizons["macro_f1"].isna().all()
    assert result[result["scope"] != "horizon"]["stable_metrics_reported"].all()


def test_empty_horizon_strata_report_zero_rows():
    frame = _frame(n_per_class=8)
    frame["forecast_months"] = 6

    result = evaluate_external_predictions(
        frame, "predicted_class", PROBABILITY_COLS, "phase_c_direct_transfer", minimum_stable_rows=1
    )

    horizons = result[result["scope"] == "horizon"].set_index("horizon")
    assert horizons.loc["0-12 months", "n_rows"] == 24
    assert horizons.loc[["12-24 months", "24-36 months", ">36 months"], "n_rows"].tolist() == [0, 0, 0]
    assert horizons.loc[["12-24 months", "24-36 months", ">36 months"], "macro_f1"].isna().all()


def test_first_follow_up_keeps_the_earliest_scan_per_subject():
    frame = pd.DataFrame(
        {
            "RID": [1, 1, 2],
            "D4_SCANDATE": ["2017-05-01", "2016-01-01", np.nan],
            "visit": ["late", "early", "undated"],
        }
    )

    selected = first_follow_up(frame)

    assert selected["visit"].tolist() == ["early", "undated"]
    assert selected["RID"].tolist() == [1, 2]


def test_first_follow_up_scope_deduplicates_repeated_subjects():
    frame = _frame(n_per_class=8)
    frame["RID"] = frame["RID"] % 8

    result = evaluate_external_predictions(
        frame, "predicted_class", PROBABILITY_COLS, "phase_c_direct_transfer", minimum_stable_rows=1
    )

    assert result.loc[result["scope"] == "row_level", "n_rows"].item() == 24
    assert result.loc[result["scope"] == "first_follow_up", "n_rows"].item() == 8
