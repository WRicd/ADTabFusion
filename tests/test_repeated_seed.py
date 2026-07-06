import pandas as pd

from src.training import summarize_seed_results


def test_repeated_seed_summary_columns():
    df = pd.DataFrame(
        {
            "model": ["a", "a"],
            "task_mode": ["all_visits", "all_visits"],
            "accuracy": [0.5, 0.7],
            "balanced_accuracy": [0.5, 0.7],
            "macro_f1": [0.4, 0.6],
            "roc_auc_ovr": [0.8, 0.9],
        }
    )
    summary = summarize_seed_results(df, ["model", "task_mode"])
    assert summary.loc[0, "accuracy_mean"] == 0.6
    assert "macro_f1_std" in summary.columns

