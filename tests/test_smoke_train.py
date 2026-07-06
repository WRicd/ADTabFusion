import pandas as pd

from src.training import run_single_experiment


def test_smoke_train_pipeline(tmp_path):
    df = pd.DataFrame(
        {
            "RID": list(range(20)),
            "AGE": [70 + i % 5 for i in range(20)],
            "PTGENDER": ["M", "F"] * 10,
            "MMSE": [20 + i % 8 for i in range(20)],
            "label": [0, 1] * 10,
        }
    )
    cfg = {
        "project": {"output_dir": str(tmp_path)},
        "data": {"subject_col": "RID"},
        "split": {"test_size": 0.2, "val_size": 0.1},
        "preprocessing": {"numeric_impute": "median"},
        "models": {"logistic_regression": {"max_iter": 200}},
    }
    metrics, _, _ = run_single_experiment(
        df, "label", ["AGE", "PTGENDER", "MMSE"], "logistic_regression", cfg, 42
    )
    assert "macro_f1" in metrics

