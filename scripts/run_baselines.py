from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs, load_config
from src.explainability import write_basic_feature_importance
from src.reporting import analyze_error_cases
from src.training import run_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = config["project"].get("output_dir", "outputs")
    ensure_output_dirs(output_dir)
    results = run_baselines(config, quick=args.quick)
    model_path = Path(output_dir) / "models" / "best_model.joblib"
    if model_path.exists():
        import joblib

        model = joblib.load(model_path)
        write_basic_feature_importance(model, output_dir)
    analyze_error_cases(Path(output_dir) / "reports" / "best_model_predictions.csv", output_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
