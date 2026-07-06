from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs, load_config
from src.reporting import analyze_error_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    ensure_output_dirs(output_dir)
    errors = analyze_error_cases(output_dir / "reports" / "best_model_predictions.csv", output_dir)
    print(f"Wrote {len(errors)} error cases.")


if __name__ == "__main__":
    main()
