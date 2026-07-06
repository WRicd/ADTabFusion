from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs, load_config
from src.explainability import run_explainability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config["project"].get("output_dir", "outputs"))
    feature_importance, modality_importance = run_explainability(config, quick=args.quick)
    print(feature_importance.head(20).to_string(index=False))
    print(modality_importance.to_string(index=False))


if __name__ == "__main__":
    main()

