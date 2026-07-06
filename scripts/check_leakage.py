from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_output_dirs, load_config
from src.data_loader import load_tadpole_csv
from src.leakage import write_leakage_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = config["project"].get("output_dir", "outputs")
    ensure_output_dirs(output_dir)
    df = load_tadpole_csv(config["data"]["raw_csv"])
    blacklist = write_leakage_report(config, df, output_dir)
    print(f"Wrote leakage report with {len(blacklist)} excluded columns.")


if __name__ == "__main__":
    main()

