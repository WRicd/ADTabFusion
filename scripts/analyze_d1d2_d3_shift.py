from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.distribution_shift import analyze_d1d2_d3_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze D1/D2 to D3 distribution shift without D4 labels.")
    parser.add_argument("--config", default="configs/phase_c_direct_transfer.yaml")
    args = parser.parse_args()
    result = analyze_d1d2_d3_shift(load_config(args.config))
    print(f"Wrote {len(result)} dataset-shift rows.")


if __name__ == "__main__":
    main()
