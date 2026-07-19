from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.tadpole.phase_b_experiments import run_compact_vs_full


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare compact and full TADPOLE fairly.")
    parser.add_argument("--compact-config", required=True)
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_compact_vs_full(
        load_config(args.compact_config),
        load_config(args.full_config),
        quick=args.quick,
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

