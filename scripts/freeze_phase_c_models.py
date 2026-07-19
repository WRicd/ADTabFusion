from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.external.model_freezing import freeze_direct_transfer_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze preselected Phase C direct-transfer pipelines.")
    parser.add_argument("--config", default="configs/phase_c_direct_transfer.yaml")
    args = parser.parse_args()
    manifests = freeze_direct_transfer_models(load_config(args.config), args.config)
    for role, manifest in manifests.items():
        print(f"Frozen {role}: {manifest['candidate_id']} ({manifest['model_sha256'][:12]})")


if __name__ == "__main__":
    main()
