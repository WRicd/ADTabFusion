from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_d.verification import verify_phase_c_artifacts, write_verification_outputs


def main() -> None:
    result = verify_phase_c_artifacts()
    write_verification_outputs(result, "outputs/phase_d/audit")
    print(f"Phase C verification gate: {'PASS' if result['passed'] else 'FAIL'}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
