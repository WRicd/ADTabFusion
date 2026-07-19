from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.phase_d.calibration import calibrate_phase_d
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/phase_d_calibration.yaml");a=p.parse_args();r=calibrate_phase_d(load_config(a.config));print(f"Selected calibration: {r['manifest']['calibration_method']}")
if __name__=="__main__":main()
