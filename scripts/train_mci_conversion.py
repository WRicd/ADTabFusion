from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.phase_d.mci_conversion import train_landmark_models
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/phase_d_mci_conversion.yaml");a=p.parse_args();r=train_landmark_models(load_config(a.config));print(r.to_string(index=False))
if __name__=="__main__":main()
