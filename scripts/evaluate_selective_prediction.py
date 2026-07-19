from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.phase_d.selective_prediction import evaluate_selective_prediction
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/phase_d_selective_prediction.yaml");a=p.parse_args();r=evaluate_selective_prediction(load_config(a.config));print(f"Frozen abstention threshold: {r['manifest']['threshold']:.4f}")
if __name__=="__main__":main()
