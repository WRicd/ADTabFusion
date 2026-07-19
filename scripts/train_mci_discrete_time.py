from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.phase_d.discrete_time import train_discrete_time_model
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/phase_d_discrete_time_risk.yaml");a=p.parse_args();r=train_discrete_time_model(load_config(a.config));print(f"Discrete-time {r['manifest']['model_name']} PR-AUC={r['metrics']['pr_auc']:.3f}")
if __name__=="__main__":main()
