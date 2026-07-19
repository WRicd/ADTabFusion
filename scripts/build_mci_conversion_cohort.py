from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.phase_d.mci_conversion import build_all_landmarks

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/phase_d_mci_conversion.yaml"); a=p.parse_args(); c=load_config(a.config)
    cohorts,summary,_=build_all_landmarks(c); out=Path(c["project"]["output_dir"])/"cohorts"; out.mkdir(parents=True,exist_ok=True); summary.to_csv(out/"mci_landmark_summary.csv",index=False)
    print(summary[["horizon_months","eligible_subjects","converters","non_converters","censored_subjects"]].to_string(index=False))
if __name__=="__main__": main()
