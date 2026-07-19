from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.phase_d.transition_model import train_transition_aware

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default="configs/phase_d_transition_model.yaml"); args=parser.parse_args()
    result=train_transition_aware(load_config(args.config), args.config)
    print(f"Selected {result['manifest']['model_name']}; locked temporal Macro F1={result['temporal_test']['macro_f1']:.3f}")
if __name__ == "__main__": main()
