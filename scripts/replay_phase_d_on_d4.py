from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.phase_d.d4_replay import replay_phase_d_on_d4
def main():
 r=replay_phase_d_on_d4();print(f"Exploratory D4 replay Macro F1={r['metrics']['macro_f1']:.3f}; coverage={r['coverage']:.1%}")
if __name__=="__main__":main()
