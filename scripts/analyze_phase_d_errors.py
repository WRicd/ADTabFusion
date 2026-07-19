from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.phase_d.error_analysis import generate_transition_error_analysis
if __name__=="__main__":print(generate_transition_error_analysis())
