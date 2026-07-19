from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.external.model_freezing import sha256_file

def main():
 root=Path("outputs/phase_d");models=sorted((root/"models").glob("*.joblib"));manifests=sorted((root/"manifests").glob("*.json"))
 required=[root/"models/transition_aware_pipeline.joblib",root/"models/calibrated_transition_pipeline.joblib",root/"models/mci_discrete_time_pipeline.joblib"]
 if not all(p.exists() for p in required):raise SystemExit("Required Phase D models are missing.")
 artifacts=[{"path":str(p).replace('\\','/'),"sha256":sha256_file(p)} for p in [*models,*manifests] if p.name!="phase_d_frozen_registry.json"]
 registry={"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"d4_used_for_decisions":False,"artifacts":artifacts}
 (root/"manifests/phase_d_frozen_registry.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
 print(f"Frozen {len(artifacts)} Phase D artifacts before D4 replay.")
if __name__=="__main__":main()
