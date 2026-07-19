import pandas as pd
from src.phase_d.transition_model import build_transition_pairs

def test_source_and_future_diagnosis_are_separate_and_future_not_feature():
    frame=pd.DataFrame({"RID":[1,1],"EXAMDATE":["2020-01-01","2021-01-01"],"DX":["CN","AD"],"x":[4,99]})
    pair=build_transition_pairs(frame,["x"]).iloc[0]
    assert pair.SOURCE_DX=="CN" and pair.FUTURE_DX=="AD" and pair.x==4
