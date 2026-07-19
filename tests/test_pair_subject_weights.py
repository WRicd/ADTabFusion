import pandas as pd
from src.phase_d.transition_model import build_transition_pairs

def test_subject_balanced_pair_weights_sum_equally_per_rid():
    frame=pd.DataFrame({"RID":[1,1,1,2,2],"EXAMDATE":["2020-01-01","2021-01-01","2022-01-01","2020-01-01","2021-01-01"],"DX":["CN","MCI","AD","MCI","AD"],"x":[1,2,3,4,5]})
    pairs=build_transition_pairs(frame,["x"])
    assert pairs.groupby("RID")["subject_weight"].sum().round(12).nunique()==1
