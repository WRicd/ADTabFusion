import pandas as pd
from src.phase_d.transition_model import build_transition_pairs

def test_transition_matrix_keeps_reversions():
    frame=pd.DataFrame({"RID":[1,1],"EXAMDATE":["2020-01-01","2021-01-01"],"DX":["AD","MCI"],"x":[1,2]})
    pairs=build_transition_pairs(frame,["x"])
    assert pairs.iloc[0]["transition"]=="AD -> MCI"
