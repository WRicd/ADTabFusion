import pandas as pd
from src.phase_d.mci_conversion import select_mci_index_visits
def test_only_mci_sources_with_future_observation_are_eligible():
 f=pd.DataFrame({"RID":[1,1,2],"EXAMDATE":["2020-01-01","2021-01-01","2020-01-01"],"DX":["MCI","AD","MCI"],"x":[1,2,3]})
 result=select_mci_index_visits(f,["x"]);assert result.RID.tolist()==[1]
