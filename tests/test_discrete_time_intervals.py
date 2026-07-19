import pandas as pd
from src.phase_d.discrete_time import build_person_period_data
def test_intervals_stop_after_first_conversion():
 index=pd.DataFrame({"RID":[1],"max_followup_months":[60],"first_ad_months":[20],"x":[1]});p=build_person_period_data(index,[12,24,36]);assert p.elapsed_months.tolist()==[12,24] and p.event.tolist()==[0,1]
