import pandas as pd
from src.phase_d.mci_conversion import build_landmark_cohort
def test_24_month_landmark_uses_18_to_30_month_tolerance():
 index=pd.DataFrame({"RID":[1,2,3],"max_followup_months":[20,30,40],"first_ad_months":[20,31,float('nan')]})
 cohort,_=build_landmark_cohort(index,24,6);labels=cohort.set_index("RID").label.to_dict();assert labels[1]==1 and labels[2]==0 and labels[3]==0
