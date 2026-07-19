import pandas as pd
from src.phase_d.mci_conversion import build_landmark_cohort
def test_insufficient_followup_is_censored_not_negative():
 index=pd.DataFrame({"RID":[1,2],"max_followup_months":[10,30],"first_ad_months":[float('nan'),float('nan')]})
 cohort,summary=build_landmark_cohort(index,24,6);assert 1 not in cohort.RID.tolist();assert summary["censored_subjects"]==1
