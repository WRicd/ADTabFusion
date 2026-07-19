import numpy as np
import pandas as pd

from src.external.bootstrap import resample_subject_blocks, subject_bootstrap_ci


def test_bootstrap_resamples_complete_subject_blocks():
    frame = pd.DataFrame({"RID": [1, 1, 2], "value": [10, 11, 20]})
    sampled = resample_subject_blocks(frame, np.random.default_rng(4))
    for _, block in sampled.groupby("_bootstrap_subject"):
        rid = block["RID"].iloc[0]
        assert len(block) == int((frame["RID"] == rid).sum())


def test_subject_bootstrap_is_seeded():
    frame = pd.DataFrame(
        {
            "RID": [1, 2, 3, 4, 5, 6], "truth": [0, 1, 2, 0, 1, 2],
            "pred": [0, 1, 2, 1, 1, 2], "p0": [.8, .1, .1, .3, .1, .1],
            "p1": [.1, .8, .1, .5, .8, .1], "p2": [.1, .1, .8, .2, .1, .8],
        }
    )
    a = subject_bootstrap_ci(frame, "truth", "pred", ["p0", "p1", "p2"], repetitions=20, seed=2)
    b = subject_bootstrap_ci(frame, "truth", "pred", ["p0", "p1", "p2"], repetitions=20, seed=2)
    pd.testing.assert_frame_equal(a, b)
