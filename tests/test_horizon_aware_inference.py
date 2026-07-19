import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def test_horizon_is_part_of_frozen_inference_schema():
    frame = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0], "forecast_months": [6, 24, 6, 24]})
    pre = build_preprocessor(["x", "forecast_months"], [])
    transformed = pre.fit_transform(frame)
    pipe = Pipeline([("preprocessor", pre), ("model", RandomForestClassifier(n_estimators=5, random_state=1).fit(transformed, [0, 1, 1, 2]))])
    probability = pipe.predict_proba(pd.DataFrame({"x": [0.0], "forecast_months": [12]}))
    assert probability.shape == (1, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
