import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.external.schema_alignment import align_to_frozen_schema
from src.preprocessing import build_preprocessor


def test_d3_transform_does_not_change_fitted_preprocessor():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "group": ["a", "b", "a"]})
    preprocessor = build_preprocessor(["x"], ["group"])
    transformed = preprocessor.fit_transform(train)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", LogisticRegression().fit(transformed, [0, 1, 1]))]
    )
    before = pickle.dumps(pipeline.named_steps["preprocessor"])
    pipeline.predict(align_to_frozen_schema(pd.DataFrame({"x": [1.5]}), ["x", "group"]))
    after = pickle.dumps(pipeline.named_steps["preprocessor"])
    assert before == after
