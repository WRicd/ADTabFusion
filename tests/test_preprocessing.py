import pandas as pd

from src.feature_groups import infer_feature_types
from src.preprocessing import build_preprocessor


def test_preprocessor_fits_train_only_shape():
    train = pd.DataFrame({"age": [70, 71, None], "sex": ["M", "F", "F"]})
    test = pd.DataFrame({"age": [90], "sex": ["X"]})
    numeric, categorical = infer_feature_types(train, ["age", "sex"])
    pre = build_preprocessor(numeric, categorical)
    pre.fit(train)
    out = pre.transform(test)
    assert out.shape[0] == 1

