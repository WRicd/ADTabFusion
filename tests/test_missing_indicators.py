import pandas as pd

from src.preprocessing import build_preprocessor


def test_missing_indicators_add_columns_and_fit_train_only():
    train = pd.DataFrame({"value": [1.0, None, 3.0]})
    test = pd.DataFrame({"value": [1000.0]})
    plain = build_preprocessor(["value"], [], add_missing_indicators=False)
    indicated = build_preprocessor(["value"], [], add_missing_indicators=True)
    plain.fit(train)
    indicated.fit(train)
    assert plain.transform(test).shape[1] == 1
    assert indicated.transform(test).shape[1] == 2
    statistic = indicated.named_transformers_["num"].named_steps["imputer"].statistics_[0]
    assert statistic == 2.0

