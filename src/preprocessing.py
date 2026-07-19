from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    impute_strategy: str = "median",
    add_missing_indicators: bool = False,
) -> ColumnTransformer:
    """Build a train-fitted-only preprocessing transformer."""
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=impute_strategy,
                    add_indicator=add_missing_indicators,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
