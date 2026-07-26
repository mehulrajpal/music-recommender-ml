"""
preprocessing.py
----------------
Builds a scikit-learn ColumnTransformer from whatever numeric/categorical
features schema.py detected. This is what lets the exact same training
code run unchanged on a totally different dataset -- the preprocessor
shape is derived from the data, not hard-coded.
"""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(numeric_features, categorical_features, config: dict) -> ColumnTransformer:
    pre_cfg = config["preprocessing"]

    numeric_steps = [("imputer", SimpleImputer(strategy=pre_cfg["numeric_impute_strategy"]))]
    if pre_cfg.get("scale_numeric", True):
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=pre_cfg["categorical_impute_strategy"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
