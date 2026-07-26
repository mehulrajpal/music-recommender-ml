"""
models.py
---------
Defines the classifiers used for genre / artist prediction:

  - "knn"      : K-Nearest Neighbors               (instance-based)
  - "bagging"  : Random Forest                       (bagging of decision trees)
  - "boosting" : Gradient Boosting                    (sklearn boosting of decision trees)
  - "xgboost"  : XGBoost                              (optimized gradient-boosted trees)
  - "ensemble" : Soft-voting combination of knn + bagging + xgboost (default, usually best)

Each is wrapped in a full sklearn Pipeline: preprocessor -> classifier,
so calling .fit(X, y) / .predict(X) handles raw dataframes end to end.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def _base_estimators(config: dict):
    m = config["models"]
    knn = KNeighborsClassifier(
        n_neighbors=m["knn"]["n_neighbors"],
        weights=m["knn"]["weights"],
    )
    bagging = RandomForestClassifier(
        n_estimators=m["bagging"]["n_estimators"],
        max_depth=m["bagging"]["max_depth"],
        min_samples_leaf=m["bagging"]["min_samples_leaf"],
        random_state=config["training"]["random_state"],
        n_jobs=-1,
    )
    boosting = GradientBoostingClassifier(
        n_estimators=m["boosting"]["n_estimators"],
        learning_rate=m["boosting"]["learning_rate"],
        max_depth=m["boosting"]["max_depth"],
        random_state=config["training"]["random_state"],
    )
    xgb = XGBClassifier(
        n_estimators=m["xgboost"]["n_estimators"],
        learning_rate=m["xgboost"]["learning_rate"],
        max_depth=m["xgboost"]["max_depth"],
        subsample=m["xgboost"]["subsample"],
        colsample_bytree=m["xgboost"]["colsample_bytree"],
        random_state=config["training"]["random_state"],
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    return knn, bagging, boosting, xgb


def build_model_pipeline(model_type: str, numeric_features, categorical_features, config: dict) -> Pipeline:
    """
    model_type: one of "knn", "bagging", "boosting", "xgboost", "ensemble"
    Returns a full sklearn Pipeline (preprocessing + classifier).
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features, config)
    knn, bagging, boosting, xgb = _base_estimators(config)

    if model_type == "knn":
        clf = knn
    elif model_type == "bagging":
        clf = bagging
    elif model_type == "boosting":
        clf = boosting
    elif model_type == "xgboost":
        clf = xgb
    elif model_type == "ensemble":
        clf = VotingClassifier(
            estimators=[("knn", knn), ("bagging", bagging), ("xgboost", xgb)],
            voting=config["models"]["ensemble"]["voting"],
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose one of knn/bagging/boosting/xgboost/ensemble."
        )

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


AVAILABLE_MODELS = ["knn", "bagging", "boosting", "xgboost", "ensemble"]
