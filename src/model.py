"""
Build preprocessing and the configured estimator together in one pipeline.
A new algorithm is one entry in the registry below.
"""
import os

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import build_preprocessor
from src.selection import FeatureSelector

_MODELS = {
    "classification": {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
    },
    "regression": {
        "dummy": DummyRegressor,
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
        "catboost": CatBoostRegressor,
    },
}

# The floor every model must beat: the best answer that ignores the features.
#   classification: "returns" for everyone (best for F1, because returning is the more frequent class)
#   regression: the mean spend of the training rows (best constant for RMSE)
FLOOR = {
    "classification": {"type": "dummy", "params": {"strategy": "constant", "constant": 1}},
    "regression": {"type": "dummy", "params": {"strategy": "mean"}},
}


def build_pipeline(task: str, config: dict, model_config: dict) -> Pipeline:
    """Build an unfitted preprocessing/model pipeline from explicit model settings.

    `model_config` contains "type", optional "params", and optional "log_target".
    Each fold fits its own preprocessing. The seed is used unless params already
    specify random_state. A log-target regressor converts predictions back to EUR.
    Steps: "select" (per-fold feature selection, when feature_selection.method is "permutation"), "prep", "model".
    """
    registry = _MODELS[task]
    model_type = model_config["type"]
    if model_type not in registry:
        raise ValueError(f"Unknown {task} model: {model_type}. Options: {list(registry)}")
    params = model_config.get("params") or {}
    model = registry[model_type](**params)
    if "random_state" not in params:
        try:
            model.set_params(random_state=config["seed"])
        except ValueError:  # Some estimators, such as DummyRegressor, have no seed.
            pass
    if model_config.get("log_target"):
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

    preprocessor = build_preprocessor(config["preprocessing"])
    selection = config.get("feature_selection") or {}
    if selection.get("method", "none") == "none" or model_type == "dummy":
        return Pipeline([("prep", preprocessor), ("model", model)])
    selector = FeatureSelector(
        task, config["preprocessing"], forest_params=selection.get("params"), n_splits=selection["n_splits"],
        n_repeats=selection["n_repeats"], corr_threshold=selection["corr_threshold"],
        reference=selection["reference"], random_state=config["seed"])
    # Fitted steps are cached on disk, keyed by their settings and input rows. The folds are fixed, so each fold's
    # selection is computed once and reused by every model, every `compare`/`train` run and every tuning trial.
    # ponytail: the key ignores code; delete outputs/cache after editing src/selection.py or src/preprocessing.py
    return Pipeline([("select", selector), ("prep", preprocessor), ("model", model)],
                    memory=os.path.join(config["paths"]["outputs_dir"], "cache"))
