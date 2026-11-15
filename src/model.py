"""
Model construction: the model alone (`build_model`) and preprocessing + model as one
estimator (`build_pipeline`). A new algorithm is one new entry in a registry below.
"""
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


def build_model(task: str, model_config: dict, seed: int = None):
    """
    Builds the `task` model named by `model_config["type"]` with `model_config["params"]`.
    The model gets `seed` as its `random_state`, unless the params set one. With
    `model_config["log_target"]` a regressor is fitted on log(1 + spend), which is close to
    symmetric while spend itself is heavily skewed, and its predictions are converted back to EUR.
    """
    registry = _MODELS[task]
    model_type = model_config["type"]
    if model_type not in registry:
        raise ValueError(f"Unknown {task} model: {model_type}. Options: {list(registry)}")
    params = model_config.get("params") or {}
    model = registry[model_type](**params)
    if "random_state" not in params:
        try:
            model.set_params(random_state=seed)
        except ValueError:      # the model has no randomness (the mean-spend floor)
            pass
    if model_config.get("log_target"):
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def configured_model(config: dict, task: str, name: str = None) -> dict:
    """The model config of `name` in config[task]["models"]; by default the one `train` saves."""
    name = name or config[task]["model"]
    models = config[task]["models"]
    if name not in models:
        raise ValueError(f"{task}.model is {name}, which is not in {task}.models: {list(models)}")
    return {"type": name, **models[name]}


def build_pipeline(task: str, config: dict, model_config: dict = None) -> Pipeline:
    """
    The estimator that is cross-validated, refit and saved: preprocessing and model as one
    Pipeline, so the preprocessing is re-fitted on the training part of every fold. The model
    is the one named by config[task]["model"] unless `model_config` gives another one.
    Step names are fixed, so hyperparameters are addressed as `model__<name>`
    (`model__regressor__<name>` for a regressor with `log_target`).
    """
    model_config = model_config or configured_model(config, task)
    return Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(task, model_config, config["seed"])),
    ])
