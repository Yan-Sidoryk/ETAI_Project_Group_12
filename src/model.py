"""
Model construction: the model alone (`build_model`) and preprocessing + model as one
estimator (`build_pipeline`). A new algorithm is one new entry in a registry below.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

_MODELS = {
    "classification": {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
    },
    "regression": {
        "dummy": DummyRegressor,
        "ridge": Ridge,
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
    Models that have a `random_state` get `seed`, unless the params set one. With
    `model_config["log_target"]` a regressor is fitted on log(1 + spend), which is close to
    symmetric while spend itself is heavily skewed, and its predictions are converted back to EUR.
    """
    registry = _MODELS[task]
    model_type = model_config["type"]
    if model_type not in registry:
        raise ValueError(f"Unknown {task} model: {model_type}. Options: {list(registry)}")
    model = registry[model_type](**(model_config.get("params") or {}))
    if model.get_params().get("random_state", 0) is None:
        model.set_params(random_state=seed)
    if model_config.get("log_target"):
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def build_pipeline(task: str, config: dict, model_config: dict = None) -> Pipeline:
    """
    The estimator that is cross-validated, refit and saved: preprocessing and model as one
    Pipeline, so the preprocessing is re-fitted on the training part of every fold. The model
    is config[task]["model"] unless `model_config` gives another one (used for the floor).
    Step names are fixed, so hyperparameters are addressed as `model__<name>`
    (`model__regressor__<name>` for a regressor with `log_target`).
    """
    model_config = model_config or config[task]["model"]
    return Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(task, model_config, config["seed"])),
    ])
