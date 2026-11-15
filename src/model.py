"""
Model construction: the model alone (`build_model`) and preprocessing + model as one
estimator (`build_pipeline`). A new algorithm is one new entry in a registry below.
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

_MODELS = {
    "classification": {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
    },
}

# The floor every classifier must beat. Predicting "returns" for everyone is the best
# no-model answer for F1, because returning is the more frequent class.
FLOOR = {"classification": {"type": "dummy", "params": {"strategy": "constant", "constant": 1}}}


def build_model(task: str, model_config: dict, seed: int = None):
    """
    Builds the `task` model named by `model_config["type"]` with `model_config["params"]`.
    Models that have a `random_state` get `seed`, unless the params set one.
    """
    registry = _MODELS[task]
    model_type = model_config["type"]
    if model_type not in registry:
        raise ValueError(f"Unknown {task} model: {model_type}. Options: {list(registry)}")
    model = registry[model_type](**(model_config.get("params") or {}))
    if model.get_params().get("random_state", 0) is None:
        model.set_params(random_state=seed)
    return model


def build_pipeline(task: str, config: dict, model_config: dict = None) -> Pipeline:
    """
    The estimator that is cross-validated, refit and saved: preprocessing and model as one
    Pipeline, so the preprocessing is re-fitted on the training part of every fold. The model
    is config[task]["model"] unless `model_config` gives another one (used for the floor).
    Step names are fixed, so hyperparameters are addressed as `model__<name>`.
    """
    model_config = model_config or config[task]["model"]
    return Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(task, model_config, config["seed"])),
    ])
