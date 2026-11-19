"""
Build preprocessing and the configured estimator together in one pipeline.
A new algorithm is one entry in the registry below.
"""
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
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


def build_pipeline(task: str, config: dict, model_config: dict) -> Pipeline:
    """Build an unfitted preprocessing/model pipeline from explicit model settings.

    `model_config` contains "type", optional "params", and optional "log_target".
    Each fold fits its own preprocessing. The seed is used unless params already
    specify random_state. A log-target regressor converts predictions back to EUR.
    Steps are "prep", then "select" unless feature_selection.method is "none", then "model".
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

    steps = [("prep", build_preprocessor(config["preprocessing"]))]
    selection = config.get("feature_selection") or {}
    if selection.get("method", "none") != "none" and model_type != "dummy":
        steps.append(("select", build_selector(task, config)))
    return Pipeline(steps + [("model", model)])


class PermutationSelector(SelectorMixin, BaseEstimator):
    """Keeps the columns whose permutation importance is above zero on rows the forest did not see.

    fit() holds out `holdout` of its (training-fold) rows, fits the forest on the rest, and shuffles
    each column of the held-out rows `n_repeats` times. A column that carries no signal, such as
    random_noise, scores about 0 however many distinct values it has, unlike impurity importance.
    """

    def __init__(self, estimator, scoring, holdout=0.25, n_repeats=5, random_state=None):
        self.estimator, self.scoring, self.holdout = estimator, scoring, holdout
        self.n_repeats, self.random_state = n_repeats, random_state

    def fit(self, X, y):
        # ponytail: random split inside the fold; repeat customers sit on both sides, a grouped split is stricter
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=self.holdout, random_state=self.random_state)
        self.estimator_ = clone(self.estimator).fit(X_fit, y_fit)
        result = permutation_importance(self.estimator_, X_val, y_val, scoring=self.scoring,
                                        n_repeats=self.n_repeats, random_state=self.random_state, n_jobs=-1)
        self.importances_, self.importances_std_ = result.importances_mean, result.importances_std
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        return self.importances_ > 0


def build_selector(task: str, config: dict):
    """The configured selection step. It runs on the encoded columns, so a single category level
    (one salesperson) can be kept while the other levels are dropped. Refitted in every fold.

      model        SelectFromModel on random-forest impurity importance, cut at `threshold`
      permutation  PermutationSelector: permutation importance > 0 on rows held out of the forest
    """
    selection = config["feature_selection"]
    forest = _MODELS[task]["random_forest"](**selection.get("params", {}), random_state=config["seed"])
    if selection["method"] == "model":
        return SelectFromModel(forest, threshold=selection.get("threshold", "median"))
    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    return PermutationSelector(forest, scoring, n_repeats=selection.get("n_repeats", 5), random_state=config["seed"])
