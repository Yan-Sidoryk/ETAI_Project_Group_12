"""
The learned preprocessing: imputation, scaling and encoding. It is the first step of the model
Pipeline, so it is fitted on training rows only (again in every validation fold) and saved
together with the model.
"""

import re

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder,
                                   OrdinalEncoder, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

_SCALERS = {
    "none": (None, {}),
    "standard": (StandardScaler, {}),
    "minmax": (MinMaxScaler, {}),
    "robust": (RobustScaler, {}),
    "maxabs": (MaxAbsScaler, {}),
    "normalize": (Normalizer, {}),
    "power": (PowerTransformer, {}),
    "quantile": (QuantileTransformer, {"random_state": 42}),
}
_NUMERIC_IMPUTERS = {
    "median": (SimpleImputer, {"strategy": "median", "add_indicator": True}),
    "mean": (SimpleImputer, {"strategy": "mean", "add_indicator": True}),
    "most_frequent": (SimpleImputer, {"strategy": "most_frequent", "add_indicator": True}),
    "constant": (SimpleImputer, {"strategy": "constant", "fill_value": 0, "add_indicator": True}),
    "knn": (KNNImputer, {"add_indicator": True}),
    "none": (None, {}),
}
_CATEGORICAL_IMPUTERS = {
    "constant": (SimpleImputer, {"strategy": "constant", "fill_value": "missing"}),
    "most_frequent": (SimpleImputer, {"strategy": "most_frequent"}),
    "none": (None, {}),
}


def _build_component(setting, registry, key):
    """Accept a name or {type: name, params: {...}}, with useful config errors."""
    if isinstance(setting, str):
        name, params = setting, {}
    elif isinstance(setting, dict) and set(setting) <= {"type", "params"}:
        name, params = setting.get("type"), setting.get("params", {})
    else:
        raise ValueError(f"preprocessing.{key} must be a name or a mapping with type and params")
    if not isinstance(name, str) or name not in registry:
        raise ValueError(f"Unknown preprocessing.{key}: {name!r}. Options: {list(registry)}")
    if not isinstance(params, dict):
        raise ValueError(f"preprocessing.{key}.params must be a mapping")
    constructor, defaults = registry[name]
    if constructor is None:
        if params:
            raise ValueError(f"preprocessing.{key}: 'none' does not accept params")
        return "passthrough"
    # YAML represents ranges as lists; sklearn requires tuples for some ranges.
    params = {key: tuple(value) if key in {"feature_range", "quantile_range"}
              and isinstance(value, list) else value for key, value in params.items()}
    try:
        return constructor(**{**defaults, **params})
    except TypeError as exc:
        raise ValueError(f"Invalid params for preprocessing.{key} ({name}): {exc}") from exc


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Build configurable imputation -> encoding/scaling, fitted only inside each fold.

    Each component accepts a name or a mapping with `type` and optional `params`.
    Omitted settings preserve median + indicators, constant "missing", and one-hot
    defaults. Listed categorical columns are encoded; all other columns are numeric.
    """
    encoders = {
        "onehot": (OneHotEncoder, {"handle_unknown": "ignore", "sparse_output": False,
                                  "min_frequency": preprocessing_config.get("min_frequency")}),
        "ordinal": (OrdinalEncoder, {"handle_unknown": "use_encoded_value", "unknown_value": -1,
                                    "min_frequency": preprocessing_config.get("min_frequency")}),
    }

    def component(key, default, registry):
        return _build_component(preprocessing_config.get(key, default), registry, key)

    numeric = Pipeline([
        ("impute", component("numeric_imputer", "median", _NUMERIC_IMPUTERS)),
        ("scale", component("scaler", "standard", _SCALERS)),
    ])
    categorical = Pipeline([
        ("impute", component("categorical_imputer", "constant", _CATEGORICAL_IMPUTERS)),
        ("encode", component("encoder", "onehot", encoders)),
    ])
    # Matched by name, so a categorical column listed in drop_columns is simply absent instead of an error.
    names = "|".join(re.escape(c) for c in preprocessing_config["categorical_features"])
    return ColumnTransformer(
        [("categorical", categorical, make_column_selector(pattern=f"^(?:{names})$"))],
        remainder=numeric,
    )
