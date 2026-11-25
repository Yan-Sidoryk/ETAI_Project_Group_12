"""
SHAP explanations of the saved models (global and per purchase).

Rows go through the saved pipeline's own steps (feature selection, then preprocessing); SHAP's TreeExplainer then
explains the fitted tree model. One-hot levels and missing-value flags are summed back into their raw feature, so
every explanation is in terms of the columns of test.csv. Units: probability (classification) or EUR (regression).
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import VotingClassifier

from src.features import prepare, split_features_target


def model_inputs(model: dict, purchases: pd.DataFrame, clients: pd.DataFrame, task: str) -> pd.DataFrame:
    """The feature columns a saved model receives for these purchases (same preparation as prediction)."""
    rows = prepare(purchases, clients, model["config"])
    X, _ = split_features_target(rows, model["config"], task)
    return X[model["features"]]


def _raw_name(encoded: str, categorical: list) -> str:
    """'categorical__salesperson_X' -> 'salesperson'; 'remainder__missingindicator_spend_90d' -> 'spend_90d'."""
    name = encoded.split("__", 1)[1].removeprefix("missingindicator_")
    return next((c for c in categorical if name == c or name.startswith(c + "_")), name)


def shap_values(model: dict, X: pd.DataFrame, task: str) -> tuple:
    """
    (SHAP values per purchase and raw feature, base value, the selected raw feature values).
    For every purchase, base value + the row's SHAP values = the model's prediction (probability or EUR).
    """
    steps = model["pipeline"].named_steps
    selected = steps["select"].transform(X) if "select" in steps else X
    encoded = steps["prep"].transform(selected)
    estimator = steps["model"]
    if hasattr(estimator, "regressor_"):  # a log-target wrapper: explain the inner model (log scale)
        estimator = estimator.regressor_
    if isinstance(estimator, VotingClassifier):
        raise SystemExit("SHAP explanations support single tree models; the saved model is a voting ensemble.")
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(encoded))
    base = np.atleast_1d(explainer.expected_value)
    if task == "classification" and values.ndim == 3:  # (rows, features, classes): keep "returns" (class 1)
        values, base = values[..., 1], base[-1]
    else:
        base = base[0]
    categorical = model["config"]["preprocessing"]["categorical_features"]
    names = [_raw_name(n, categorical) for n in steps["prep"].get_feature_names_out()]
    raw = pd.DataFrame(values, columns=names, index=X.index).T.groupby(level=0, sort=False).sum().T
    return raw, float(base), selected


def global_importance(raw: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """
    Mean |SHAP| per feature (its average effect on a prediction) and, for numeric features, the direction:
    the Spearman correlation between the feature's value and its SHAP value (+: higher value -> higher prediction).
    """
    table = pd.DataFrame({"mean_abs_shap": raw.abs().mean()})
    table["direction"] = [
        raw[c].corr(selected[c], method="spearman") if pd.api.types.is_numeric_dtype(selected[c]) else np.nan
        for c in table.index]
    return table.sort_values("mean_abs_shap", ascending=False)


def top_factors(raw_row: pd.Series, values: pd.Series, n: int = 5) -> list:
    """The `n` features that moved one prediction the most: (feature, its value, SHAP contribution)."""
    order = raw_row.abs().sort_values(ascending=False).index[:n]
    return [(name, values.get(name), raw_row[name]) for name in order]
