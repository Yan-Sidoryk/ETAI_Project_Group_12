"""
Model assessment. The test file is later in time than the training file, so every split here
is ordered by purchase date: models are always fitted on the past and scored on the future.
"""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)


def split_holdout(df: pd.DataFrame, date_col: str, fraction: float):
    """
    Returns (dev, holdout), both sorted by date: `holdout` is the most recent `fraction` of the
    rows, kept out of every decision; `dev` is everything before it.
    """
    ordered = df.sort_values(date_col, key=pd.to_datetime, kind="stable").reset_index(drop=True)
    cut = len(ordered) - round(len(ordered) * fraction)
    return ordered.iloc[:cut], ordered.iloc[cut:].reset_index(drop=True)


# name -> function(y_true, predicted label, predicted probability of returning)
_METRICS = {
    "f1": lambda y, label, proba: f1_score(y, label, zero_division=0),
    "precision": lambda y, label, proba: precision_score(y, label, zero_division=0),
    "recall": lambda y, label, proba: recall_score(y, label, zero_division=0),
    "accuracy": lambda y, label, proba: accuracy_score(y, label),
    "balanced_accuracy": lambda y, label, proba: balanced_accuracy_score(y, label),
    "roc_auc": lambda y, label, proba: roc_auc_score(y, proba),
    "average_precision": lambda y, label, proba: average_precision_score(y, proba),
}


# name -> function(y_true, predicted spend), in EUR except r2
_REGRESSION_METRICS = {"rmse": root_mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}


def score_model(model, X, y, task: str, task_config: dict) -> dict:
    """Score a fitted model using configured metrics and classification threshold.

    Regression metrics use raw EUR predictions, as before. Spend clipping and
    rounding are applied only by the inference helper.
    """
    metrics = task_config["metrics"]
    if task == "regression":
        spend = model.predict(X)
        return {name: _REGRESSION_METRICS[name](y, spend) for name in metrics}

    probability = model.predict_proba(X)[:, 1]
    label = (probability >= task_config["threshold"]).astype(int)
    return {name: _METRICS[name](y, label, probability) for name in metrics}


def cross_validate_pipeline(pipeline, X, y, cv, task: str, task_config: dict) -> pd.DataFrame:
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training rows and
    scores it on that fold's validation rows. One row per fold: every metric on the validation
    rows, plus the train score and train-validation gap of `selection_metric`.
    """
    selection_metric = task_config["selection_metric"]
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        validation = score_model(model, X.iloc[val_idx], y.iloc[val_idx], task, task_config)
        train = score_model(model, X.iloc[train_idx], y.iloc[train_idx], task, task_config)[selection_metric]
        rows.append({"fold": fold, "train_rows": len(train_idx), "train": train,
                     "validation": validation[selection_metric],
                     "gap": train - validation[selection_metric], **validation})
    return pd.DataFrame(rows)


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    return "\n".join([
        f"{title} ({len(fold_scores)} time-ordered folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ])
