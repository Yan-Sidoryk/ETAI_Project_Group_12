"""
Model assessment. The test file is later in time than the training file, so every split here
is ordered by purchase date: models are always fitted on the past and scored on the future.
"""
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)

def split_holdout(df: pd.DataFrame, date_col: str, fraction: float, gap_days: int):
    """
    Returns (dev, holdout) sorted by date.
    The 'dev' set ends `gap_days` days before the first purchase in the 'holdout' set, preventing leakage.
    """
    ordered = df.sort_values(date_col, key=pd.to_datetime, kind="stable").reset_index(drop=True)
    cut = len(ordered) - round(len(ordered) * fraction)
    
    holdout = ordered.iloc[cut:].reset_index(drop=True)
    holdout_start_date = pd.to_datetime(holdout[date_col].min())
    
    # Apply the gap_days embargo
    cutoff_date = holdout_start_date - pd.Timedelta(days=gap_days)
    
    dev = ordered[pd.to_datetime(ordered[date_col]) < cutoff_date].reset_index(drop=True)
    return dev, holdout

class TimeCalendarSplit:
    """
    Replaces the standard TimeSeriesSplit. Imposes a strict calendar time gap 
    instead of row count, ensuring isolation between blocks.
    """
    def __init__(self, date_series, n_splits, gap_days):
        self.date_series = pd.to_datetime(date_series).reset_index(drop=True)
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        fold_size = len(X) // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            val_end = (i + 1) * fold_size if i < self.n_splits else len(X)
            val_start = i * fold_size
            val_idx = indices[val_start:val_end]
            
            val_start_date = self.date_series.iloc[val_start]
            
            # Training ends strictly gap_days before validation starts
            cutoff_date = val_start_date - pd.Timedelta(days=self.gap_days)
            train_idx = indices[(self.date_series < cutoff_date) & (indices < val_start)]
            
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


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
