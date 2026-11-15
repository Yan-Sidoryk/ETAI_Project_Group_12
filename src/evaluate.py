"""
Model assessment. The test file is later in time than the training file, so every split here
is ordered by purchase date: models are always fitted on the past and scored on the future.
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_holdout(df: pd.DataFrame, date_col: str, fraction: float):
    """
    Returns (dev, holdout), both sorted by date: `holdout` is the most recent `fraction` of the
    rows, kept out of every decision; `dev` is everything before it.
    """
    ordered = df.sort_values(date_col, key=pd.to_datetime, kind="stable").reset_index(drop=True)
    cut = len(ordered) - round(len(ordered) * fraction)
    return ordered.iloc[:cut], ordered.iloc[cut:].reset_index(drop=True)


def make_cv(config: dict) -> TimeSeriesSplit:
    """
    Expanding-window folds: each fold trains on the earliest rows and validates on the block
    that follows. The rows passed to it must be sorted by date, as `split_holdout` returns them.
    """
    return TimeSeriesSplit(n_splits=config["validation"]["n_splits"])
