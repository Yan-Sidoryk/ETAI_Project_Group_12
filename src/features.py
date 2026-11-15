"""
Feature preparation shared by training, batch prediction and the client interface.

Everything here works row by row and learns nothing from the data, so a purchase gets the same
features whether it arrives alone or inside a file. Anything learned from data (imputation,
encoding, scaling) belongs in the model Pipeline instead.
"""
import pandas as pd

from src.cleaning import clean_dataset
from src.data import join_clients


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the engineered features. New features go here and nowhere else."""
    out = df.copy()
    out["is_first_purchase"] = (out["prior_purchase_count"] == 0).astype(int)
    out["purchase_month"] = pd.to_datetime(out["purchase_date"]).dt.month
    return out


def prepare(df: pd.DataFrame, clients: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Raw purchase rows -> cleaned rows with the customer columns and the engineered features."""
    out = clean_dataset(df, config["diagnosis"])
    out = join_clients(out, clients, config["client_columns"])
    return add_features(out)


def split_features_target(df: pd.DataFrame, config: dict, task: str):
    """
    Returns (X, y) for `task` ("classification" or "regression"). X has no identifier, no
    target and none of config["drop_columns"]; y is None when the data has no target column.
    """
    targets = list(config["targets"].values())
    not_features = config["id_cols"] + config["drop_columns"] + targets
    X = df.drop(columns=[c for c in not_features if c in df.columns])
    target = config["targets"][task]
    return X, (df[target] if target in df.columns else None)
