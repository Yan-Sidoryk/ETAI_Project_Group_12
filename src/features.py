"""
Feature preparation shared by training, batch prediction and the client interface.

Everything here works row by row and learns nothing from the data, so a purchase gets the same
features whether it arrives alone or inside a file. Anything learned from data (imputation,
encoding, scaling) belongs in the model Pipeline instead.
"""
import pandas as pd

from src.cleaning import clean_dataset


def prepare(df: pd.DataFrame, clients: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean purchase records, attach customer attributes, and derive features.

        Apply the cleaning rules in `config["diagnosis"]`, then left-join the
        customer columns in `config["client_columns"]` on `customer_id`. The join
        keeps purchases with no matching customer and can duplicate purchase
        rows if `clients` contains multiple matches for a customer.

        Set `is_first_purchase` to 1 when `prior_purchase_count` is zero and to 0
        otherwise. Extract `purchase_month` as the calendar month (1–12) from
        `purchase_date`.

        Imputation, encoding, and scaling are fitted separately in the pipeline.
        """
    out = clean_dataset(df, config["diagnosis"])
    client_columns = ["customer_id"] + config["client_columns"]
    out = out.merge(clients[client_columns], on="customer_id", how="left")
    out["is_first_purchase"] = (out["prior_purchase_count"] == 0).astype(int)
    out["purchase_month"] = pd.to_datetime(out["purchase_date"]).dt.month
    return out


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
