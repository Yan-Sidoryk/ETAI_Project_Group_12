"""
Feature preparation shared by training, batch prediction and the client interface.

Everything here works row by row and learns nothing from the data, so a purchase gets the same
features whether it arrives alone or inside a file. Anything learned from data (imputation,
encoding, scaling) belongs in the model Pipeline instead.
"""
import pandas as pd

from src.cleaning import clean_dataset


def prepare(df: pd.DataFrame, clients: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean purchases, join customer attributes, and add row-local features.

    This preparation learns nothing from the data and preserves purchase rows.
    Imputation, encoding and scaling are fitted separately inside the pipeline.
    """
    out = clean_dataset(df, config["diagnosis"])
    client_columns = ["customer_id"] + config["client_columns"]
    out = out.merge(clients[client_columns], on="customer_id", how="left")
    return add_features(out)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the engineered features, row by row. New features go here and nowhere else."""
    out = df.copy()
    out["is_first_purchase"] = (out["prior_purchase_count"] == 0).astype(int)
    out["purchase_month"] = pd.to_datetime(out["purchase_date"]).dt.month

    def ratio(num, den):  # zero or missing denominator -> NaN, left to the pipeline imputer
        return num / den.where(den != 0)

    # Recency and frequency
    out["overdue_ratio"] = ratio(out["days_since_previous_purchase"], out["prior_mean_gap_days"])
    out["purchase_rate"] = ratio(out["prior_purchase_count"], out["customer_tenure_days"])
    # Trend: last 90 days against the average quarter of the last year
    out["spend_trend_90d"] = ratio(out["spend_90d"] * 4, out["spend_365d"])
    out["count_trend_90d"] = ratio(out["purchase_count_90d"], out["purchase_count_365d"])
    # Current purchase against the customer's usual one
    out["value_vs_usual"] = ratio(out["Total a Pagar"], out["prior_mean_purchase_value"])
    # Margin, discounts, basket shape
    out["margin_rate"] = ratio(out["Total Liquido"] - out["Custo"], out["Total Liquido"])
    out["discount_rate"] = ratio(out["Total Desc_ Global"] + out["Total Desc_ Linha"], out["Total Bruto"])
    out["value_per_line"] = ratio(out["Total Liquido"], out["N_ Detalhes"])
    out["shipping_rate"] = ratio(out["Total Portes"], out["Total Liquido"])
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
