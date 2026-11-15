"""Loading: the configuration, the raw CSVs and the customer table join."""
import pandas as pd
import yaml


def load_config(path: str = "config.yaml", diagnosis_path: str = "diagnosis.yaml") -> dict:
    """Returns config.yaml as a dict, with the data rules of diagnosis.yaml under "diagnosis"."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(diagnosis_path, "r", encoding="utf-8") as f:
        config["diagnosis"] = yaml.safe_load(f)
    return config


def load_raw(config: dict):
    """Returns (train, test, clients) exactly as stored in the CSVs."""
    paths = config["paths"]
    return pd.read_csv(paths["train"]), pd.read_csv(paths["test"]), pd.read_csv(paths["clients"])


def join_clients(df: pd.DataFrame, clients: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Adds the customer-level `columns` to every purchase row (left join on customer_id).
    Row count and order are kept; a customer missing from `clients` gets NaN.
    """
    return df.merge(clients[["customer_id"] + columns], on="customer_id", how="left")
