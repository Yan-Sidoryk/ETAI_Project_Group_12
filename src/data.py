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


def load_data(config: dict, dataset: str = "train"):
    """Return raw purchase rows and customer records as (purchases, clients).

    `dataset` selects the "train" or "test" path in the configuration. Only that
    purchase file and the customer file are read, so inference does not need the
    training CSV. Cleaning, feature preparation and splitting happen in the route.
    """
    purchases = pd.read_csv(config["paths"][dataset])
    clients = pd.read_csv(config["paths"]["clients"])
    return purchases, clients
