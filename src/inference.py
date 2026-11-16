"""Load saved artifacts and apply the shared classification and spend policies."""
import os

import joblib
import numpy as np

from src.features import prepare, split_features_target

MODEL_FILES = {"classification": "classifier.joblib", "regression": "regressor.joblib"}


def load_model(config: dict, task: str) -> dict:
    """Load a trained artifact; report the required training command if absent."""
    path = os.path.join(config["paths"]["models_dir"], MODEL_FILES[task])
    if not os.path.exists(path):
        raise SystemExit(f"No saved model at {path}. Run first: python main.py train --task {task}")
    return joblib.load(path)


def classify(model: dict, purchases, clients):
    """Return probabilities and binary decisions using the artifact's settings."""
    rows = prepare(purchases, clients, model["config"])
    X, _ = split_features_target(rows, model["config"], "classification")
    probability = model["pipeline"].predict_proba(X[model["features"]])[:, 1]
    decision = (probability >= model["threshold"]).astype(int)
    return probability, decision


def predict_spend(model: dict, purchases, clients):
    """Predict conditional 90-day spend in EUR, clipped at zero and rounded to cents."""
    rows = prepare(purchases, clients, model["config"])
    X, _ = split_features_target(rows, model["config"], "regression")
    spend = model["pipeline"].predict(X[model["features"]])
    return np.clip(spend, 0, None).round(2)
