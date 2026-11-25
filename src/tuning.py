"""
Hyperparameter tuning with Optuna, on the development folds only (the hold-out is never scored here).

Each trial cross-validates one parameter set with the same time-ordered, gapped folds as `compare`. Feature
selection does not depend on the model's parameters, so every trial reuses each fold's cached selection.
"""
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

from src.evaluate import best_threshold, cross_validate_pipeline, validation_predictions
from src.model import build_pipeline


def suggest(trial, name: str, spec: list):
    """One parameter from a config spec: [int|float, low, high] or [int|float, low, high, log], or [choice, a, b, ...]."""
    kind, *values = spec
    if kind == "choice":
        return trial.suggest_categorical(name, values)
    low, high, *scale = values
    suggest_number = trial.suggest_int if kind == "int" else trial.suggest_float
    return suggest_number(name, low, high, log=scale == ["log"])


def cv_score(task: str, config: dict, model_config: dict, X, y, cv) -> tuple:
    """(score, details): F1 at the model's best threshold (classification) or mean RMSE (regression)."""
    folds = cross_validate_pipeline(build_pipeline(task, config, model_config), X, y, cv, task, config[task])
    if task == "classification":
        threshold, f1 = best_threshold(*validation_predictions(folds))
        return f1, {"threshold": threshold, "f1_at_0.5": folds["validation"].mean()}
    return folds["validation"].mean(), {"rmse_std": folds["validation"].std(ddof=1)}


def tune(config: dict, task: str, model: str, X, y, cv, trials: int) -> tuple:
    """
    Runs `trials` Optuna trials for `model`. Returns (all trials best first, best params with their types,
    score of the configured params): the configured params are scored separately, exactly as `compare` does,
    because they may leave searched parameters at the library default, which no trial reproduces.
    """
    task_config = config[task]
    base = {"type": model, **task_config["models"][model]}
    space = config["tuning"]["spaces"][model]
    maximize = task == "classification"
    configured, _ = cv_score(task, config, base, X, y, cv)

    def objective(trial):
        params = {**(base.get("params") or {}), **{name: suggest(trial, name, spec) for name, spec in space.items()}}
        score, details = cv_score(task, config, {**base, "params": params}, X, y, cv)
        for key, value in details.items():
            trial.set_user_attr(key, value)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize" if maximize else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=config["seed"]))
    study.optimize(objective, n_trials=trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"  trial {t.number:>3}: {t.value:.4f}   best {s.best_value:.4f}",
                                                 flush=True)])

    table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    table.columns = [c.removeprefix("params_").removeprefix("user_attrs_") for c in table.columns]
    table = table.rename(columns={"value": "f1_at_best" if maximize else "rmse"})
    table = table.sort_values(table.columns[1], ascending=not maximize).reset_index(drop=True)

    path = Path(config["paths"]["outputs_dir"]) / f"tuning_{task}_{model}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\nSaved all trials: {path}")
    return table, study.best_params, configured  # best_params keeps ints as ints (the table turns them into floats)
