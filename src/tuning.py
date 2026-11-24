"""Optuna model-parameter search using only gapped development folds."""
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import time

import numpy as np
import optuna
import pandas as pd
import yaml

from src.evaluate import cross_validate_pipeline
from src.model import build_pipeline


def validate_search_space(space):
    """Validate YAML distributions before starting any expensive trials."""
    constructors = {"int": optuna.distributions.IntDistribution,
                    "float": optuna.distributions.FloatDistribution,
                    "categorical": optuna.distributions.CategoricalDistribution}
    if not isinstance(space, dict) or not space:
        raise ValueError("Each tuned model needs a non-empty search_space mapping")
    for name, spec in space.items():
        if not isinstance(spec, dict) or spec.get("type") not in constructors:
            raise ValueError(f"search_space.{name}: use type int, float, or categorical")
        try:
            constructors[spec["type"]](**{k: v for k, v in spec.items() if k != "type"})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"search_space.{name}: {exc}") from exc


def suggest_parameters(trial, space):
    """Map YAML specifications to Optuna's public suggestion methods."""
    return {name: getattr(trial, f"suggest_{spec['type']}")(
        name, **{k: v for k, v in spec.items() if k != "type"})
        for name, spec in space.items()}


def run_tuning(config, task, X, y, cv, models=None, n_trials=None, timeout=None):
    """Tune selected models; checkpoint CSVs and a usable best_config.yaml.

    Trial parameters override fixed model params. Preprocessing, classification
    threshold and log_target remain fixed. The caller supplies development rows
    only; neither the holdout nor the test file is scored or fitted here.
    """
    task_config = config[task]
    names = list(dict.fromkeys(models if models is not None else [task_config["model"]]))
    if not names or any(name not in task_config["models"] for name in names):
        raise ValueError(f"Choose models from {list(task_config['models'])}")
    options = config.get("tuning", {})
    n_trials = n_trials if n_trials is not None else options.get("n_trials", 30)
    timeout = timeout if timeout is not None else options.get("timeout_seconds")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("tuning.n_trials must be a positive integer")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("tuning.timeout_seconds must be positive or null")
    metric = task_config["selection_metric"]
    if metric not in task_config["metrics"]:
        raise ValueError("selection_metric must be included in metrics")
    minimize = metric in ("rmse", "mae")
    for name in names:
        model = task_config["models"][name]
        validate_search_space(model.get("search_space"))
        # Catch misspelled estimator parameters before launching a study.
        estimator = build_pipeline(task, config, {"type": name, **model}).named_steps["model"]
        if model.get("log_target"):
            estimator = estimator.regressor
        # CatBoost.get_params() exposes only explicitly supplied parameters.
        known = set(estimator.get_params()) | set(inspect.signature(type(estimator).__init__).parameters)
        unknown = set(model["search_space"]) - known
        if unknown:
            raise ValueError(f"{name}: unknown search_space parameters: {sorted(unknown)}")

    splits = list(cv.split(X))
    if len(splits) < 2 or any(not len(train) or not len(val) for train, val in splits):
        raise ValueError("Tuning requires at least two non-empty training/validation folds")

    class FixedCV:
        def split(self, X):
            return iter(splits)

    directory = Path(config["paths"]["outputs_dir"]) / f"tuning_{task}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    directory.mkdir(parents=True, exist_ok=False)
    trial_rows, fold_rows = [], []
    print(f"Tuning {names}: up to {n_trials} trials per model x {len(splits)} folds; "
          f"{'minimize' if minimize else 'maximize'} {metric}", flush=True)
    print(f"Results: {directory}", flush=True)

    def checkpoint():
        trials = pd.DataFrame(trial_rows) if trial_rows else pd.DataFrame(
            columns=["task", "model", "trial", "params", "sampled_params", "status", "error", "cv_mean"])
        trials = trials[[c for c in trials.columns if c not in ("params", "sampled_params")]
                        + ["params", "sampled_params"]]
        display_f1 = task == "classification" and "f1" in task_config["metrics"]
        score = "f1_mean" if display_f1 and "f1_mean" in trials.columns else "cv_mean"
        trials.sort_values(score, ascending=False if display_f1 else minimize,
                           na_position="last", kind="stable").to_csv(directory / "trials.csv", index=False)
        folds = pd.DataFrame(fold_rows, columns=["model", "trial", "fold", "train_rows", "train", "validation", "gap",
                                                *task_config["metrics"]])
        folds.sort_values("f1" if display_f1 else metric, ascending=False if display_f1 else minimize,
                          na_position="last", kind="stable").to_csv(directory / "folds.csv", index=False)
        complete = trials.loc[trials.status == "ok"].sort_values("cv_mean", ascending=minimize, kind="stable")
        best = complete.drop_duplicates("model")
        best.sort_values(score, ascending=False if display_f1 else minimize,
                         kind="stable").to_csv(directory / "summary.csv", index=False)
        if not best.empty:
            selected_config = copy.deepcopy(config)
            selected_config.pop("diagnosis", None)
            # Reusing this snapshot should train the chosen parameters directly.
            selected_config.setdefault("tuning", {})["enabled"] = False
            for row in best.to_dict("records"):
                selected_config[task]["models"][row["model"]]["params"] = json.loads(row["params"])
            selected_config[task]["model"] = best.iloc[0]["model"]
            with (directory / "best_config.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(selected_config, file, sort_keys=False)
        return best

    for name in names:
        model_settings = task_config["models"][name]
        study = optuna.create_study(direction="minimize" if minimize else "maximize",
                                    sampler=optuna.samplers.TPESampler(seed=config["seed"]),
                                    pruner=optuna.pruners.NopPruner())

        def objective(trial):
            started = time.perf_counter()
            params = {**(model_settings.get("params") or {}),
                      **suggest_parameters(trial, model_settings["search_space"])}
            row = {"task": task, "model": name, "trial": trial.number,
                   "selection_metric": metric, "params": json.dumps(params, sort_keys=True),
                   "sampled_params": json.dumps(trial.params, sort_keys=True),
                   "status": "failed", "error": "", "cv_mean": np.nan, "cv_std": np.nan}
            try:
                pipeline = build_pipeline(task, config, {**model_settings, "type": name, "params": params})
                folds = cross_validate_pipeline(pipeline, X, y, FixedCV(), task, task_config)
                if not np.isfinite(folds[task_config["metrics"]].to_numpy()).all():
                    raise ValueError("Cross-validation returned non-finite metrics")
                row.update(status="ok", cv_mean=float(folds[metric].mean()),
                           cv_std=float(folds[metric].std(ddof=1)),
                           train_validation_gap=float(folds["gap"].mean()),
                           **{f"{key}_mean": float(folds[key].mean()) for key in task_config["metrics"]})
                fold_rows.extend({"model": name, "trial": trial.number, **fold}
                                 for fold in folds.to_dict("records"))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                # Returning NaN marks a failed Optuna trial and lets other trials continue.
            row["seconds"] = time.perf_counter() - started
            trial_rows.append(row)
            checkpoint()
            print(f"{name} trial {trial.number}: " +
                  (f"{metric}={row['cv_mean']:.4f}" if row["status"] == "ok" else row["error"]), flush=True)
            return row["cv_mean"]

        # Sequential trials and a seeded sampler make full-count runs reproducible.
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    table = checkpoint()
    if table.empty:
        raise ValueError(f"All tuning trials failed; inspect {directory / 'trials.csv'}")
    print(f"Best: {table.iloc[0]['model']}, {metric}={table.iloc[0]['cv_mean']:.4f}", flush=True)
    print(f"Reusable config (tuning disabled): {directory / 'best_config.yaml'}")
    return directory, table
