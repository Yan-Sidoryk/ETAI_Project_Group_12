"""Compare preprocessing choices using development folds only; preserve every run."""
import copy
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.evaluate import cross_validate_pipeline
from src.model import build_pipeline, task_preprocessing


_COMPONENTS = ("scaler", "numeric_imputer", "categorical_imputer", "encoder")


def preprocessing_candidates(config, task=None):
    """Include the current settings (the task's, when `task` is given) and all combinations from the grid."""
    experiment = config.get("preprocessing_experiment", {})
    if experiment.get("mode", "grid") != "grid":
        raise ValueError("Only grid search is supported; remove preprocessing_experiment.mode")
    candidates = experiment.get("candidates", {})
    if not isinstance(candidates, dict) or set(candidates) - set(_COMPONENTS):
        raise ValueError(f"preprocessing_experiment.candidates must map these settings to lists: {_COMPONENTS}")
    base = task_preprocessing(config, task) if task else config["preprocessing"]
    for key, values in candidates.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"preprocessing_experiment.candidates.{key} must be a non-empty list")
    variations = [base]
    keys = list(candidates)
    variations.extend({**base, **dict(zip(keys, values))}
                      for values in product(*(candidates[key] for key in keys)))
    unique = {}
    for settings in variations:
        unique.setdefault(json.dumps(settings, sort_keys=True), copy.deepcopy(settings))
    return list(unique.values())


def run_preprocessing_experiment(config, task, X, y, cv, models=None):
    """Save ranked summary and fold scores without generating configuration files.

    Invalid combinations are recorded as failures and excluded from ranking. No
    estimator is saved, and neither holdout nor test rows are accepted here.
    """
    task_config = config[task]
    models = list(dict.fromkeys(models if models is not None else [task_config["model"]]))
    if not models or any(name not in task_config["models"] for name in models):
        raise ValueError(f"Choose models from {list(task_config['models'])}")
    settings_list = preprocessing_candidates(config, task)
    metric = task_config["selection_metric"]
    if metric not in task_config["metrics"]:
        raise ValueError("selection_metric must be included in metrics")
    # Freeze the split indices once so every candidate uses identical rows.
    splits = list(cv.split(X))
    if len(splits) < 2 or any(not len(train) or not len(val) for train, val in splits):
        raise ValueError("Experiments require at least two non-empty training/validation folds")

    class FixedCV:
        def split(self, X):
            return iter(splits)

    directory = Path(config["paths"]["outputs_dir"]) / (
        f"preprocessing_{task}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    directory.mkdir(parents=True, exist_ok=False)
    latest_summary = directory.parent / f"preprocessing_{task}_summary.csv"
    rows, fold_rows = [], []
    total = len(models) * len(settings_list)
    print(f"{task}: {total} candidates x {len(splits)} folds; metric: {metric}", flush=True)
    print(f"Results: {directory}", flush=True)
    print(f"Latest summary (updated after each candidate): {latest_summary}", flush=True)
    for name in models:
        for settings in settings_list:
            candidate_id = len(rows) + 1
            candidate_config = copy.deepcopy(config)
            candidate_config["preprocessing"] = settings
            candidate_config[task].pop("preprocessing", None)  # the candidate is complete; no task override
            model_config = {"type": name, **task_config["models"][name]}
            identifiers = {"candidate": candidate_id, "task": task, "model": name}
            for key, default in zip(_COMPONENTS, ("standard", "median", "constant", "onehot")):
                value = settings.get(key, default)
                identifiers[key] = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
            row = {
                **identifiers,
                "selection_metric": metric, "cv_mean": np.nan, "cv_std": np.nan,
                "train_validation_gap": np.nan, "status": "failed", "error": "",
            }
            started = time.perf_counter()
            try:
                pipeline = build_pipeline(task, candidate_config, model_config)
                if "select" in pipeline.named_steps:
                    # Same feature selection for every candidate: the selector's probe keeps the current
                    # preprocessing, so only the model's own "prep" step varies and each fold's selection is
                    # computed once and cached.
                    pipeline.set_params(select__preprocessing=task_preprocessing(config, task))
                folds = cross_validate_pipeline(pipeline, X, y, FixedCV(), task, task_config)
                if not np.isfinite(folds[task_config["metrics"]].to_numpy()).all():
                    raise ValueError("Cross-validation returned non-finite metrics")
                row.update(status="ok", cv_mean=folds[metric].mean(), cv_std=folds[metric].std(ddof=1),
                           train_validation_gap=folds["gap"].mean(),
                           **{f"{key}_mean": folds[key].mean() for key in task_config["metrics"]},
                           **{f"{key}_std": folds[key].std(ddof=1) for key in task_config["metrics"]})
                fold_rows.extend({**identifiers, **fold}
                                 for fold in folds.to_dict("records"))
            except Exception as exc:
                # A valid experiment can contain unsupported combinations (e.g. NaNs with Ridge).
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - started
            rows.append(row)
            print(f"[{candidate_id}/{total}] {name}: " +
                  (f"{metric}={row['cv_mean']:.4f}" if row["status"] == "ok" else row["error"]), flush=True)
            # Checkpoint after each candidate so a long run retains completed results.
            table = pd.DataFrame(rows).sort_values("cv_mean", ascending=metric in ("rmse", "mae"),
                                                   na_position="last", kind="stable")
            table.to_csv(directory / "summary.csv", index=False)
            table.to_csv(latest_summary, index=False)
            fold_table = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame(
                columns=["candidate", "task", "model", *_COMPONENTS, "fold"])
            fold_table.to_csv(directory / "folds.csv", index=False)

    successful = table.loc[table["status"] == "ok"]
    if successful.empty:
        print(f"All candidates failed. See {directory / 'summary.csv'}")
    else:
        best = successful.iloc[0]
        print(f"Best result: {metric}={best['cv_mean']:.4f}")
        print(f"Best candidate: {best['candidate']} ({best['model']})")
        print("Preprocessing: " + ", ".join(f"{key}={best[key]}" for key in _COMPONENTS))
    return directory, table
