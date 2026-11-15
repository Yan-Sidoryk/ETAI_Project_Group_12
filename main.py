"""
Entry point. Run from the repository root:

    python main.py train --task classification
    python main.py train --task regression
    python main.py predict
    python main.py client --input examples/new_customer.json
"""
import argparse
import os
from datetime import datetime

import joblib
from sklearn.base import clone

from src.cleaning import drop_empty_rows
from src.data import load_config, load_raw
from src.evaluate import build_scorer, cross_validate_pipeline, cv_report, make_cv, split_holdout
from src.features import prepare, split_features_target
from src.model import FLOOR, build_pipeline
from src.results import save_run, save_scores

MODEL_FILES = {"classification": "classifier.joblib", "regression": "regressor.joblib"}


def load_training_rows(config: dict):
    """The training file, without its empty rows, cleaned and with every feature added."""
    train_raw, _, clients = load_raw(config)
    not_features = config["id_cols"] + list(config["targets"].values())
    feature_cols = [c for c in train_raw.columns if c not in not_features]
    return prepare(drop_empty_rows(train_raw, feature_cols), clients, config)


def train(args):
    """
    raw data -> cleaned rows -> hold-out set aside -> cross-validation of the no-model floor and
    of the configured model on the development rows -> score on the hold-out -> refit on every
    training row -> saved model + scoreboard row + full report.
    """
    task = args.task
    if task == "regression":
        raise SystemExit("train --task regression is not built yet")

    config = load_config()
    task_config = config[task]
    metrics, selection_metric = task_config["metrics"], task_config["selection_metric"]
    model_type = task_config["model"]["type"]
    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = load_training_rows(config)
    validation = config["validation"]
    dev, holdout = split_holdout(rows, validation["date_col"], validation["holdout_fraction"])
    X_dev, y_dev = split_features_target(dev, config, task)
    X_holdout, y_holdout = split_features_target(holdout, config, task)
    print(f"Task: {task}   |   model: {model_type}   |   selection metric: {selection_metric}")
    print(f"Data: {len(dev)} development rows   |   {len(holdout)} hold-out rows (most recent, "
          f"from {holdout[validation['date_col']].min()})")

    # Every choice is made on the development rows only, by time-ordered cross-validation.
    cv = make_cv(config)
    score = build_scorer(metrics, task_config["threshold"])
    pipeline = build_pipeline(task, config)
    floor_pipeline = build_pipeline(task, config, FLOOR[task])
    floor_scores = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv, score, selection_metric)
    fold_scores = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, score, selection_metric)

    # The hold-out is scored once, by a model fitted on the development rows: the honest estimate.
    holdout_scores = score(clone(pipeline).fit(X_dev, y_dev), X_holdout, y_holdout)
    holdout_floor = score(clone(floor_pipeline).fit(X_dev, y_dev), X_holdout, y_holdout)

    cv_mean, cv_std = fold_scores["validation"].mean(), fold_scores["validation"].std(ddof=1)
    floor_mean = floor_scores["validation"].mean()
    summary = "\n".join([
        f"\nCross-validation ({selection_metric} per fold)",
        f"  floor ({FLOOR[task]['type']}):  " + "  ".join(f"{v:.3f}" for v in floor_scores["validation"])
        + f"   mean {floor_mean:.3f}",
        f"  {model_type}:  " + "  ".join(f"{v:.3f}" for v in fold_scores["validation"])
        + f"   mean {cv_mean:.3f} +/- {cv_std:.3f}   train-validation gap {fold_scores['gap'].mean():+.3f}",
        f"\nHold-out ({len(holdout)} rows)",
        f"  floor:  {selection_metric} = {holdout_floor[selection_metric]:.3f}",
        f"  {model_type}:  " + "   ".join(f"{name} = {value:.3f}" for name, value in holdout_scores.items()),
    ])
    print(summary)

    # The model that is saved learns from every training row, hold-out included.
    X_all, y_all = split_features_target(rows, config, task)
    pipeline.fit(X_all, y_all)
    model_path = os.path.join(config["paths"]["models_dir"], MODEL_FILES[task])
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": task_config["threshold"],
                 "features": list(X_all.columns), "config": config, "run": run}, model_path)

    save_scores(config["outputs"]["scoreboard"], {
        "run": run, "task": task, "model": model_type, "params": str(task_config["model"].get("params")),
        "threshold": task_config["threshold"], "metric": selection_metric, "cv_mean": cv_mean, "cv_std": cv_std,
        "floor": floor_mean, "holdout": holdout_scores[selection_metric],
        **{f"fold_{i}": v for i, v in enumerate(fold_scores["validation"], start=1)},
    })
    report = "\n\n".join([
        f"Run: {run}\nTask: {task}\nModel: {model_type}  params={task_config['model'].get('params')}\n"
        f"Preprocessing: {config['preprocessing']}\nThreshold: {task_config['threshold']}",
        cv_report(floor_scores, metrics, selection_metric, "Floor"),
        cv_report(fold_scores, metrics, selection_metric, "Cross-validation"),
        summary,
        f"Final model: refit on all {len(X_all)} training rows, saved to {model_path}",
    ])
    report_path = save_run(config["paths"]["outputs_dir"], run, report)
    print(f"\nSaved model: {model_path}  (refit on all {len(X_all)} training rows)")
    print(f"Full report: {report_path}")


def predict(args):
    raise SystemExit("predict is not built yet")


def client(args):
    raise SystemExit("client is not built yet")


def main():
    parser = argparse.ArgumentParser(description="Repeat-purchase and 90-day spend pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", help="train one task from the raw data and save the model")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.set_defaults(run=train)

    p = commands.add_parser("predict", help="predict test.csv with the saved models")
    p.set_defaults(run=predict)

    p = commands.add_parser("client", help="predict one new purchase")
    p.add_argument("--input", required=True, help="JSON file with one purchase")
    p.set_defaults(run=client)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
