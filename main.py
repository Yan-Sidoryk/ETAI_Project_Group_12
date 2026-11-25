"""
Entry point. Run from the repository root:

    python main.py experiment --task classification
    python main.py compare --task classification
    python main.py compare --task regression
    python main.py train --task classification
    python main.py train --task regression
    python main.py predict
    python main.py client --input examples/returning_customer.json
    python main.py client --input examples/new_customer.json
"""
import argparse
import json
import os
import time
from datetime import datetime

import joblib
import pandas as pd
from sklearn.base import clone

from src.cleaning import drop_empty_rows
from src.data import load_config, load_data
from src.evaluate import cross_validate_pipeline, cv_report, score_model, split_holdout, TimeCalendarSplit
from src.features import prepare, split_features_target
from src.experiments import run_preprocessing_experiment
from src.inference import MODEL_FILES, classify, load_model, predict_spend
from src.model import FLOOR, build_pipeline
from src.results import save_run, save_scores, selection_report


# Shared by compare and train, so both always judge models on the same rows and folds
def load_training_splits(config: dict, task: str):
    """
    Returns (rows, dev, holdout, cv) for `task`:
      rows     every usable training row, used to refit the saved model
      dev      the development rows, before the hold-out minus the gap
      holdout  the most recent rows, kept out of every choice
      cv       time-ordered folds over `dev`, with the same calendar gap
    The time boundary is set on all rows first; regression then keeps only rows with a known spend.
    """
    # Load and prepare training purchases; never remove rows during inference.
    train_raw, clients = load_data(config, "train")
    excluded = config["id_cols"] + list(config["targets"].values())
    feature_columns = [column for column in train_raw.columns if column not in excluded]
    rows = drop_empty_rows(train_raw, feature_columns)
    rows = prepare(rows, clients, config)

    validation = config["validation"]
    dev, holdout = split_holdout(rows, validation["date_col"], validation["holdout_fraction"], validation["gap_days"])
    if task == "regression":
        target = config["targets"]["regression"]
        rows = rows.loc[rows[target].notna()]
        dev = dev.loc[dev[target].notna()]
        holdout = holdout.loc[holdout[target].notna()]

    cv = TimeCalendarSplit(dev[validation["date_col"]], n_splits=validation["n_splits"], gap_days=validation["gap_days"])
    return rows, dev, holdout, cv


# Route 1: compare candidates on development data
def compare(args):
    """
    Cross-validates the floor and every model in config[task]["models"] on the development rows,
    with the same time-ordered folds, and prints them from best to worst. The hold-out is not
    touched, so this table can be used to choose config[task]["model"].
    """
    task = args.task
    config = load_config(getattr(args, "config", "config.yaml"))
    task_config = config[task]
    metrics, selection_metric = task_config["metrics"], task_config["selection_metric"]

    _, dev, _, cv = load_training_splits(config, task)
    X_dev, y_dev = split_features_target(dev, config, task)

    candidates = {"floor": FLOOR[task]}
    for name, settings in task_config["models"].items():
        candidates[name] = {"type": name, **settings}
    table = []
    for name, model_config in candidates.items():
        start = time.perf_counter()
        pipeline = build_pipeline(task, config, model_config)
        folds = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, task, task_config)
        table.append({"model": name, **folds[metrics].mean(), f"{selection_metric}_std": folds["validation"].std(ddof=1),
                      "train_validation_gap": folds["gap"].mean(), "seconds": time.perf_counter() - start})
        print(f"  {name}: {selection_metric} = {folds['validation'].mean():.3f}")
    lower_is_better = task == "regression"            # errors in EUR; classification metrics are scores
    table = pd.DataFrame(table)
    table = table.sort_values(selection_metric, ascending=lower_is_better)

    path = os.path.join(config["paths"]["outputs_dir"], f"comparison_{task}.csv")
    os.makedirs(config["paths"]["outputs_dir"], exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\n{task}: mean over {cv.get_n_splits()} time-ordered folds of {len(dev)} development rows, best first\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {path}   |   `train` saves the model named by {task}.model in config.yaml "
          f"(now: {task_config['model']})")


def experiment(args):
    """Search preprocessing on development folds, leaving the holdout unscored."""
    config = load_config(getattr(args, "config", "config.yaml"))
    _, dev, _, cv = load_training_splits(config, args.task)
    X, y = split_features_target(dev, config, args.task)
    _, table = run_preprocessing_experiment(config, args.task, X, y, cv, getattr(args, "models", None))
    if not (table["status"] == "ok").any():
        raise SystemExit("No successful preprocessing candidates; inspect the saved errors.")


# Route 2: evaluate, refit and save one selected model
def train(args):
    """
    raw data -> cleaned rows -> hold-out set aside -> cross-validation of the no-model floor and
    of the configured model on the development rows -> score on the hold-out -> refit on every
    training row -> saved model + scoreboard row + full report.
    """
    task = args.task
    config = load_config(getattr(args, "config", "config.yaml"))
    task_config = config[task]
    metrics = task_config["metrics"]
    selection_metric = task_config["selection_metric"]
    model_type = task_config["model"]
    if model_type not in task_config["models"]:
        raise SystemExit(f"config.yaml: {task}.model is '{model_type}', which is not in {task}.models. "
                         f"Options: {list(task_config['models'])}")
    model_config = {"type": model_type, **task_config["models"][model_type]}
    model_config = {"type": model_type, **task_config["models"][model_type]} 
    model_params = model_config.get("params")
    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows, dev, holdout, cv = load_training_splits(config, task)
    validation = config["validation"]
    X_dev, y_dev = split_features_target(dev, config, task)
    X_holdout, y_holdout = split_features_target(holdout, config, task)
    print(f"Task: {task}   |   model: {model_type}   |   selection metric: {selection_metric}")
    print(f"Data: {len(dev)} development rows   |   {len(holdout)} hold-out rows (most recent, "
          f"from {holdout[validation['date_col']].min()})")

    # Every choice is made on the development rows only, by time-ordered cross-validation.
    threshold = task_config.get("threshold")                    # classification only
    pipeline = build_pipeline(task, config, model_config)
    floor_pipeline = build_pipeline(task, config, FLOOR[task])
    floor_scores = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv, task, task_config)
    fold_scores = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, task, task_config)

    # The hold-out is scored once, by a model fitted on the development rows: the honest estimate.
    holdout_model = clone(pipeline).fit(X_dev, y_dev)
    holdout_baseline = clone(floor_pipeline).fit(X_dev, y_dev)
    holdout_scores = score_model(holdout_model, X_holdout, y_holdout, task, task_config)
    holdout_floor = score_model(holdout_baseline, X_holdout, y_holdout, task, task_config)

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
    joblib.dump({"pipeline": pipeline, "threshold": threshold,
                 "features": list(X_all.columns), "config": config, "run": run}, model_path)

    save_scores(config["outputs"]["scoreboard"], {
        "run": run, "task": task, "model": model_type, "params": str(model_params),
        "threshold": threshold, "metric": selection_metric, "cv_mean": cv_mean, "cv_std": cv_std,
        "floor": floor_mean, "holdout": holdout_scores[selection_metric],
        **{f"fold_{i}": v for i, v in enumerate(fold_scores["validation"], start=1)},
    })
    report = "\n\n".join([
        f"Run: {run}\nTask: {task}\nModel: {model_type}  params={model_params}\n"
        f"Preprocessing: {config['preprocessing']}\nThreshold: {threshold}",
        cv_report(floor_scores, metrics, selection_metric, "Floor"),
        cv_report(fold_scores, metrics, selection_metric, "Cross-validation"),
        summary,
        f"Final model: refit on all {len(X_all)} training rows, saved to {model_path}",
        selection_report(fold_scores, pipeline),
    ])
    report_path = save_run(config["paths"]["outputs_dir"], run, report)
    if "select" in pipeline.named_steps:
        print("\n" + selection_report(fold_scores, pipeline).split("\n\n")[0])
    print(f"\nSaved model: {model_path}  (refit on all {len(X_all)} training rows)")
    print(f"Full report: {report_path}")

# Route 3: batch predictions for the test file
def predict(args):
    """
    test.csv, with the two saved models ->
      outputs/submission.csv              ID, repeat_purchase_90d                  every test row
      outputs/regression_predictions.csv  ID, repeat_purchase_90d, spend_next_90d  rows predicted to return
    The two files cannot disagree: a spend is written only where the classifier says 1.
    """
    config = load_config()
    classifier = load_model(config, "classification")
    regressor = load_model(config, "regression")
    test, clients = load_data(config, "test")
    clf_target, reg_target = config["targets"]["classification"], config["targets"]["regression"]

    _, decision = classify(classifier, test, clients)
    submission = test[["ID"]].assign(**{clf_target: decision})
    spend = predict_spend(regressor, test, clients)
    returners = submission[decision == 1].assign(**{reg_target: spend[decision == 1]})

    outputs = config["outputs"]
    os.makedirs(os.path.dirname(outputs["submission"]) or ".", exist_ok=True)
    submission.to_csv(outputs["submission"], index=False)
    returners.to_csv(outputs["regression_predictions"], index=False)
    print(f"Classifier: {classifier['config']['classification']['model']} "
          f"(run {classifier['run']}, threshold {classifier['threshold']})")
    print(f"Regressor:  {regressor['config']['regression']['model']} (run {regressor['run']})")
    print(f"Kaggle submission:      {outputs['submission']}  "
          f"({len(submission)} rows, {decision.mean():.1%} predicted to return)")
    print(f"Regression predictions: {outputs['regression_predictions']}  "
          f"({len(returners)} rows, median spend {returners[reg_target].median():,.2f} EUR)")


# Route 4: prediction for one purchase
def client(args):
    """Read one purchase from JSON, predict with saved models, and print the result."""
    config = load_config()
    classifier = load_model(config, "classification")
    regressor = load_model(config, "regression")
    test, clients = load_data(config, "test")
    with open(args.input, encoding="utf-8") as file:
        record = json.load(file)

    # Validate fields and use the test schema for names and data types.
    unknown = sorted(set(record) - set(test.columns))
    if unknown:
        raise SystemExit(f"Unknown fields: {unknown}. Allowed fields: {list(test.columns)}")
    record = {name: value for name, value in record.items() if value is not None}

    # Preserve the existing policy: an omitted prior count means a first purchase.
    if "prior_purchase_count" not in record:
        history_columns = [column for column in test.columns
                           if column.startswith(("purchase_count_", "spend_"))]
        history_columns += ["prior_purchase_count", "prior_total_spend", "customer_tenure_days"]
        record = {**dict.fromkeys(history_columns, 0), **record}
    dtypes = {name: float if pd.api.types.is_numeric_dtype(dtype) else dtype
              for name, dtype in test.dtypes.items()}
    purchase = pd.DataFrame([record]).reindex(columns=test.columns).astype(dtypes)

    # The classifier gates spend: a negative decision always means zero EUR.
    probability, decision = classify(classifier, purchase, clients)
    spend = 0.0
    if decision[0] == 1:
        spend = predict_spend(regressor, purchase, clients)[0]

    print(f"Probability of buying again within 90 days: {probability[0]:.1%}")
    print(f"Decision: {'returns' if decision[0] else 'does not return'} ({decision[0]})")
    print(f"Expected spend in the next 90 days: {spend:,.2f} EUR")


def main():
    """Parse the command line and run the chosen route."""
    parser = argparse.ArgumentParser(description="Repeat-purchase and 90-day spend pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("compare", help="cross-validate every configured model of one task")
    p.add_argument("--config", default="config.yaml", help="YAML configuration path")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.set_defaults(run=compare)

    p = commands.add_parser("experiment", help="compare YAML preprocessing candidates on development folds")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.add_argument("--config", default="config.yaml", help="YAML configuration path")
    p.add_argument("--models", nargs="+", help="configured model names; defaults to the task's selected model")
    p.set_defaults(run=experiment)

    p = commands.add_parser("train", help="train one task from the raw data and save the model")
    p.add_argument("--config", default="config.yaml", help="YAML configuration path")
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
