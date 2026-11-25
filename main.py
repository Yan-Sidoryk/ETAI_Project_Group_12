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
import warnings
from datetime import datetime

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from src.cleaning import drop_empty_rows
from src.data import load_config, load_data
from src.explain import global_importance, model_inputs, shap_values, top_factors
from src.evaluate import (best_threshold, cross_validate_pipeline, cv_report, score_model, split_holdout,
                          TimeCalendarSplit, corrected_paired_ttest, fold_f1_at, validation_predictions)
from src.features import prepare, split_features_target
from src.experiments import run_preprocessing_experiment
from src.inference import MODEL_FILES, classify, load_model, predict_spend
from src.model import FLOOR, build_pipeline, task_preprocessing
from src.results import save_run, save_scores, selection_report
from src.tuning import tune


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
    table, per_fold, sizes = [], {}, None
    for name, model_config in candidates.items():
        start = time.perf_counter()
        pipeline = build_pipeline(task, config, model_config)
        folds = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, task, task_config)
        row = {"model": name, **folds[metrics].mean(), f"{selection_metric}_std": folds["validation"].std(ddof=1),
               "train_validation_gap": folds["gap"].mean(), "seconds": time.perf_counter() - start}
        if task == "classification":  # one threshold for all folds, chosen on their pooled validation rows
            row["best_threshold"], row["f1_at_best"] = best_threshold(*validation_predictions(folds))
            per_fold[name] = fold_f1_at(folds, row["best_threshold"])
        else:
            per_fold[name] = folds["validation"].to_numpy()
        sizes = folds["train_rows"].mean(), folds["val_rows"].mean()   # identical folds for every model
        table.append(row)
        print(f"  {name}: {selection_metric} = {folds['validation'].mean():.3f}")
    lower_is_better = task == "regression"            # errors in EUR; classification metrics are scores
    # Classification is ranked by F1 at each model's own best threshold, regression by RMSE.
    rank_by = "f1_at_best" if task == "classification" else selection_metric
    table = pd.DataFrame(table).sort_values(rank_by, ascending=lower_is_better).reset_index(drop=True)
    # Each model against the best one, fold by fold: difference and corrected paired t-test p-value.
    best = table["model"].iloc[0]
    tests = [corrected_paired_ttest(per_fold[name], per_fold[best], *sizes) for name in table["model"]]
    table["diff_vs_best"] = [per_fold[name].mean() - per_fold[best].mean() for name in table["model"]]
    table["p_vs_best"] = [p for _, p in tests]

    path = os.path.join(config["paths"]["outputs_dir"], f"comparison_{task}.csv")
    os.makedirs(config["paths"]["outputs_dir"], exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\n{task}: mean over {cv.get_n_splits()} time-ordered folds of {len(dev)} development rows, best first\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nRanked by {rank_by}. p_vs_best: corrected paired t-test (Nadeau & Bengio) against {best} on the same "
          f"{len(per_fold[best])} folds; p < 0.05 means the difference is unlikely to be fold noise.")
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


def tune_command(args):
    """Tune one model's hyperparameters on the development folds; prints the best params to copy into config.yaml."""
    config = load_config(getattr(args, "config", "config.yaml"))
    task = args.task
    model = args.model or config[task]["model"]
    if model not in config["tuning"]["spaces"] or model not in config[task]["models"]:
        raise SystemExit(f"No tuning space or model config for '{model}'. Spaces: {list(config['tuning']['spaces'])}")
    trials = args.trials or config["tuning"]["trials"]
    _, dev, _, cv = load_training_splits(config, task)
    X_dev, y_dev = split_features_target(dev, config, task)
    print(f"Tuning {task} / {model}: {trials} trials x {cv.get_n_splits()} folds of {len(dev)} development rows")
    table, best_params, configured = tune(config, task, model, X_dev, y_dev, cv, trials)

    print(f"\nTop 5 of {len(table)} trials:")
    print(table.head(5).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    score = table.columns[1]
    print(f"\nBest trial {int(table['number'].iloc[0])}: {score} {table[score].iloc[0]:.4f}   "
          f"configured params: {configured:.4f}   (best of {len(table)} trials is slightly optimistic)")
    params = {**(config[task]["models"][model].get("params") or {}), **best_params}
    print(f"Params to copy into config.yaml {task}.models.{model}.params: {params}")


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
    if task == "classification":
        # Threshold diagnostics: the best threshold per fold (drift over time), one threshold chosen on all
        # validation rows (slightly optimistic: picked and scored on the same rows), and how that CV choice
        # does on the later hold-out. Reported only; the threshold used stays classification.threshold.
        cv_threshold, cv_f1 = best_threshold(*validation_predictions(fold_scores))
        holdout_probability = holdout_model.predict_proba(X_holdout)[:, 1]
        holdout_f1 = f1_score(y_holdout, holdout_probability >= cv_threshold, zero_division=0)
        summary += "\n".join([
            f"\n\nThreshold (configured: {threshold})",
            "  best per fold:  " + "  ".join(f"{t:.2f}" for t in fold_scores["best_threshold"])
            + "   F1 there: " + "  ".join(f"{v:.3f}" for v in fold_scores["f1_at_best"]),
            f"  best on all validation rows: {cv_threshold:.2f}   CV F1 {cv_f1:.3f} (vs {cv_mean:.3f} at {threshold})",
            f"  hold-out F1 at {cv_threshold:.2f}: {holdout_f1:.3f} (vs {holdout_scores['f1']:.3f} at {threshold})",
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
        f"Preprocessing: {task_preprocessing(config, task)}\nThreshold: {threshold}",
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
def explain(args):
    """Global SHAP explanation of one saved model on the test purchases: ranked CSV and beeswarm plot."""
    import matplotlib
    matplotlib.use("Agg")  # write files only, never open a window
    import matplotlib.pyplot as plt
    import shap

    config = load_config()
    task = args.task
    model = load_model(config, task)
    test, clients = load_data(config, "test")
    X = model_inputs(model, test, clients, task)
    print(f"Explaining the saved {task} model on {len(X)} test purchases...")
    raw, base, selected = shap_values(model, X, task)
    table = global_importance(raw, selected)

    outputs = config["paths"]["outputs_dir"]
    os.makedirs(outputs, exist_ok=True)
    csv_path = os.path.join(outputs, f"shap_{task}_global.csv")
    table.to_csv(csv_path)
    numeric = selected[raw.columns].apply(pd.to_numeric, errors="coerce")  # categoricals: no colour scale
    with warnings.catch_warnings():  # categorical columns are all-NaN here, which numpy warns about
        warnings.simplefilter("ignore", RuntimeWarning)
        shap.summary_plot(raw.to_numpy(), numeric, feature_names=list(raw.columns), max_display=15, show=False)
    unit = "probability of returning" if task == "classification" else "spend in EUR"
    plt.title(f"SHAP values, {task} ({unit}); average prediction {base:,.3g}")
    plot_path = os.path.join(outputs, f"shap_{task}_beeswarm.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nMean |SHAP| per feature (effect on the {unit}); direction: + means a higher value raises it\n")
    print(table.head(15).to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved: {csv_path}\nSaved: {plot_path}")


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

    # Why: the features that moved this purchase's predictions most, from the average prediction (SHAP).
    explained = [("classification", classifier, "probability", lambda v: f"{v * 100:+.1f} pp")]
    if decision[0] == 1:
        explained.append(("regression", regressor, "spend", lambda v: f"{v:+,.0f} EUR"))
    for task, model, label, fmt in explained:
        X = model_inputs(model, purchase, clients, task)
        raw, base, selected = shap_values(model, X, task)
        start = f"{base:.1%}" if task == "classification" else f"{base:,.0f} EUR"
        print(f"\nWhy this {label} (average {start}; main contributions):")
        for name, value, contribution in top_factors(raw.iloc[0], selected.iloc[0]):
            shown = ("missing" if pd.isna(value) else value if not isinstance(value, float)
                     else f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3g}")
            print(f"  {name} = {shown}: {fmt(contribution)}")


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

    p = commands.add_parser("tune", help="tune one model's hyperparameters with Optuna on development folds")
    p.add_argument("--config", default="config.yaml", help="YAML configuration path")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.add_argument("--model", help="configured model name; defaults to the task's selected model")
    p.add_argument("--trials", type=int, help="number of trials; defaults to tuning.trials")
    p.set_defaults(run=tune_command)

    p = commands.add_parser("train", help="train one task from the raw data and save the model")
    p.add_argument("--config", default="config.yaml", help="YAML configuration path")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.set_defaults(run=train)

    p = commands.add_parser("predict", help="predict test.csv with the saved models")
    p.set_defaults(run=predict)

    p = commands.add_parser("explain", help="global SHAP explanation of a saved model on test.csv")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.set_defaults(run=explain)

    p = commands.add_parser("client", help="predict one new purchase")
    p.add_argument("--input", required=True, help="JSON file with one purchase")
    p.set_defaults(run=client)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
