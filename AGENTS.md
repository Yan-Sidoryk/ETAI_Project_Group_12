# Project guide for contributors and agents

## Purpose

ETAI Group 12 is building a terminal-based tool for the wholesale company
Everything & Then Some. Each row is a purchase occasion. The tool predicts:

1. `repeat_purchase_90d`: whether the customer buys again within 90 days.
2. `spend_next_90d`: how much they spend in that period, conditional on returning.

The classifier gates the delivered spend prediction. A negative decision produces
zero spend in the client interface and no row in the regression export. Spend is
not multiplied by the probability of return.

## Read these sources

- `docs/project_guidelines.pdf`: original assignment, read for this guide.
- `docs/guidelines.md`: convenient Markdown summary of the brief.
- `docs/column_descriptions.md`: data dictionary and raw-data observations.
- `README.md`: setup and terminal commands.
- `notebooks/01_data_audit.ipynb`: exploration and cleaning rationale.
- `catch_up.md`: earlier detailed review. Its file map and successful-run checks
  predate the refactor; consult current code for function locations.

The brief requires terminal training per task, saved models, batch exports, a
client interface for new and returning customers, configurable choices, and
Mac/Windows run instructions. It also requires at least five classification
algorithms and four regression algorithms, feature selection, hyperparameter
tuning, global and individual explanations of both models, and an additional
objective or analysis. Deliver the repository, `Group12_ETAI_2627.zip`, and a
matching Kaggle submission by 22 December at 18:00.

## Start reading at main.py

There are exactly four function definitions in `main.py`. They contain the
workflow steps directly, with no nested functions or intermediate workflow
wrappers. Argument parsing is in the `if __name__ == "__main__"` block, which
selects a route and calls `args.run(args)`.

| Route | Read its steps in this order | Result |
| --- | --- | --- |
| `compare(args)` | Load configuration and raw training data; prepare rows; reserve chronological holdout; cross-validate candidates on development data; rank and export | `outputs/comparison_<task>.csv` |
| `train(args)` | Load and prepare rows; reserve holdout; cross-validate selected model and baseline; assess holdout; refit on all usable rows; save | Model artifact, scoreboard row and report |
| `predict(args)` | Load both saved models and test purchases; classify; predict spend; export coherent results | Submission for every test ID and spend for predicted returners |
| `client(args)` | Load models and JSON; validate/normalize one purchase; classify; predict spend only on a positive decision; print | Probability, decision and spend |

The routes are alternatives. They do not call one another. Intended user flow:

```text
compare both tasks -> review results and choose settings in config.yaml
                   -> train both tasks -> predict OR client
```

Comparison is optional and does not update the selected model automatically.
The two training commands can run in either order. Both prediction commands need
both artifacts and never train. Read data preparation directly in `compare` and
`train`: the short repeated sequence intentionally keeps filtering and split
order visible instead of hiding it behind loaders calling more loaders.

## Supporting modules

| Module | Responsibility |
| --- | --- |
| `src/data.py` | Load configuration and cleaning rules; `load_data(config, dataset)` returns raw train/test purchases and customer records |
| `src/cleaning.py` | Deterministic repairs, rule checks and training-only empty-row removal |
| `src/features.py` | `prepare`: clean, join customer attributes and add features; `split_features_target`: select X and y |
| `src/preprocessing.py` | Build learned imputation, scaling and categorical encoding |
| `src/model.py` | Model registry, dummy baselines and one explicit `build_pipeline` function |
| `src/evaluate.py` | Chronological holdout, direct model scoring, CV loop and fold-report formatting |
| `src/inference.py` | Artifact filenames/loading and shared classification/spend policies |
| `src/results.py` | Append scoreboard rows and write reports |
| `tests/test_pipeline.py` | Integration checks for all four routes using temporary artifacts |

`build_pipeline(task, config, model_config)` receives explicit model settings
from the route. There is no separate model-config lookup or estimator wrapper.
It builds the estimator and combines it with preprocessing. `score_model`
scores directly; it does not create nested scorer functions or closures.
`classify` and `predict_spend` remain shared because both inference routes must
use the same preparation, saved feature order, threshold and spend policy.

## Data and assessment

Raw inputs in `data/` are `train.csv` (6,534 rows, 40 columns), `test.csv` (986
rows, 38 columns), `clientes.csv` (1,143 rows, 13 columns), and the sample
submission. Paths are relative to the repository root. `config.yaml` controls
paths, targets, dropped features, preprocessing, validation and model settings.
`diagnosis.yaml` specifies repairs derived from the audit.

Training removes fully empty feature rows before preparation. Preparation learns
nothing from the data: it repairs sentinels/sign flips/inconsistencies, joins
selected customer attributes on `customer_id`, then adds `is_first_purchase` and
`purchase_month`. X excludes identifiers, both targets, `purchase_date` and
`random_noise`. Prediction rows are never dropped.

The most recent 20% of usable classification rows form the holdout. Both tasks
share that date boundary; regression filters unknown spend targets only after
splitting. Current counts are 6,469 total / 5,175 development / 1,294 holdout for
classification and 3,425 / 2,733 / 692 for regression.

Five expanding-window folds assess development data. Each fold fits its own
imputation, scaling and encoding inside the estimator pipeline. The selected
model and a dummy baseline are assessed separately. The holdout score comes from
a development-only fit; the deployment artifact is subsequently refitted on all
usable training rows, including the holdout.

Current selections are random forest for classification (F1, threshold 0.5) and
CatBoost for regression (RMSE, also reporting MAE/R2). Each task has six non-dummy
candidates. The artifact stores pipeline, threshold, feature names, configuration
and run ID. Inference uses each artifact's saved configuration and clips spend
at zero, rounding to cents. An omitted prior count in client JSON is treated as
a first purchase; other omitted fields are unknown.

## Checks and current limits

The 16 November 2026 refactor repaired the incomplete imports and training-data
connections. Against the last committed implementation, feature preparation,
split sizes, and all 14 candidate/baseline fold tables matched using reduced
iteration counts and two folds. Integration checks exercise both training tasks,
both comparisons, batch exports, new/returning client inputs, missing models,
unknown fields, single-row/batch consistency and the negative-decision spend gate.
They write temporary artifacts and do not replace the owner's saved models.

Run checks from the repository root:

```sh
python -m unittest discover -s tests -v
python main.py --help
```

Full comparisons and training with the production iteration counts have not been
rerun after this refactor. Locally the existing artifacts include a classifier
and classification report, but no production regressor. Run both training
commands before production batch/client prediction. An available model entry is
not evidence of completed comparison or tuning.

## Rules to preserve and remaining work

- Preserve in-progress owner edits. This project is being rearranged. Separate
  structural changes from modelling choices and keep the four routes readable.
- Never use targets as features. Features must be available at purchase time.
  Predict each test row using only that row, training information and
  `clientes.csv`; never derive features from other test rows.
- Preserve prediction row count, IDs and order. Check customer-key uniqueness so
  the join cannot multiply purchases.
- Keep learned preprocessing and future feature selection inside CV. Select
  models and thresholds on development data, not repeated holdout scores.
- Review calendar-based validation with a 90-day outcome-window gap and dates
  kept together. Current row-based splits have no gap and can cross equal dates.
  This assessment policy was preserved by the refactor.
- Regression assessment uses raw predictions; inference clips and rounds. Align
  or quantify those policies in future modelling work. Comparison currently
  sorts all regression metrics ascending, which would rank R2 backwards if it
  became the selection metric. Current RMSE sorting is correct.
- Complete and retain evidence for model comparisons, feature selection,
  hyperparameter tuning, both model explanations and the additional objective.
  No systematic tuning or model-explanation implementation was found here.
- Use Python 3.12 and documented uv/pip setup. Keep dependency files consistent.
  Generated artifacts are ignored by Git; preserve final evidence deliberately.
  Verify the documented commands in a clean setup and the Kaggle export against
  the sample format.
- The brief forbids AutoML-like tools and asks for peer-reviewed sources in the
  write-up. Every group member must be able to explain the entire submission.
