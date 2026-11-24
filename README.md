# ETAI Project - Group 12

"Everything & Then Some": predict whether a customer buys again within 90 days, and how much they spend.
See [docs/guidelines.md](docs/guidelines.md) for the full brief.

## Structure

```
main.py         entry point: compare | experiment | train | predict | client
config.yaml     paths, seed, validation, preprocessing, models, hyperparameters, outputs
diagnosis.yaml  data-cleaning rules (ranges, placeholder values, consistency checks and their fixes)
src/
  data.py           load config.yaml + diagnosis.yaml, read the CSVs
  cleaning.py       rule checks and repairs from diagnosis.yaml; training-only empty-row removal
  features.py       prepare(): clean, join clientes.csv, add_features(); split X and y
  preprocessing.py  configurable imputation, scaling, encoding (fitted inside the model pipeline)
  model.py          model registry per task, baseline ("floor") models, build_pipeline()
  evaluate.py       time-ordered hold-out and CV folds with a calendar gap, metrics, CV reports
  experiments.py    preprocessing grid search, summaries, fold scores
  inference.py      load saved models; shared classify / predict_spend for predict and client
  results.py        scoreboard and run reports
tests/          preprocessing, experiments, and command integration tests
examples/       client inputs: returning_customer.json, new_customer.json
data/           raw input CSVs (train, test, clientes, sample_submission)
docs/           project brief (PDF + guidelines.md) and column descriptions
notebooks/      data audit (exploration and cleaning rationale)
models/         saved models (generated, git-ignored)
outputs/        submissions, predictions, run reports (generated, git-ignored)
```

## Setup

Requires Python 3.12.

### With uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root (same on Mac and Windows):
```bash
uv sync                 # creates .venv with Python 3.12 and the locked dependencies
uv run jupyter lab      # or: uv run python <script>
```
Add a package with `uv add <pkg>==<version>`, and add the same pin to `requirements.txt` so pip users get it too.

### With pip

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```



## Running the pipeline

From the repo root, with the environment active (or prefix each command with `uv run`). The commands are the same on
Mac and Windows. Saved models are not in the repository, so run the two `train` commands first, in either order:
```bash
python main.py compare --task classification # optional: cross-validates every model in config.yaml
python main.py compare --task regression
python main.py train --task classification   # saves models/classifier.joblib
python main.py train --task regression       # saves models/regressor.joblib
python main.py predict                       # needs both models
python main.py client --input examples/returning_customer.json   # one purchase, needs both models
```

`predict` writes two files:
- `outputs/submission.csv`: the Kaggle file, one row per test purchase with `repeat_purchase_90d`.
- `outputs/regression_predictions.csv`: predicted `spend_next_90d` for the purchases predicted to return.

`client` reads one purchase from a JSON file whose fields are the columns of `data/test.csv`, and prints the probability
of buying again, the decision and the expected 90-day spend (0 when the decision is "does not return").
Any field can be left out; for a new customer, leave out the history fields and `customer_id`,
as in `examples/new_customer.json`.

`compare` prints the task's table, best model first, and saves it to `outputs/comparison_<task>.csv`.
`train` saves the model named by `classification.model` / `regression.model` in `config.yaml`.

Each `train` run prints its cross-validation and hold-out scores against the floor model, adds a row to `outputs/runs.csv`
and writes the full report to `outputs/run_<id>.txt`.
Models, metrics and the threshold are set in `config.yaml`; no code edits are needed.
Runs are reproducible: the seed is `seed` in `config.yaml`, and two runs from the raw data give identical
`submission.csv` and `regression_predictions.csv` (checked on Windows, 17 Nov 2026).

Integration tests (temporary models, quick settings; they never overwrite `models/` or `outputs/`):
```bash
python -m unittest discover -s tests -v
```

## Project planning

**Deadline:** 22 December 2026, 18:00. Early delivery earns +0.15 per day (max +1); each day late costs 1 point.
**Deliverables:** this repository, `Group12_ETAI_2627.zip`, and a final Kaggle submission produced by `predict`.
Grading and rules are in [docs/guidelines.md](docs/guidelines.md).

### How the pipeline works

Training, batch prediction and the client all prepare a purchase through the same `prepare()`:

```
raw rows ──> clean (diagnosis.yaml) ──> join clientes.csv ──> add_features ──> X ──> Pipeline(prep, model)
└──────────── prepare(): row by row, learns nothing ─────────────────────┘     └ fitted, saved with joblib ┘
```

- `prepare()` learns nothing from the data, so a test purchase uses only itself, the training data and `clientes.csv`
  (brief rule 1). Everything learned (imputation, encoding, scaling, later feature selection) sits inside the
  scikit-learn `Pipeline`, so it is refitted in every CV fold and saved with the model.
- **Hold-out:** the most recent 20% of training rows (from 17 Jun 2023), never used for a choice. Development rows
  end `validation.gap_days` (90) days before it, because a purchase's label looks 90 days ahead.
- **Cross-validation:** 5 time-ordered folds on the development rows, with the same 90-day gap before each validation
  block. Test purchases are all later than training, so shuffled folds would leak the future.
- **Saved model:** refitted on all usable training rows (hold-out included) after assessment.
- **Classification:** selected on F1, the Kaggle metric, at `classification.threshold` (0.5).
- **Regression:** trained only on returners with an observed spend (3,425 rows; 229 returners have none, so a
  missing spend is not zero). Log target is off (`log_target: false`): on the tree models it lowers MAE but raises RMSE.
- **Coherence:** spend is predicted only where the classifier says "returns"; the regression file holds exactly
  those IDs and the client shows 0 EUR otherwise.
- **Not used as features:** IDs, `purchase_date` (splitting only), `random_noise`, both targets, and the
  `clientes.csv` columns other than `country_or_region`, `commercial_region` and `customer_since_year`
  (mostly missing or almost unique per customer).

### Choosing preprocessing

Edit `preprocessing` in `config.yaml`. Each component accepts a name or
`{type: name, params: {...}}`, where `params` contains scikit-learn constructor parameters.
The current choices preserve the original preprocessing:

```yaml
preprocessing:
  scaler: standard
  numeric_imputer: median
  categorical_imputer: constant
  encoder: onehot
  min_frequency: 20
  categorical_features: # keep the existing list of eight columns
    - payment_type
    - salesperson
    - commercial_zone
    - transport_method
    - warehouse
    - document_series
    - country_or_region
    - commercial_region
```

| Setting | Supported names | Defaults / behavior |
| --- | --- | --- |
| `scaler` | `standard`, `minmax`, `robust`, `maxabs`, `normalize`, `power`, `quantile`, `none` | Applied after numeric imputation; `normalize` normalizes each row, `power` defaults to Yeo-Johnson, `quantile` to a uniform distribution with random seed 42 |
| `numeric_imputer` | `median`, `mean`, `most_frequent`, `constant`, `knn`, `none` | Missingness indicators enabled except for `none`; `constant` fills with 0 |
| `categorical_imputer` | `constant`, `most_frequent`, `none` | `constant` fills with `"missing"` |
| `encoder` | `onehot`, `ordinal` | Dense one-hot output ignores unseen levels; ordinal output uses -1 for unseen levels |

For example, replace the corresponding settings with:

```yaml
  scaler: {type: robust, params: {quantile_range: [10, 90]}}
  numeric_imputer: {type: knn, params: {n_neighbors: 5, add_indicator: false}}
  categorical_imputer: most_frequent
  encoder: {type: ordinal, params: {min_frequency: 10}}
```

`min_frequency` applies to both encoders; `encoder.params.min_frequency` overrides it
(use `null` to disable grouping). Older configs without imputer or encoder settings keep
the original defaults. `none` skips that step and does not accept parameters. Skipping
imputation leaves missing values for later steps and the estimator to handle.

Ordinal encoding assigns numeric codes that introduce an ordering between categories.
KNN imputation measures distances before scaling in this pipeline, so feature units affect
its neighbors; it can also be slower on large datasets. Power/quantile transforms also
operate on any appended numeric missingness indicators; set `add_indicator: false` if
you want to avoid that. Transformer parameter constraints still apply (for example,
Box-Cox requires positive inputs).

Run `compare` again to assess each configuration on the development folds, then `train`
to save a model using those choices. Saved models retain their fitted preprocessing;
editing YAML alone does not change their predictions. See the
[scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html),
[KNNImputer reference](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html),
and [OrdinalEncoder reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
for parameter details.

### Comparing preprocessing automatically

Run the new experiment command from the repository root:

```bash
python main.py experiment --task classification
python main.py experiment --task regression
```

By default each task uses its selected model (`classification.model` or `regression.model`)
and the candidate lists in `preprocessing_experiment` in YAML:

```yaml
preprocessing_experiment:
  candidates:
    scaler: [none, standard, robust]
    numeric_imputer: [median, mean]
    categorical_imputer: [constant, most_frequent]
    encoder: [onehot, ordinal]
```

This grid has 24 combinations and 120 fits per model with five folds. Each component
may also contain parameter mappings, such as `{type: knn, params: {n_neighbors: 3}}`.
Omitted components retain their current `preprocessing` settings. The current full
configuration is always included as a reference, with exact duplicate settings removed.

Grid search evaluates every combination. To compare several models on the same grid:

```bash
python main.py experiment --task classification --models logistic_regression xgboost
python main.py experiment --task regression --models ridge catboost
```

Every candidate uses identical development folds and the existing calendar gap, with
fresh preprocessing fitted inside each fold. The holdout is set aside and never scored
by `experiment`; existing model artifacts are not overwritten.

Each invocation creates `outputs/preprocessing_<task>_<timestamp>/` containing:

- `summary.csv`: ranked results with `candidate`, `task`, `model`, `scaler`,
  `numeric_imputer`, `categorical_imputer`, and `encoder` columns,
  metric means/stds, training-validation gap, runtime, and failure details.
- `folds.csv`: metrics for each successful combination's individual folds, linked by
  `candidate` and carrying the same model and preprocessing columns.

Simple preprocessing choices appear as names; choices with custom parameters appear as
JSON in the corresponding component column. Model hyperparameters and shared preprocessing
settings remain in your existing `config.yaml`. Experiments generate CSV files only.

For a fixed CSV path that updates on every run, open
`outputs/preprocessing_classification_summary.csv` or
`outputs/preprocessing_regression_summary.csv`. This latest summary is overwritten after
each completed candidate. The timestamped folders retain their own results unchanged.

Classification F1 is ranked highest first; regression RMSE is ranked lowest first. Failed
combinations are recorded at the bottom and excluded from selection. Results are saved after
each candidate, and separate experiment folders preserve earlier comparisons. If all candidates
fail, the command exits with an error and retains diagnostics.

Use the CV mean, variation between folds, and runtime to assess the winner. After making your
choice, update the four `preprocessing` choices and the task's `model` in your existing
`config.yaml` from the corresponding columns in the selected row. Then evaluate it on the holdout
and save the trained model:

```bash
python main.py train --task classification
python main.py train --task regression
```

Set the chosen preprocessing before training each task if their winning settings differ.
Each saved model retains its own fitted preprocessing.

### Status (17 Nov 2026)

Done: cleaning, features, gapped splits, 6 algorithms per task (logistic/ridge, random forest, hist gradient boosting,
XGBoost, LightGBM, CatBoost), all four commands, integration tests, reproducibility on Windows.

| Task | Model | CV (5 folds) | Floor | Hold-out |
|---|---|---|---|---|
| Classification (F1) | XGBoost | 0.725 ± 0.023 | 0.712 | 0.771 (floor 0.753), ROC-AUC 0.809 |
| Regression (RMSE, EUR) | CatBoost | 2,047 ± 200 | 2,472 | 2,483 (floor 3,289), MAE 1,288, R² 0.42 |

`compare --task classification` (CV F1): random forest 0.741, CatBoost 0.726, XGBoost 0.725, hist gradient boosting 0.719,
LightGBM 0.716, floor 0.712, logistic regression 0.709. XGBoost stays selected for now; random forest is the
current CV winner.

### Next steps

Roughly ordered by grade points per effort. The grading line from the brief is in brackets.

1. **Threshold tuning** on out-of-fold probabilities; an early check (logistic regression, before the
   90-day gap) gave CV F1 0.742 at threshold 0.35 against 0.714 at 0.5 [classification metrics 0.5].
2. **Hyperparameter tuning** with Optuna, scored by the same gapped CV [tuning 0.75 + regression 1.5].
3. **Feature engineering** in `add_features()`: discount rate, margin, spend trend, gap ratios
   [preprocessing and FE 1 + regression 1.5].
4. **Feature selection** as a `"select"` step in `build_pipeline()`; `random_noise` must be rejected
   [feature selection 0.75].
5. **Preprocessing comparison:** encoders, scalers, imputers [preprocessing and FE].
6. **Statistical model comparison** (corrected paired t-test on the same folds) [model assessment].
7. **Client history lookup:** fill history fields from `train.csv` given a `customer_id` [deployment 1].
8. **Explainability:** SHAP for both models, global and per customer, in a new `src/explain.py` [explainability 1.5].
9. **Open-ended objective:** error analysis, stability over time or spend uncertainty [open-ended 1.5].

### Known issues

| # | Issue | Fix |
|---|---|---|
| R1 | Every `train` run prints the hold-out score, so it can drift into choices | Choose models, features and the threshold from CV only |
| R2 | `compare` sorts every regression metric ascending: correct for RMSE/MAE, backwards for R² | Store a direction per metric |
| R3 | Scores use raw spend; delivered spend is clipped at 0 and rounded | Score the clipped values, or measure the difference |
| R4 | The `clientes.csv` join does not check `customer_id` is unique | Merge with `validate="many_to_one"` |
| R5 | `client` crashes on non-numeric values or bad dates; saved models are not checked against their task | Validate input with clear messages |
| R6 | `client` treats a missing `prior_purchase_count` as a first purchase | Decide and document; see next step 7 |
| R7 | Train-validation gap is positive for F1 overfitting but negative for RMSE (CatBoost: about −1,300 EUR) | Report it so positive always means overfitting |

### Open questions

1. **Coherence:** keep "spend only for predicted returners", or show probability × conditional spend in the client?
2. **Client input:** JSON file now; an interactive prompt or small UI could count toward the open-ended part.

### Coding rules

- Only the libraries the brief allows: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, LIME,
  imbalanced-learn, TensorFlow, Keras, PyTorch. No AutoML-style tools.
- No leakage: nothing in `prepare()` may use other test rows; nothing learned may sit outside the `Pipeline`.
- Every command must run on a fresh clone; run the tests before pushing.
- Keep code small and explainable.
- A new dependency goes into both `pyproject.toml` and `requirements.txt` with the same pin.
- Update the status in readme if you change anything
