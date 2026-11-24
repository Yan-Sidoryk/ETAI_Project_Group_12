# Training pipeline: a detailed guide

This document explains the implementation in this repository as inspected on **17 November 2026**. It focuses on what happens when you run `python main.py train --task classification` or `python main.py train --task regression`, with model comparison and prediction included where they explain the training design.

The source code and current `config.yaml` determine the behavior described here. In particular, the selected classifier is currently **XGBoost**, although `AGENTS.md` describes an earlier random forest selection. The selected regressor is **CatBoost**. Changing the configuration can change those choices without changing the workflow.

The row counts and cross-validation fold sizes below were checked by running the current loading, preparation, splitting, and feature-selection functions against the supplied CSVs. Full production model training was not rerun to write this explanation, and no saved models were replaced.

## 1. What is being learned?

Each purchase row describes a customer **at one purchase occasion**. The model uses the current purchase, historical information already available at that purchase, and selected customer attributes to predict future behavior.

There are two separate supervised learning tasks:

| Task | Target | Meaning | Training population |
|---|---|---|---|
| Classification | `repeat_purchase_90d` | Whether the customer purchases again within the next 90 days: 0 or 1 | Usable purchase rows with the supplied classification labels |
| Regression | `spend_next_90d` | Spending on subsequent purchases in that 90-day period, in EUR | Usable purchase rows with an observed spend target |

The current purchase's value is an input; the regression target concerns **later** purchases. Similarly, `spend_90d` is historical spending, whereas `spend_next_90d` is the future outcome. They are different variables.

The two models are trained independently. The classifier is not fitted first and then used to select regression training rows. Regression training uses the observed spend target, rather than classifier predictions. Neither model consumes the other model's output during training.

They are combined during inference. If the classifier predicts no return, the client displays zero spend. If it predicts a return, the client displays the regressor's conditional spend prediction. The conditional amount is **not multiplied by the return probability**.

For example, a return probability of 0.70 and conditional spend of EUR 1,000 produce a positive decision at threshold 0.5 and a displayed amount of EUR 1,000. The pipeline does not display EUR 700.

## 2. The complete training flow

```text
python main.py train --task <classification or regression>
    |
    v
Read config.yaml and diagnosis.yaml
Select this task's configured algorithm and parameters
    |
    v
Read train.csv and clientes.csv
Remove training rows with completely empty purchase inputs
    |
    v
Apply fixed cleaning rules
Left-join selected customer attributes
Add is_first_purchase and purchase_month
    |
    v
Sort by purchase date
Reserve latest 20% of rows as holdout
    |
    +-- Regression: remove missing spend targets separately from
    |              development, holdout, and the full training population
    v
Separate input features X from target y
    |
    v
Create unfitted selected-model pipeline and dummy-baseline pipeline
    |
    v
Five expanding-window CV folds on development data
    - clone and fit preprocessing + baseline in each fold
    - clone and fit preprocessing + selected model in each fold
    - record training/validation metrics
    |
    v
Fit fresh selected-model and baseline copies on all development rows
Score both once on the reserved holdout
    |
    v
Fit selected-model pipeline on ALL usable rows, including holdout
    |
    v
Save fitted pipeline + settings; append scoreboard row; write report
```

There are two meanings of “pipeline” here:

1. **The overall workflow:** the full sequence above, implemented directly in `main.py:train`.
2. **The scikit-learn `Pipeline`:** the learned preprocessing and estimator fitted together inside that workflow.

Fixed cleaning and feature preparation happen before the scikit-learn pipeline. Learned imputation, encoding, scaling, and model fitting happen inside it.

## 3. Entry point and configuration

The argument parser at the bottom of [`main.py`](../main.py) defines four routes: `compare`, `train`, `predict`, and `client`. The selected command dispatches through `args.run(args)`. The routes do not call one another.

`train(args)` starts by reading configuration through [`load_config`](../src/data.py). It loads `config.yaml`, then inserts the contents of `diagnosis.yaml` under `config["diagnosis"]`. This produces one dictionary containing both experiment settings and fixed cleaning rules.

The task argument selects `config["classification"]` or `config["regression"]`. The route looks up the selected model name, then its settings in that task's `models` mapping. Merely listing an algorithm in this mapping does not train it during `train`.

Current settings include:

| Setting | Current value | Effect |
|---|---|---|
| `seed` | `42` | Requested random seed for estimators that support it |
| `validation.date_col` | `purchase_date` | Orders development/holdout rows |
| `validation.holdout_fraction` | `0.2` | Reserves the latest 20% of usable rows |
| `validation.n_splits` | `5` | Number of expanding-window development folds |
| `preprocessing.scaler` | `standard` | Standardizes numeric values after imputation |
| `preprocessing.numeric_imputer` | `median` | Configurable numeric imputation; missingness indicators enabled by default |
| `preprocessing.categorical_imputer` | `constant` | Configurable categorical imputation; defaults to filling with `"missing"` |
| `preprocessing.encoder` | `onehot` | One-hot or ordinal encoding |
| `preprocessing.min_frequency` | `20` | Groups categorical levels occurring fewer than 20 times in a fit |
| `classification.model` | `xgboost` | Classifier fitted and saved by the classification command |
| `classification.selection_metric` | `f1` | Primary classification metric |
| `classification.threshold` | `0.5` | Minimum return probability for a positive decision |
| `regression.model` | `catboost` | Regressor fitted and saved by the regression command |
| `regression.selection_metric` | `rmse` | Primary regression metric |

The route also generates a run identifier using local time with the format `YYYYMMDD_HHMMSS`. It is used in the artifact, scoreboard, and report filename.

Paths are relative to the current working directory, so commands should run from the repository root. Edit `config.yaml` to change settings; `compare`, `experiment`, and `train` also accept an optional `--config` path.

## 4. Loading data and removing empty training rows

[`load_data(config, "train")`](../src/data.py) reads two CSVs using pandas:

| File | Current dimensions | Role |
|---|---|---|
| `data/train.csv` | 6,534 rows × 40 columns | Purchase inputs and both targets |
| `data/clientes.csv` | 1,143 rows × 13 columns | Customer attributes |

Training does not read `test.csv` or the sample submission. Their contents do not determine fitted medians, category levels, model parameters, or validation scores.

Before cleaning or joining customers, the route constructs a list of raw purchase columns excluding `ID`, `customer_id`, and both targets. It passes this list to `drop_empty_rows` in [`src/cleaning.py`](../src/cleaning.py).

A row is removed only when **all** those raw purchase input fields are missing. This check includes fields such as `purchase_date` and `random_noise`, even though those columns are later excluded from model inputs. It is therefore an empty-record filter on the raw purchase schema, not a test of whether every final model feature is missing.

For the supplied data, it removes 65 rows, leaving **6,469**. Customer attributes are joined afterward so a customer match cannot make an otherwise empty purchase look complete. A row with some missing fields is retained for later imputation.

This function is used only for training/comparison. Prediction must retain every requested row.

## 5. Fixed cleaning rules: what changes, and in what order?

[`prepare`](../src/features.py) first calls `clean_dataset`, which returns a copy of the purchases with the same rows. Rules come from [`diagnosis.yaml`](../diagnosis.yaml), informed by the [audit notebook](../notebooks/01_data_audit.ipynb).

Cleaning is deterministic: it uses configured rules and values within the current row, rather than estimating statistics across purchases. Its order is significant.

### 5.1 Replace configured sentinel values with missing values

`prior_mean_purchase_value` and `spend_365d` values at or above `1e8` become `NaN`. This catches the `1e9` placeholder values found in the audit.

These are column-specific rules, not a general policy that all large monetary values are invalid.

### 5.2 Repair known sign flips

The same two columns are converted to absolute values. The audit found negative values consistent with sign-flipped amounts.

This happens before range checks; otherwise recoverable negative values would already have been discarded. Other negative numeric columns are not universally converted to positive values.

### 5.3 Replace out-of-range values with missing values

Configured inclusive ranges are enforced. Examples include nonnegative money and history values, at least one invoice/detail line, and a classification label between zero and one. Negative time gaps such as the `-4` placeholder become missing.

The rules also include target columns when present. Target range checking does not put targets into `X`. However, the training route does not separately filter missing classification labels; it relies on the supplied classification labels remaining valid. The current data meet that assumption.

### 5.4 Repair selected cross-column contradictions

Only consistency rules containing `set_nan` change data:

| Contradiction | Action |
|---|---|
| `spend_365d` exceeds `prior_total_spend` by more than EUR 0.01, within the rule's validity conditions | Clear `spend_365d` |
| `prior_max_purchase_value` exceeds `prior_total_spend` by more than EUR 0.01 | Clear `prior_max_purchase_value` |
| `prior_mean_purchase_value` differs from `prior_total_spend / prior_purchase_count` by more than EUR 0.01, within the rule's conditions | Clear the recorded mean |

Other consistency rules, such as rolling purchase counts being out of order, are diagnostic. `check_rules` can count them, but `train` does not call `check_rules`, and `clean_dataset` does not repair rules lacking `set_nan`.

### 5.5 Recompute an exactly derivable historical value

For rows with a positive prior purchase count:

```text
prior_mean_purchase_value = prior_total_spend / prior_purchase_count
```

This recomputation applies to all rows satisfying that condition, not just previously missing or inconsistent means. A missing numerator still produces a missing result. The code does not invent a historical mean when the prior count is zero.

No learned imputation occurs here. Missing values remaining after repairs are handled later, using only the training portion of each fit.

## 6. Customer join and feature engineering

After cleaning, `prepare` selects the customer key and three configured attributes from `clientes.csv`, then performs a left join on `customer_id`:

- `country_or_region`
- `commercial_region`
- `customer_since_year`

A left join keeps purchases lacking a customer match; the joined attributes become missing. Other customer columns are not loaded into the feature frame. This excludes fields such as nearly customer-specific company names and addresses, and the entirely empty `customer_type`.

The supplied customer table was checked and its keys are unique. The implementation does **not** enforce a many-to-one join with `validate="many_to_one"`; duplicate customer keys in future input could multiply purchases.

Two row-local features are added:

| Feature | Calculation | Meaning |
|---|---|---|
| `is_first_purchase` | `(prior_purchase_count == 0).astype(int)` | Explicit flag for no prior recorded purchases |
| `purchase_month` | Parse `purchase_date`, extract month | Month number from 1 to 12 |

These functions do not calculate customer histories by aggregating other purchase rows. They use the supplied historical columns. This also allows the same preparation to work on a single purchase at inference.

A missing prior count yields `is_first_purchase = 0` in `prepare`; unknown history is not automatically interpreted as first purchase during training. The client route separately defaults an omitted prior count to zero before preparation. `purchase_month` is treated as numeric downstream, with no cyclic encoding. `customer_since_year` is also numeric.

Date parsing uses `pd.to_datetime` without an explicit coercion policy for malformed text; the workflow assumes parseable dates. A historical snapshot guarantee for customer attributes is not enforced in code, so their availability at each purchase remains a data assumption.

## 7. Reserve the holdout before filtering regression targets

[`split_holdout`](../src/evaluate.py) sorts prepared rows by parsed purchase date using a stable sort, then splits by **row position**:

```text
holdout_size = round(number_of_rows × holdout_fraction)
cut = number_of_rows - holdout_size
development = ordered_rows before cut
holdout = ordered_rows from cut onward
```

With 6,469 rows and a fraction of 0.2, development contains **5,175** rows and holdout contains **1,294** rows. The holdout is kept out of comparison and development cross-validation.

Both task routes make this split on the same prepared population. Regression then removes rows whose cleaned spend target is missing from development, holdout, and the full refit population separately. This preserves the common initial chronological cut rather than selecting a different 20% split among only observed spend outcomes.

| Task | All rows available for final refit | Development | Holdout |
|---|---:|---:|---:|
| Classification | 6,469 | 5,175 | 1,294 |
| Regression | 3,425 | 2,733 | 692 |

Missing spend is not replaced with zero. In the raw data, all 2,880 non-returners have missing spend, but so do 229 returners. After empty-row removal, 196 returners still lack spend. The observed-spend population therefore represents returners with known spend, rather than every actual returner.

The regression route does not explicitly test `repeat_purchase_90d == 1`: its implemented criterion is `spend_next_90d.notna()`. Those criteria are consistent for observed spends in the supplied data, but are not guaranteed by a code-level assertion.

**The current split crosses equal dates:** the latest development date and earliest holdout date are both **2023-06-17**. The stable sort retains input order within a date; it does not keep all purchases on that date in one partition.

## 8. Constructing X and y

[`split_features_target`](../src/features.py) returns:

- `X`: the model input table.
- `y`: the selected task's target column.

It removes identifiers, configured dropped columns, and **both targets** from `X`, regardless of which task is running:

```text
ID
customer_id
purchase_date
random_noise
repeat_purchase_90d
spend_next_90d
```

This prevents the regression target from leaking into classification and prevents the return label from becoming a regression feature. Customer ID remains useful for joining, and date remains useful for ordering splits, without either being an estimator input.

There are currently **39 input columns before learned transformations**, shared by both tasks: 31 numeric and 8 categorical. They cover current purchase totals/details, historical frequency and spend, historical timing/tenure, the selected customer attributes, and the two engineered features. The exact columns can be inspected through `X.columns`; their definitions are in the [data dictionary](column_descriptions.md).

The number of transformed estimator inputs is different from 39 because one-hot encoding expands categoricals and imputation can add missingness indicators. It may vary between folds because category frequencies and observed missingness vary by training window.

The current feature removal is explicit and configuration driven. There is no fitted feature selector, recursive elimination, or automatic feature-importance filtering in the pipeline.

## 9. The learned preprocessing pipeline

[`build_pipeline`](../src/model.py) constructs a scikit-learn pipeline with two named steps:

```text
Pipeline
├── prep: ColumnTransformer
│   ├── categorical: constant imputation -> one-hot encoding
│   └── remaining numeric: median imputation + indicators -> scaling
└── model: selected estimator, optionally wrapped for target transformation
```

[`build_preprocessor`](../src/preprocessing.py) defines the transformations. Building it does not fit anything; `.fit(...)` learns its state. The diagram and descriptions below show the default configuration. All four components can be selected by name or with `{type: name, params: {...}}`; see [Choosing preprocessing](../README.md#choosing-preprocessing) for the full options and examples.

### 9.1 Categorical columns

The eight explicitly listed categorical inputs are `payment_type`, `salesperson`, `commercial_zone`, `transport_method`, `warehouse`, `document_series`, `country_or_region`, and `commercial_region`.

They pass through:

1. `SimpleImputer(strategy="constant", fill_value="missing")`: replaces blanks with the literal category `missing`.
2. `OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)`: converts learned categorical levels into dense indicator columns and groups infrequent levels.

The frequency threshold is an absolute count in that fit's training rows. It is recalculated in each fold. The `missing` level may itself be grouped if it is infrequent.

A category never seen during fitting produces all zeros for that feature's encoded block, rather than raising an error. With `handle_unknown="ignore"`, it is not automatically assigned to the infrequent bucket.

### 9.2 Numeric columns

Every remaining input column is treated as numeric and passes through:

1. `SimpleImputer(strategy="median", add_indicator=True)`: learns each numeric feature's training median, fills missing values, and appends missingness flags for features missing during fitting.
2. The configured scaler, currently `StandardScaler`: learns training means and standard deviations and standardizes the imputed matrix.

Because scaling follows imputation, appended missingness flags are scaled too. An indicator is not created merely because a feature first becomes missing in validation; indicator creation depends on missingness observed during fitting. Entirely empty numeric columns follow `SimpleImputer`'s default behavior; the repository does not explicitly configure `keep_empty_features`.

Scaler alternatives are `none` (pass through), `minmax`, `robust`, `maxabs`, `normalize`, `power`, and `quantile`. Numeric imputers also support `mean`, `most_frequent`, `constant`, `knn`, and `none`; categorical imputers support `most_frequent` and `none`. The alternative encoder is `ordinal`, which uses -1 for unseen categories by default. Explicit component parameters override defaults, including the shared `min_frequency` setting for either encoder. All candidate algorithms use the same configured preprocessing. CatBoost, XGBoost, and LightGBM receive the transformed numeric matrix; CatBoost is not using its native categorical-feature mechanism here.

`python main.py experiment --task classification` (or `regression`) compares candidate
lists from `preprocessing_experiment` on identical development folds. It performs a full grid
search, with optional `--models` to compare several configured models.
It never scores the holdout or saves fitted models. Each run saves summary and per-fold CSVs
in a unique output folder and updates a fixed latest-summary CSV. Model and preprocessing
choices are recorded in separate `model`, `scaler`, `numeric_imputer`, `categorical_imputer`,
and `encoder` columns, with `candidate` and `task` identifying results. It generates
no YAML files; selected settings are applied manually to the existing `config.yaml`.
See [Comparing preprocessing automatically](../README.md#comparing-preprocessing-automatically)
for configuration and training examples.

### 9.3 Why preprocessing is fitted inside each fold

Suppose validation contains unusually high spending values. If the median or scaler were fitted before splitting, those future values would influence the training transformation. Here, fitting a fresh pipeline on a fold's training rows ensures that its medians, category vocabulary, rare-category grouping, scaling statistics, and estimator all depend on that training window only.

The validation rows are transformed using the already fitted preprocessing. They do not update it. The same principle applies to holdout evaluation and later prediction.

Fixed cleaning can precede splitting because those configured repairs do not estimate statistics across rows. The rules were informed by exploratory auditing, so this is a fixed engineering policy, rather than a claim that all project decisions were made without looking at any later-period data.

## 10. Model construction and current algorithms

The model registry in [`src/model.py`](../src/model.py) maps configuration names to estimator classes. `build_pipeline(task, config, model_config)` creates the estimator with its configured parameters, attempts to set `random_state=42` when not explicitly provided, optionally wraps a log-target estimator, and combines it with preprocessing.

It uses the estimator library's defaults for parameters not present in YAML. The random seed improves repeatability where supported; it does not make timestamped reports identical or guarantee byte-identical artifacts across machines and library versions.

### 10.1 Selected classifier: XGBoost

Current explicit parameters:

```yaml
n_estimators: 200
learning_rate: 0.05
max_depth: 4
```

`XGBClassifier` builds a boosted ensemble of trees: successive trees improve the ensemble's classification objective. These settings control the number of boosting rounds, the contribution of each round, and maximum tree depth. The fitted classifier supplies `predict_proba`, and the evaluation/inference code applies the configured threshold to the positive-class probability.

The primary evaluation metric is F1, but that does not mean XGBoost directly optimizes F1 during fitting. The route does not supply a custom F1 objective or search for the best threshold. It also does not pass a validation set or early-stopping configuration to `.fit(...)`.

### 10.2 Selected regressor: CatBoost

Current explicit parameters:

```yaml
iterations: 300
learning_rate: 0.05
depth: 6
verbose: 0
allow_writing_files: false
log_target: false
```

`CatBoostRegressor` is another boosted tree ensemble. Iterations, learning rate, and depth control its ensemble capacity and learning steps. `verbose: 0` suppresses training progress, while `allow_writing_files: false` prevents CatBoost's own auxiliary files. The repository still explicitly saves the completed pipeline with joblib.

`log_target` is a repository-level option, not a parameter passed directly to CatBoost. It is false for every configured regression candidate. Thus, current regression fits use spend in EUR directly.

If enabled, `build_pipeline` uses `TransformedTargetRegressor` to fit on `log1p(y)` and convert predictions back with `expm1`. Metrics would still receive predictions in EUR. No separate retransformation-bias correction is implemented.

### 10.3 Other available candidates

| Classification | Regression |
|---|---|
| Logistic regression | Ridge regression |
| Random forest | Random forest |
| Histogram gradient boosting | Histogram gradient boosting |
| XGBoost | XGBoost |
| LightGBM | LightGBM |
| CatBoost | CatBoost |

There are six non-dummy candidates per task. `compare` evaluates all configured candidates; `train` evaluates and saves only the selected one, alongside its baseline. Neither command performs a systematic hyperparameter search.

## 11. Baselines: what performance must the model improve on?

The `FLOOR` mapping defines feature-ignoring reference estimators:

| Task | Estimator | Prediction |
|---|---|---|
| Classification | `DummyClassifier(strategy="constant", constant=1)` | Predict return for every purchase |
| Regression | `DummyRegressor(strategy="mean")` | Predict the training window's mean spend for every purchase |

The always-return classifier is a useful F1 reference when positive labels are common. If the positive fraction is `p`, its precision is `p`, recall is 1, and F1 is `2p / (1+p)`. Its probability predictions are constant, so it has no ranking ability.

The mean-spend baseline is appropriate for comparison under squared error/RMSE: the mean minimizes training squared error among constant predictions. It is recalculated from each fold's training targets, not from validation targets.

Baselines are constructed with the same preprocessing structure and assessed on the same folds, though their estimators ignore the features. The code reports baseline performance; it does not automatically refuse to save a selected model that fails to beat it.

## 12. Expanding-window cross-validation

The route creates `TimeSeriesSplit(n_splits=5)` and passes it to [`cross_validate_pipeline`](../src/evaluate.py), first for the baseline and then for the selected model.

For each fold, that function:

1. Takes training and validation row positions from the splitter.
2. Calls `clone(pipeline)` to obtain a fresh, unfitted copy.
3. Fits that copy on the fold's training `X` and `y`.
4. Scores all configured metrics on the validation rows.
5. Scores the selection metric on the training rows.
6. Records the fold number, training row count, training score, validation score, gap, and validation metrics.

Cloning preserves configuration but does not carry fitted trees, medians, or encodings from another fold. There is no incremental continuation from the previous fold's estimator.

Training windows grow and validation windows move forward:

```text
time ------------------------------------------------------->
fold 1: [train] [validate]
fold 2: [------ train ------] [validate]
fold 3: [------------- train -------------] [validate]
...
```

These are the actual fold sizes with the current inputs:

| Fold | Classification train | Classification validation | Regression train | Regression validation |
|---|---:|---:|---:|---:|
| 1 | 865 | 862 | 458 | 455 |
| 2 | 1,727 | 862 | 913 | 455 |
| 3 | 2,589 | 862 | 1,368 | 455 |
| 4 | 3,451 | 862 | 1,823 | 455 |
| 5 | 4,313 | 862 | 2,278 | 455 |

Regression folds are built after filtering unknown spend targets. They therefore do not necessarily share classification's internal CV date boundaries. The common split is the initial development/holdout cut.

The splitter uses rows, not fixed calendar periods. Validation windows have equal row counts within each task but may cover unequal durations. There is no shuffling, no customer grouping, no configured gap, and no date-group enforcement. A customer can occur in both past training and later validation rows.

With five folds, one `train` command performs **13 pipeline fits**: five baseline fits, five selected-model fits, two development fits for holdout evaluation, and one final selected-model fit on all usable rows. No fold model or baseline is saved as the deployment artifact.

## 13. Metrics and interpretation

[`score_model`](../src/evaluate.py) calculates metrics directly from a fitted model's outputs.

### 13.1 Classification

The evaluator calls `predict_proba(X)[:, 1]`, then computes:

```text
predicted_return = 1 if probability >= 0.5, otherwise 0
```

The configured threshold is applied both during validation and inference. Current metrics are:

| Metric | Input | Interpretation |
|---|---|---|
| F1 | Thresholded labels | Harmonic mean of positive-class precision and recall; primary metric |
| Precision | Thresholded labels | Fraction of predicted returners who actually return |
| Recall | Thresholded labels | Fraction of actual returners detected |
| Accuracy | Thresholded labels | Fraction of all labels predicted correctly |
| ROC AUC | Probabilities | How well returners rank above non-returners across thresholds |

F1, precision, and recall use `zero_division=0`. Balanced accuracy and average precision are also supported by the metric registry but are not currently requested in YAML. Metrics such as ROC AUC require suitable target variation; the evaluator does not implement special handling for a single-class evaluation window.

### 13.2 Regression

The evaluator calls `model.predict(X)` and measures:

| Metric | Definition | Better direction |
|---|---|---|
| RMSE | Square root of mean squared prediction error, in EUR | Lower |
| MAE | Mean absolute prediction error, in EUR | Lower |
| R² | `1 - squared_error_sum / target_variation_sum` | Higher |

RMSE penalizes large errors more heavily than MAE. R² is unitless and can be negative. Regression scores describe rows with known spending; they do not measure the complete classifier-gated system across all purchases.

Regression evaluation uses raw predictions in EUR. It does **not** clip negative predictions or round to cents, whereas inference does both. This distinction should be considered when interpreting reported errors against delivered behavior.

### 13.3 Fold summaries and the gap

The code takes the arithmetic mean of fold validation scores. For RMSE, this is the mean of the five fold RMSEs, not RMSE recomputed from all validation predictions pooled together. Standard deviation uses sample standard deviation (`ddof=1`); it describes variation across folds, not a confidence interval.

The stored gap always equals:

```text
gap = training selection metric - validation selection metric
```

For F1, a large positive gap suggests better training performance. For RMSE, better training performance usually produces a negative gap, because lower error is better. Interpret the sign using the metric's direction; it is not a universally positive overfitting score.

## 14. Holdout evaluation uses a separate fit

After cross-validation, `train` fits two fresh clones on **all development rows**:

```python
holdout_model = clone(pipeline).fit(X_dev, y_dev)
holdout_baseline = clone(floor_pipeline).fit(X_dev, y_dev)
```

Both are scored on the reserved holdout. Their learned preprocessing has seen development rows only. The selected model's configured holdout metrics and the baseline's primary holdout metric appear in the printed summary/report.

This is a separate assessment from CV. CV supplies several development-period estimates; holdout assesses the already selected settings on the later reserved period. The route does not automatically revise settings after seeing that score.

The holdout remains useful only if its results are not repeatedly used to choose models, thresholds, or features. Running many experiments and choosing whichever gives the best holdout score would turn it into another selection set.

## 15. Final refit: which fitted model is actually saved?

Once assessment is complete, the route builds `X_all, y_all` from every usable row and calls:

```python
pipeline.fit(X_all, y_all)
```

This fits the original selected-model pipeline, which has remained unfitted while clones handled CV and holdout assessment. It learns new preprocessing state and model parameters from all available task-specific training data, **including the holdout**.

The saved classifier therefore uses 6,469 rows; the saved regressor uses 3,425 rows with observed spend. This makes the final deployment model different from the model whose holdout score is reported.

Using holdout rows in the final refit is a deployment decision after evaluation. The previously measured score remains a score for the development-only model. The workflow does not then re-score the final model on the now-incorporated holdout and present that as independent performance.

## 16. Saved artifacts and experiment evidence

### 16.1 Model artifact

`joblib.dump` writes:

- Classification: `models/classifier.joblib`
- Regression: `models/regressor.joblib`

Each artifact contains a dictionary:

| Key | Contents |
|---|---|
| `pipeline` | Final fitted preprocessing and estimator |
| `threshold` | Classification threshold; `None` for current regression settings |
| `features` | Ordered list of the 39 pre-transformation input names |
| `config` | Full configuration snapshot, including cleaning rules |
| `run` | Timestamp-based run identifier |

The fitted preprocessor inside `pipeline` contains learned medians, scaling statistics, and encoder categories. The `features` entry is the original model input schema, not the expanded one-hot feature names.

The fixed artifact filename is overwritten by the next training command for that task. Fitting the regressor does not overwrite the classifier, and vice versa. Baselines and individual fold models are not persisted.

### 16.2 Scoreboard

[`save_scores`](../src/results.py) appends a row to `outputs/runs.csv`. The row contains the run ID, task, algorithm, parameter string, threshold, primary metric, CV mean/std, baseline CV mean, selected-model holdout score, and each fold's primary validation score.

The scoreboard is a compact experiment history. Its parameter string does not capture every possible setting, such as the optional log-target flag or full preprocessing configuration; the artifact and report supply additional context.

### 16.3 Text report

`outputs/run_<run>.txt` contains:

- Run ID, task, model parameters, preprocessing settings, and threshold.
- Full baseline and selected-model CV fold tables.
- Means and standard deviations for each configured validation metric.
- Training/validation gap summaries.
- Holdout summary and final refit row count/path.

Run IDs have one-second resolution and report filenames do not include the task. Runs started in the same second can share an identifier and overwrite a report. Generated models and outputs are Git-ignored, so retain final experiment evidence deliberately.

## 17. How comparison fits into training

The intended sequence is:

```bash
python main.py compare --task classification
python main.py compare --task regression
# Review the tables; edit selected models/settings in config.yaml.
python main.py train --task classification
python main.py train --task regression
```

`compare` repeats training preparation and development splitting, then cross-validates the baseline and all six configured candidates per task. It writes `outputs/comparison_classification.csv` or `outputs/comparison_regression.csv`, with mean validation metrics, primary-metric standard deviation, average gap, and elapsed time.

It does not score holdout, save a trained artifact, change YAML, or automatically choose the top-ranked algorithm. Each candidate is one configured parameter set, not a grid search. With seven entries and five folds, a full comparison performs 35 fits per task.

Current ranking is descending for classification and ascending for regression. This is correct for F1 and RMSE. If regression's selection metric changes to R², the current sorting logic would incorrectly put lower R² first and should be updated.

## 18. How prediction reuses training decisions

After both models are trained:

```bash
python main.py predict
python main.py client --input examples/returning_customer.json
python main.py client --input examples/new_customer.json
```

Both inference routes load saved artifacts rather than fitting anything. [`classify` and `predict_spend`](../src/inference.py) use each artifact's saved configuration for preparation and its saved `features` list to enforce column order, then use the fitted pipeline.

This means changing the current YAML threshold or preprocessing settings does not silently change an already fitted model's inference policy. Retraining creates a new artifact with the changed settings. Current configuration still locates model/data paths, and inference reads the current customer CSV rather than an embedded copy of that table.

The client applies the return decision before calling the regressor. Batch prediction computes spend predictions for the batch and exports only those rows whose classification decision is positive. Inference clips spend at zero and rounds it to two decimals.

The batch classification file contains every test ID in order, while the regression file contains only predicted returners. Neither inference command needs the training CSV or uses other test rows to build purchase-history features.

## 19. Current assessment limits and unfinished modeling work

The following distinguish implemented behavior from future improvements:

1. **No 90-day outcome-window gap.** Splits place earlier purchase rows before later ones, but labels summarize the next 90 days. Near a split boundary, a training outcome window can extend into the validation period. The current code does not purge such rows. A deployment-faithful policy should account for when outcomes become observable, not just when the purchase occurred.
2. **Equal dates can cross partitions.** The current development/holdout boundary demonstrably divides purchases on 2023-06-17. CV also splits by row position. Grouping dates and defining calendar cutoffs would make the temporal separation clearer.
3. **Customers are not held out as groups.** This evaluates future purchases in a mixture that can include known customers. It is not a dedicated test of generalization to entirely unseen customers.
4. **Regression has an observed-target population.** Missing spend among actual returners may affect representativeness. Missing targets are excluded, with no missing-outcome model or correction.
5. **No combined-system error analysis.** Classification and conditional regression are assessed separately. The reports do not quantify spending missed by false negatives or spurious spend delivered for false positives.
6. **Assessment and delivery differ for spend.** Evaluation uses raw predictions; delivery clips and rounds. Their effect on metrics has not been quantified by this workflow.
7. **Selection is manual.** There is no systematic hyperparameter tuning, threshold search, fitted feature selection, probability calibration, or global/individual model-explanation stage in the current training route.
8. **Schema assumptions remain.** Customer-key uniqueness, valid classification labels, parseable dates, and numeric types outside the categorical list are not comprehensively enforced by the training route.

These limitations do not change the execution order described above. They identify where future modeling work would extend or revise it.

## 20. Running and checking the workflow

Use Python 3.12 and the environment setup in [`README.md`](../README.md). From the repository root, `uv sync` installs the locked environment; commands can then use `uv run python ...`. With the documented pip setup, activate the virtual environment and use `python ...`. The training command syntax is the same on Mac and Windows.

To inspect available commands and run the existing integration suite:

```bash
python main.py --help
python -m unittest discover -s tests -v
```

The tests in [`tests/test_pipeline.py`](../tests/test_pipeline.py) are designed to use temporary output/model paths, reduced model iterations, and two folds. They cover training outputs, comparisons, export coherence, saved feature schemas, client behavior, and missing-model/invalid-field handling. They do not replace a full production comparison or verify modeling quality.

One current test-configuration mismatch is visible in the source: test setup replaces the classification candidate mapping with only `random_forest`, but inherits the currently selected `classification.model: xgboost`. Without synchronizing that selected name, training in test setup will look up a missing candidate. This documentation change does not modify the tests or the owner's model configuration, and does not claim that the full suite was run successfully.

### Source map

| Read this file | For this part of training |
|---|---|
| [`main.py`](../main.py) | Full `train` sequence, comparison route, command dispatch |
| [`config.yaml`](../config.yaml) | Selected algorithms, parameters, metrics, columns, paths |
| [`diagnosis.yaml`](../diagnosis.yaml) | Fixed repairs, validity bounds, consistency rules |
| [`src/data.py`](../src/data.py) | YAML and raw CSV loading |
| [`src/cleaning.py`](../src/cleaning.py) | Empty-row filtering and deterministic repairs |
| [`src/features.py`](../src/features.py) | Join, engineered features, separation of X and y |
| [`src/preprocessing.py`](../src/preprocessing.py) | Learned imputation, encoding, scaling |
| [`src/model.py`](../src/model.py) | Estimator registry, baselines, pipeline construction |
| [`src/evaluate.py`](../src/evaluate.py) | Chronological split, CV loop, metrics, fold reports |
| [`src/results.py`](../src/results.py) | Scoreboard append and text-report writing |
| [`src/inference.py`](../src/inference.py) | Loading artifacts and applying saved policies |
