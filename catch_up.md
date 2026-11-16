# Project catch-up — ETAI Group 12

Reviewed: 16 November 2026. This report describes the code, notebook, raw data and local generated artifacts available at review time. Results below are local validation results, not Kaggle scores. The filename `catch_up.d` follows the requested name; its contents are Markdown.

## 1. What the project does

The project builds a purchase-level prediction tool for Everything & Then Some, a wholesaler. Given a current purchase, it predicts:

1. Whether the customer buys again within 90 days (`repeat_purchase_90d`).
2. Their subsequent 90-day spending in EUR (`spend_next_90d`), conditional on returning.

The classifier decides whether to return a spending prediction. The client reports zero spending for predicted non-returners; batch regression output contains only predicted returners. This makes the two outputs coherent. The displayed spending figure is a decision-gated conditional estimate, not a probability-weighted expected value across both possible outcomes.

The intended business uses are incentives for customers unlikely to return and retention/account management for valuable customers. The brief in `docs/guidelines.md` specifies a deadline of 22 December at 18:00, a runnable repository, final Kaggle submission and `Group12_ETAI_2627.zip`.

**Overall assessment:** a working modelling foundation with substantial data auditing and a trained classifier. It is not yet a fully evidenced final submission: the local regressor is missing, model-comparison results are absent, validation needs tightening, and feature selection, systematic tuning and explainability remain unfinished.

## 2. Data and modelling approach

| Dataset | Rows | Columns | Role |
|---|---:|---:|---|
| `data/train.csv` | 6,534 | 40 | Purchase inputs plus both targets |
| `data/test.csv` | 986 | 38 | Later purchase occasions without targets |
| `data/clientes.csv` | 1,143 | 13 | Customer attributes joined on `customer_id` |
| `data/sample_submission.csv` | 986 | 2 | Required Kaggle column layout and ID order |

The raw classification labels contain 3,654 returners and 2,880 non-returners. Spending is missing for all 2,880 non-returners and another 229 returners. Missing spend therefore cannot generally be replaced with zero.

Training removes 65 rows whose purchase features are entirely missing, leaving 6,469 classification rows. Of these, 3,425 have observed spending and can train the regressor. The audit's 196 returners with missing spend refers to the filtered population; the data dictionary's 229 refers to the raw population. These are consistent counts from different populations.

Training dates run from July 2015 to February 2025; test dates run from May 2025 to October 2026. Test customers generally have longer histories, so validation must represent future predictions. The current code reserves the latest 20% of cleaned rows and uses five expanding-window folds on the earlier 80%.

The shared prediction flow is:

```text
Raw purchase row(s)
  -> deterministic cleaning from diagnosis.yaml
  -> join selected customer attributes
  -> add first-purchase flag and purchase month
  -> remove IDs, targets, raw date and random_noise
  -> learned imputation, categorical encoding and numeric scaling
  -> classifier probability and threshold decision
  -> conditional spend estimate from a separate regressor
  -> coherent batch files or one-purchase terminal response
```

Cleaning and feature engineering operate independently on each purchase. Learned preprocessing is inside the scikit-learn pipeline, so it is fitted separately within each training fold and saved with the estimator. Prediction does not fit models or aggregate other test rows.

## 3. Every file in src/

### `src/__init__.py`

Empty package marker. It makes `src` an importable Python package; it has no runtime logic.

### `src/data.py` — configuration and raw inputs

- `load_config()` reads `config.yaml` and attaches `diagnosis.yaml` under the `diagnosis` key.
- `load_raw()` reads train, test and customer CSVs and returns all three.
- `join_clients()` left-joins the configured customer columns onto purchase rows.

The selected customer fields are `country_or_region`, `commercial_region` and `customer_since_year`. A missing customer match produces missing joined values for downstream imputation. The provided customer keys are unique; the join does not explicitly enforce that requirement, so a future duplicate key could multiply purchase rows. Paths are interpreted relative to the working directory, which is why commands run from the repository root.

### `src/cleaning.py` — deterministic data repairs

- `_violations()` evaluates a consistency expression and its optional row condition.
- `check_rules()` counts range, sentinel and cross-column violations for auditing.
- `clean_dataset()` copies the data and applies repairs in order: large sentinel values to missing; known sign flips to absolute values; out-of-range values to missing; selected contradictions to missing; then exact recomputation where possible.
- `drop_empty_rows()` removes training records whose supplied feature columns are all missing.

Rules live in `diagnosis.yaml`. Examples include invalid negative purchase gaps, billion-value placeholders, historical spending exceeding lifetime spending and inconsistent historical means. Only consistency rules with `set_nan` trigger repairs; other rules are diagnostic. Prediction keeps every input row. On the cleaned classification training rows, the configured checks found no remaining violations during this review.

### `src/features.py` — shared feature construction

- `add_features()` adds `is_first_purchase` from a zero prior count and `purchase_month` from the date.
- `prepare()` performs cleaning, the customer join and feature engineering in the same order for training and inference.
- `split_features_target()` returns inputs and the selected target, excluding both targets, identifiers and configured dropped columns. It returns no target when that column is absent.

This centralises feature preparation and prevents the spending target from leaking into classification. Current engineering is deliberately small: there are no trend, margin, discount or recency/frequency ratio features yet. A missing prior count becomes a false first-purchase flag unless the client has first supplied its default history values.

### `src/preprocessing.py` — learned input transformations

`build_preprocessor()` constructs a `ColumnTransformer` with:

- Configured categorical columns: constant missing-value imputation followed by dense one-hot encoding. Categories occurring fewer than 20 times are grouped. Unknown categories produce zero indicators rather than an error.
- All remaining columns: median imputation, missingness indicators for columns missing during fitting, then the configured scaler.

Supported scalers are none, standard, min-max and robust; the current setting is standard. This preprocessing is fitted inside each model pipeline. All algorithms currently share it: there are no algorithm-specific preprocessing choices or native categorical boosting paths.

### `src/model.py` — model registry and pipelines

- `_MODELS` maps task/model names to estimator classes.
- `FLOOR` supplies an always-return classifier and mean-spend regressor as baselines.
- `build_model()` instantiates the configured estimator, sets its random seed where supported and optionally wraps regression in a log-target transformation.
- `configured_model()` resolves a named candidate or the selected default from configuration.
- `build_pipeline()` combines preprocessing and model into one estimator with `prep` and `model` steps.

Classification candidates: logistic regression, random forest, histogram gradient boosting, XGBoost, LightGBM and CatBoost. Regression candidates: ridge plus those same five tree/boosting families. There are six candidates per task, exceeding the brief's minimum algorithm counts if comparisons are actually run and documented.

The selected models are random forest classification (300 trees, minimum leaf size 5) and CatBoost regression (300 iterations, learning rate 0.05, depth 6). All configured regressors have `log_target: false`. Configuration comments describe earlier log-target tradeoffs, but there are no local regression reports to verify those numerical claims.

### `src/evaluate.py` — chronological assessment and metrics

- `split_holdout()` stably sorts purchase rows by date and takes the most recent fraction as holdout.
- `make_cv()` creates five-fold `TimeSeriesSplit`.
- Metric registries support classification F1, precision, recall, accuracy, balanced accuracy, ROC AUC and average precision; regression RMSE, MAE and R².
- `build_scorer()` scores classifier labels at the configured probability threshold or scores regression predictions.
- `cross_validate_pipeline()` clones and fits the full pipeline per fold, recording validation metrics, training score and their gap.
- `cv_report()` formats fold results and metric means/standard deviations.

Current selection metrics are classification F1 and regression RMSE. Classification threshold is 0.5. Splits are chronological by row but have no calendar embargo for the 90-day target horizon. The reported gap is always training minus validation: positive gaps suggest overfitting for F1, whereas negative gaps can indicate overfitting for error metrics such as RMSE.

### `src/results.py` — experiment records

- `save_scores()` appends a summary row to the configured scoreboard CSV and returns the full table.
- `save_run()` writes a detailed text report named with the run identifier.

Training records the task, model, parameters, threshold, selection metric, CV mean/variation, baseline, holdout result and fold scores. Detailed reports include preprocessing and all requested metrics. These are useful records, but they do not contain a source revision or data fingerprint. The scoreboard is append-only without deduplication; report names have second-level timestamps and could collide for runs started together.

## 4. main.py — how everything is coordinated

`main.py` is the terminal entry point and workflow coordinator. It provides four commands:

| Command | Behaviour |
|---|---|
| `compare --task classification` or `regression` | Cross-validates the baseline and all six configured candidates on development rows, ranks the average scores and writes `outputs/comparison_<task>.csv`. Does not automatically select or save the winner. |
| `train --task classification` or `regression` | Assesses the selected model against its baseline, evaluates the held-out period, refits on all usable rows and saves the final pipeline plus reports. |
| `predict` | Loads both models, predicts all test purchases and writes the two required batch files. |
| `client --input <json>` | Loads one purchase from JSON and prints return probability, decision and spending. |

The main helper functions are:

- `load_training_rows()`: loads raw files, removes empty training rows and runs shared preparation.
- `with_observed_spend()`: selects rows with a nonmissing spend target. It does not explicitly require a positive classification label; the provided data satisfies that relationship.
- `load_splits()`: establishes one chronological holdout boundary for both tasks, then filters regression partitions to observed spend. Regression CV folds are subsequently formed on that filtered population, so their dates need not match classification CV folds.
- `load_model()`: reads the saved bundle or exits with the required training command if it is absent.
- `classify()`: prepares raw purchases using saved settings, orders the saved features and returns probabilities plus threshold decisions.
- `predict_spend()`: prepares inputs using saved settings and returns spending clipped to nonnegative EUR and rounded to cents.
- `score_purchase()`: rejects unknown JSON fields, converts omitted fields to missing values and defaults history counts/totals to zero when no prior purchase count is supplied. It runs both stages as needed and returns a result dictionary.
- `main()`: parses commands and dispatches to the selected handler.

Saved files are `models/classifier.joblib` and `models/regressor.joblib`. Each bundle includes the fitted pipeline, threshold, feature names, configuration and run identifier. The saved model is refitted on all usable training rows, including the holdout; holdout metrics come from a separate model fitted only on development rows.

Batch output is `outputs/submission.csv` (`ID,repeat_purchase_90d`) and `outputs/regression_predictions.csv` (`ID,repeat_purchase_90d,spend_next_90d`, predicted returners only). Spending is currently computed for all test rows before filtering. The client and batch commands load all raw CSVs through `load_raw()`, even where the training CSV is not needed for inference.

## 5. Other project files

| File or directory | Summary |
|---|---|
| `README.md` | Project introduction, structure, Mac/Windows setup, command examples and output descriptions. Documents both uv and pip workflows. |
| `config.yaml` | Central paths, targets, excluded columns, joined attributes, validation settings, preprocessing, six candidate definitions per task, selected models and output locations. |
| `diagnosis.yaml` | Executable cleaning specification: allowed ranges, sentinel cutoffs, sign repairs, consistency rules and exact historical-mean reconstruction. |
| `notebooks/01_data_audit.ipynb` | 76-cell audit covering schema, missingness, duplicates, corrupted values, target distributions, numeric/categorical signals, time, noise, customers and train/test shift; ends with a cleaning plan. It contains saved outputs, but was not re-executed in this review. |
| `docs/column_descriptions.md` | Detailed raw-data dictionary, relationships, missing counts, target caveats and field interpretations. Distinguishes raw statistics from filtered notebook statistics. |
| `docs/guidelines.md` | Readable brief, grading rubric, deployment requirements, leakage restrictions, deliverables and defence expectations. |
| `docs/project_guidelines.pdf` | Original brief kept as reference. This report uses the Markdown transcription; the PDF was not independently checked. |
| `examples/new_customer.json` | Purchase details without a customer ID or purchase history; exercises the client defaults. |
| `examples/returning_customer.json` | Purchase details, customer ID and historical behaviour for a returning-customer example. |
| `pyproject.toml` | Project metadata, Python 3.12 requirement and pinned modelling/notebook dependencies. |
| `requirements.txt` | Matching direct dependency pins for pip users. |
| `uv.lock` | Dependency lockfile for the uv setup. |
| `.python-version` | Python version selection for environment tooling. |
| `.gitignore` | Excludes environments, caches, generated models/results, `planning/` and `tests/`. Ignoring tests is worth revisiting when adding verification to the repository. |
| `models/` | Generated estimators; currently contains the classifier and a tracked placeholder. |
| `outputs/` | Generated results; currently contains one classification report, one scoreboard row and a tracked placeholder. |

The audit identifies purchase frequency and short gaps as important return signals, and historical spending as an important spend signal. It finds considerable redundancy among monetary totals and rolling windows. It also motivates dropping `random_noise`, handling previously unseen categories and distinguishing structurally absent history from injected missingness. The current numeric pipeline uses median imputation plus missingness indicators for both types; the notebook's suggestion of neutral structural fills has not been implemented separately.

## 6. Verified current state

### Existing local classifier result

Run `20261116_133434`, random forest, threshold 0.5:

| Metric | Result |
|---|---:|
| Five-fold mean F1 | 0.7431 |
| Fold F1 standard deviation | 0.0233 |
| Always-return baseline mean F1 | 0.7112 |
| Mean training minus validation F1 | +0.144 |
| Holdout F1 | 0.7661 |
| Holdout baseline F1 | 0.753 |
| Holdout precision | 0.794 |
| Holdout recall | 0.740 |
| Holdout accuracy | 0.727 |
| Holdout ROC AUC | 0.804 |

The model improves on the baseline, but the holdout F1 gain is modest (about 0.013 absolute). The training/validation gap suggests overfitting worth investigating. The figures are from the existing report and scoreboard; training was not repeated for this review.

Current partitions are 5,175 development / 1,294 holdout classification rows and 2,733 development / 692 holdout regression rows. The saved classifier was fitted on all 6,469 usable classification rows.

### Checks performed for this report

- Parsed `main.py` and every Python file in `src` successfully.
- Imported pipeline modules and ran CLI help successfully in the existing Python 3.12.7 environment.
- Confirmed raw dataset shapes, customer-key uniqueness, target counts and cleaned split sizes directly.
- Checked all configured rules on the prepared classification training rows: no remaining violations.
- Loaded the saved classifier; its saved configuration matches the current configuration.
- Produced finite classifier probabilities for all 986 test rows, with 660 positive decisions at threshold 0.5. These were in-memory checks; no submission file was written.
- Checked that the first test purchase receives the same probability alone and in the batch. This is a limited consistency check, not exhaustive proof for all inputs.
- Tried both example client calls: both stop with the explicit missing-regressor message.

### What is not yet evidenced locally

There is no saved `regressor.joblib`, regression run report, model-comparison table, generated submission or regression prediction file. There is no evidence here of a Kaggle submission or score. Generated files are ignored by Git, so their absence locally does not establish what teammates have run elsewhere.

No automated test suite is present in the inspected tree. Full retraining, all-model comparisons, regression inference, complete batch export, notebook execution, reproducibility across repeated runs and a clean installation on another machine were not verified. Dependency imports work in this environment; that does not establish that every estimator fits successfully.

## 7. Risks and unfinished requirements

1. **The validation horizon needs attention before further selection.** `TimeSeriesSplit` uses no gap, and holdout splitting is also immediate. A training purchase immediately before validation can have a target determined by purchases during the validation period. That makes the assessment optimistic relative to a deployment cutoff where those labels are not yet observable. Review date-based splits that exclude training purchases whose 90-day outcome window overlaps validation. The current holdout boundary is 17 June 2023 on both sides; four of five classification CV boundaries also share a date, with the remaining boundary separated by only three days. Keep dates together and use a calendar-day gap rather than assuming a fixed number of rows represents 90 days.

2. **The holdout is not mechanically protected from repeated selection.** `compare` leaves it out, but every `train` command scores it again. Model and threshold choices should use development data only. If existing holdout results influence choices, reserve a fresh final assessment period.

3. **Algorithm coverage is implemented, but optimization evidence is incomplete.** No saved comparison tables establish the best candidates. There is no systematic hyperparameter search, threshold search or fold-fitted feature selection. Dropping noise and identifiers is useful initial filtering, but does not complete the feature-selection requirement.

4. **Regression and the combined workflow still need verification.** The chosen CatBoost configuration has no local result. Conditional RMSE/MAE/R² should be reported on actual returners with observed targets, alongside classifier false negatives and any end-to-end business evaluation. Missing returner targets may affect representativeness and should be discussed.

5. **Regression comparison direction has a configuration trap.** `compare()` sorts every regression selection metric ascending. This is correct for current RMSE and MAE, but would rank R² backwards if chosen. Use an explicit metric direction if expanding configuration choices.

6. **Regression scoring differs slightly from delivered predictions.** Assessment uses raw regressor output, whereas inference clips negative predictions to zero and rounds to cents. Make the evaluated prediction policy consistent with the delivered policy, or quantify the difference.

7. **Client robustness needs clearer semantics and checks.** An omitted prior count is treated as first purchase even if the caller simply does not know it. Invalid numeric strings or malformed dates can raise library errors. Neither saved-model compatibility across the two tasks nor customer-join cardinality is explicitly validated. Exercise minimal, missing-field, unknown-category and invalid-input cases.

8. **Explainability and the additional objective are missing.** No global or individual explanations for either task, SHAP/LIME dependency, dashboard or implemented additional analysis was found. These account for 3 of 20 points in the brief.

9. **Final reproducibility and delivery evidence is incomplete.** The README's identical-output statement has not been established here; run identifiers and timings naturally vary. Preserve the exact final settings, source/data identifiers and assessment evidence. Model names alone are insufficient to reproduce a choice, and ignored output reports will not travel with a clone unless deliberately included in the handoff.

## 8. Recommended next steps, in order

### First: establish trustworthy assessment

1. Agree a date-based validation policy that respects the 90-day label horizon and keeps each purchase date on one side of a boundary. Apply it to CV and the final holdout. Confirm enough data remains for both classes and regression observations in every fold.
2. Keep one final assessment period separate from model, feature and threshold decisions. Rerun the baselines after changing validation; use these as the new reference scores.
3. Add focused verification for cleaning rules, target exclusion, join row preservation, split boundaries, unknown categories, single-row/batch consistency and output alignment. Remove the `tests/` ignore rule if tests are added there.

### Then: complete and improve the two models

4. Run and retain comparison tables for all six algorithms per task under the agreed assessment policy. Document the selected models using F1 for classification and RMSE with MAE/R² for regression.
5. Evaluate a small, justified set of preprocessing and feature changes: structural-history handling, redundant-feature removal, purchase/spend trends, average basket value, safe discount/margin ratios and appropriate scaling. Fit learned transformations and feature selection inside CV.
6. Tune shortlisted models and the classification threshold using development folds. Revisit log-spend transformation with saved RMSE/MAE evidence; a more symmetric target is not by itself proof of better EUR predictions.
7. Train and save both selected models. Run batch prediction and both client examples; exercise minimal input and new categorical levels. Verify 986 unique test IDs in the correct order, binary labels, expected columns, nonnegative spend and exact consistency between output files.

### Finally: finish explanation, evidence and delivery

8. Add global and individual explanations for both models, covering at least a new customer, returning customer and useful error cases. Explain the role of history, customer attributes and missingness in business terms.
9. Choose one additional objective with a clear evaluation, such as time/cohort error analysis, calibrated return probabilities or spending uncertainty. A focused result is enough if implemented and discussed well.
10. Perform a clean installation and terminal run using documented commands, ideally on Mac and Windows. Test repeated predictions with the same artifacts and record exactly what is reproducible.
11. Submit the pipeline-generated classification file to Kaggle, record the score and preserve final configuration/results in a tracked handoff location. Prepare the required ZIP and ensure every member can explain the entire workflow for the defence.

### Commands for the current implementation

Run from the repository root after setup. These commands were not all executed during this review; they describe the current workflow, and results should be regenerated after validation changes.

```bash
uv sync
uv run python main.py compare --task classification
uv run python main.py compare --task regression
# Review comparison results and set the two selected models in config.yaml.
uv run python main.py train --task classification
uv run python main.py train --task regression
uv run python main.py predict
uv run python main.py client --input examples/returning_customer.json
uv run python main.py client --input examples/new_customer.json
```

For the immediate local missing-model issue, `uv run python main.py train --task regression` creates the missing regressor; that alone does not address the assessment and remaining deliverable gaps above.
