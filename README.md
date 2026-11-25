# ETAI Project - Group 12

"Everything & Then Some": predict whether a customer buys again within 90 days, and how much they spend.
See [docs/guidelines.md](docs/guidelines.md) for the full brief.

## Structure

```
main.py         entry point: compare | experiment | tune | train | predict | explain | client
config.yaml     paths, seed, validation, preprocessing, models, hyperparameters, outputs
diagnosis.yaml  data-cleaning rules (ranges, placeholder values, consistency checks and their fixes)
src/
  data.py           load config.yaml + diagnosis.yaml, read the CSVs
  cleaning.py       rule checks and repairs from diagnosis.yaml; training-only empty-row removal
  features.py       prepare(): clean, join clientes.csv, add_features(); split X and y
  selection.py      per-fold feature selection, the first pipeline step (permutation importance + correlation)
  preprocessing.py  configurable imputation, scaling, one-hot encoding (fitted inside the model pipeline)
  model.py          model registry per task (including soft voting), baseline ("floor") models, build_pipeline()
  evaluate.py       time-ordered hold-out and CV folds with a calendar gap, metrics, threshold search, corrected t-test
  experiments.py    preprocessing grid search, summaries, fold scores
  tuning.py         Optuna hyperparameter search on the development folds
  inference.py      load saved models; shared classify / predict_spend for predict and client
  explain.py        SHAP explanations of the saved models (global and per purchase)
  results.py        scoreboard, run reports, feature-selection report
tests/          preprocessing, experiments, evaluation, tuning and command integration tests
examples/       client inputs: returning_customer.json, new_customer.json
data/           raw input CSVs (train, test, clientes, sample_submission)
docs/           project brief (PDF + guidelines.md) and column descriptions
notebooks/      01 data audit (exploration and cleaning rationale), 02 feature selection (method walkthrough)
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
python main.py tune --task classification --model xgboost   # optional: Optuna search; prints params to copy
python main.py train --task classification   # saves models/classifier.joblib
python main.py train --task regression       # saves models/regressor.joblib
python main.py predict                       # needs both models
python main.py client --input examples/returning_customer.json   # one purchase, needs both models
python main.py explain --task classification   # SHAP: outputs/shap_<task>_global.csv and _beeswarm.png
python main.py explain --task regression
```

`predict` writes two files:
- `outputs/submission.csv`: the Kaggle file, one row per test purchase with `repeat_purchase_90d`.
- `outputs/regression_predictions.csv`: predicted `spend_next_90d` for the purchases predicted to return.

`client` reads one purchase from a JSON file whose fields are the columns of `data/test.csv`, and prints the probability
of buying again, the decision and the expected 90-day spend (0 when the decision is "does not return").
It also lists the five features that moved the probability and the spend most (SHAP).
Any field can be left out, but a missing `prior_purchase_count` means "first purchase": leave out the history fields
and `customer_id` only for a genuinely new customer, as in `examples/new_customer.json`. For a returning customer,
fill in the history fields as `test.csv` provides them.

`compare` prints the task's table, best model first, and saves it to `outputs/comparison_<task>.csv`.
`train` saves the model named by `classification.model` / `regression.model` in `config.yaml`.

Each `train` run prints its cross-validation and hold-out scores against the floor model, adds a row to `outputs/runs.csv`
and writes the full report to `outputs/run_<id>.txt`.
Models, preprocessing, feature selection, metrics and the threshold are set in `config.yaml`; no code edits are needed.
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
raw rows ──> clean (diagnosis.yaml) ──> join clientes.csv ──> add_features ──> X ──> Pipeline(select, prep, model)
└──────────── prepare(): row by row, learns nothing ─────────────────────┘     └──── fitted, saved with joblib ────┘
```

- `prepare()` learns nothing from the data, so a test purchase uses only itself, the training data and `clientes.csv`
  (brief rule 1). Everything learned (feature selection, imputation, encoding, scaling) sits inside the
  scikit-learn `Pipeline`, so it is refitted in every CV fold and saved with the model.
- **Each row is one purchase**, scored on its own. A customer's past is seen only through the history columns of that
  row (`prior_*`, `spend_*d`, `purchase_count_*d`, gaps). `customer_id` is used only to join `clientes.csv`.
- **Hold-out:** the most recent 20% of training rows (from 17 Jun 2023), never used for a choice. Development rows
  end `validation.gap_days` (90) days before it, because a purchase's label looks 90 days ahead.
- **Cross-validation:** 5 time-ordered folds on the development rows, with the same 90-day gap before each validation
  block. Test purchases are all later than training, so shuffled folds would leak the future.
- **Saved model:** refitted on all usable training rows (hold-out included) after assessment.
- **Classification:** selected on F1, the Kaggle metric, at each model's best threshold; the saved model predicts
  "returns" when the probability is at least `classification.threshold` (0.41).
- **Regression:** trained only on returners with an observed spend (3,425 rows; 229 returners have none, so a
  missing spend is not zero).
- **Coherence:** spend is predicted only where the classifier says "returns"; the regression file holds exactly
  those IDs and the client shows 0 EUR otherwise.
- **Feature selection** is the first pipeline step (`src/selection.py`), fitted on each fold's training rows only and
  separately per task, so validation rows never influence which features are used. Fitted steps are cached in
  `outputs/cache` (the folds are fixed): a fold's selection is computed once, then reused by every model, rerun and
  tuning trial. **Delete `outputs/cache` after editing `src/selection.py` or `src/preprocessing.py`**, because the cache
  does not see code changes.
- **Not used as features:** IDs, `purchase_date` (splitting only), both targets, and the
  `clientes.csv` columns other than `country_or_region`, `commercial_region` and `customer_since_year`
  (mostly missing or almost unique per customer).

### Configuring preprocessing

`preprocessing` in `config.yaml` holds the shared settings; `classification.preprocessing` or
`regression.preprocessing` override them for one task (regression uses `encoder: onehot`). Each component accepts a
name or `{type: name, params: {...}}` with scikit-learn constructor parameters; examples are in `config.yaml`.

| Setting | Supported names | Defaults / behavior |
| --- | --- | --- |
| `scaler` | `standard`, `minmax`, `robust`, `maxabs`, `normalize`, `power`, `quantile`, `none` | Applied after numeric imputation; `normalize` normalizes each row, `power` defaults to Yeo-Johnson, `quantile` to a uniform distribution |
| `numeric_imputer` | `median`, `mean`, `most_frequent`, `constant`, `knn`, `none` | Missingness indicators enabled except for `none`; `constant` fills with 0 |
| `categorical_imputer` | `constant`, `most_frequent`, `none` | `constant` fills with `"missing"` |
| `encoder` | `onehot`, `ordinal` | One-hot ignores unseen levels; ordinal uses -1 for unseen levels |

`min_frequency` (20) groups rare category levels for both encoders; `encoder.params.min_frequency` overrides it
(`null` disables grouping). `none` skips a step and leaves missing values to the estimator. KNN imputation runs before
scaling, so feature units affect its neighbours. Saved models keep their fitted preprocessing: after editing the YAML,
run `train` again.

### Comparing preprocessing (`experiment`)

```bash
python main.py experiment --task classification
python main.py experiment --task regression --models ridge catboost   # optional: several configured models
```

Every combination of the lists in `preprocessing_experiment.candidates` is cross-validated on the same development
folds (the current settings are always included; duplicates removed). Feature selection is fitted once per fold with
the task's current preprocessing and shared by every candidate, so only the model's own preprocessing varies. The
hold-out is never scored and saved models are not touched. Each run writes `outputs/preprocessing_<task>_<timestamp>/`
with `summary.csv` (ranked: F1 highest first, RMSE lowest first; failed candidates at the bottom with their error)
and `folds.csv`; `outputs/preprocessing_<task>_summary.csv` always holds the latest run. Copy the chosen settings into
`config.yaml`, then `train`.

### Status (25 Nov 2026)

Done: cleaning, engineered features, gapped splits, 6 algorithms per task plus soft voting, per-fold feature
selection, preprocessing grid, threshold choice, statistical model comparison, Optuna tuning, regression loss
comparison, SHAP explanations, all seven commands, integration tests, reproducibility on Windows.

| Task | Model | CV (5 folds) | Floor | Hold-out |
|---|---|---|---|---|
| Classification (F1) | random forest (300 trees, `min_samples_leaf` 5), threshold 0.41 | 0.752 ± 0.017 | 0.712 | 0.791 (floor 0.753), ROC-AUC 0.808 |
| Regression (RMSE, EUR) | LightGBM (tuned, `lightgbm_tuned`) | 1,955 ± 178 | 2,472 | 2,528 (floor 3,289), MAE 1,273, R² 0.40 |

Kaggle (public leaderboard): 0.809 at threshold 0.5, 0.799 at 0.41.

**Engineered features** (19 Nov 2026, `add_features()`): recency (`overdue_ratio`, `purchase_rate`), trend
(`spend_trend_90d`, `count_trend_90d`), `value_vs_usual`, `margin_rate`, `discount_rate`, `value_per_line`,
`shipping_rate`. A zero denominator gives a missing value.

**Feature selection** (25 Nov 2026), per fold and per task (`feature_selection` in `config.yaml`). Choosing a drop list
on the development rows (notebook 02, 24 Nov) was leakage, because CV then validated on rows that helped choose the
features. The notebook now only walks through the method. Inside each fold's training rows:
- each raw column is ranked by permutation importance through a probe (preprocessing plus a random forest, 5 inner
  folds × 2 shuffles), so a categorical is judged as a whole and after preprocessing;
- a column is dropped when its importance is not above max(`random_noise`, `min_importance`), or when it is a weaker
  member of a numeric family with Spearman |r| >= 0.9;
- `min_importance` is 0 for classification (ROC-AUC) and 5 EUR of RMSE for regression.

Classification keeps 23-27 of 49 features per fold (11 in every fold, 29 in the final model); regression keeps 8-20
(4 in every fold, 19 in the final model). `random_noise` is dropped in every fold of both tasks. The first `train`
run takes about 4 min; repeated runs about 1 min (cache). CV with per-fold selection against all features, before
tuning:

| Classification (F1) | Selected | All | Regression (RMSE, EUR) | Selected | All |
|---|---|---|---|---|---|
| random forest | **0.748** | 0.744 | random forest | **1,976** | 1,967 |
| XGBoost | 0.733 | 0.729 | XGBoost | 2,005 | 1,993 |
| CatBoost | 0.729 | 0.727 | ridge | 2,006 | 2,216 |
| hist gradient boosting | 0.726 | 0.721 | LightGBM | 2,008 | 2,008 |
| LightGBM | 0.725 | 0.723 | CatBoost | 2,018 | 2,001 |
| logistic regression | 0.722 | 0.712 | hist gradient boosting | 2,054 | 2,049 |
| floor | 0.712 | 0.712 | floor | 2,472 | 2,472 |

Selection helps every classifier slightly and the linear models most (ridge 2,216 → 2,006); for the regression tree
models it is neutral, always within one fold std.

**Preprocessing** (25 Nov 2026): `robust` scaler, `median` imputer, `constant` for missing categories; encoder
`ordinal` for classification and `onehot` for regression. The grid (`experiment`, random forest) put all 24
classification setups within 0.007 F1 of each other (noise), while for regression one-hot beat ordinal in every fold,
by 43 EUR on average.

**Threshold** (25 Nov 2026): 0.41, chosen on training data only. `compare` reports each model's best threshold on the
pooled validation rows and F1 there; `train` adds the best threshold per fold and the hold-out F1 at the configured
threshold. Every model's best threshold is below 0.5 (0.31-0.41). Random forest per validation block, oldest to
newest: 0.38, 0.42, 0.36, 0.42, 0.42 (no drift). Pooled F1 is within 0.003 of its best from 0.38 to 0.42, and 0.5 is
lower in every fold (pooled 0.743 vs 0.752; hold-out 0.771 vs 0.791). The return rate per block (59, 54, 58, 55, 50%,
hold-out ~60%) shows no trend. On Kaggle 0.5 scored higher (0.809 vs 0.799): a difference the size of the fold noise,
not explained by the training data, so the threshold stays 0.41.

**Model selection** (25 Nov 2026): `compare` ranks classification by F1 at each model's best threshold and tests every
model against the best with the corrected paired t-test (Nadeau & Bengio) on the same 5 folds (`p_vs_best`).
- Classification: random forest 0.752; XGBoost 0.745 (p 0.20), CatBoost 0.742 (p 0.36) and hist gradient boosting
  (p 0.14) are not distinguishable from it; LightGBM (p 0.021), logistic regression (p 0.043) and the floor (p 0.032)
  are significantly worse.
- Regression: random forest 1,975; no model differs significantly (all p >= 0.18; even the floor, +497 EUR, has
  p 0.053). With 5 folds and noisy RMSE the test has little power: "not significant" means "cannot tell apart".

**Tuning** (25 Nov 2026): `tune` runs Optuna (40 trials, seeded) on the development folds; the configured params are
scored first as the score to beat. The best params are in `config.yaml` as `<name>_tuned` entries and were checked
with `compare` (best of 40 trials is slightly optimistic):
- Classification: random forest gained nothing (0.752 both). XGBoost 0.745 -> 0.750 with shallow trees (`max_depth`
  2), its train-validation gap falling from 0.139 to 0.012. Soft voting of random forest and tuned XGBoost
  (`soft_voting`): 0.749, no gain (p 0.61). Classification stays random forest.
- Regression: LightGBM 2,002 -> 1,955 (best CV, about 10x faster than random forest), random forest 1,975 -> 1,964;
  not significantly different (p 0.69). Regression saves `lightgbm_tuned`, chosen on CV (its hold-out is mixed
  against random forest: RMSE 2,528 vs 2,501, MAE 1,273 vs 1,298).

**Regression target and loss** (25 Nov 2026). Spend is right-skewed (median 959 vs mean 1,838 EUR) and higher in the
hold-out (mean 2,206). A few large spenders dominate RMSE: the top 1% of rows carry 26% of the squared deviation in
the development rows and 40% (7 rows, up to 25,654 EUR) in the hold-out, which is why hold-out RMSE (~2,500) is far
above CV (~1,955). The extremes are plausible business customers, so they are kept. Loss variants of
`lightgbm_tuned` (CV RMSE / MAE): squared error 1,955 / 1,242 (kept); Huber 2,016 / 1,253; tweedie 2,027 / 1,232;
gamma 2,050 / 1,228; log target 2,204 / 1,282. Every loss that downweights the extremes lowers MAE slightly and raises
RMSE, because RMSE is minimised by the mean; outlier removal and target clipping would act the same way. All
differences are within fold noise (p >= 0.07).

**Explainability** (25 Nov 2026, `src/explain.py`, SHAP TreeExplainer on the saved models). Rows pass through the saved
pipeline's own selection and preprocessing; one-hot levels and missing flags are summed back into their raw feature,
and base value + contributions equals each prediction exactly (checked in the tests).
- Global (`explain`, on the 986 test purchases): classification is driven by purchase frequency and recent activity
  (`purchase_count_365d`, `purchase_rate`, `spend_365d`, all raising the probability), then margin and discounts;
  longer gaps (`days_since_previous_purchase`, `prior_std_gap_days`) and newer customers (`customer_since_year`) lower
  it. Regression is driven by past spend (`spend_365d`, about 590 EUR average effect; `prior_total_spend`, typical
  order size) and the purchase month (seasonality).
- Per purchase (`client`): the five features that moved the probability (percentage points) and the spend (EUR) most
  from the average prediction, e.g. a new customer is pulled down by having no purchases in the last year.

### Next steps

1. **Open-ended objective** [open-ended 1.5]: error analysis, stability over time or spend uncertainty.
2. **Known issues** below.
3. **Before delivery:** final `train` for both tasks, `predict`, choose the final Kaggle submission, build
   `Group12_ETAI_2627.zip`.

### Known issues

| # | Issue | Fix |
|---|---|---|
| R1 | Every `train` run prints the hold-out score, so it can drift into choices | Choose models, features and the threshold from CV only |
| R2 | `compare` sorts every regression metric ascending: correct for RMSE/MAE, backwards for R² | Store a direction per metric |
| R3 | Scores use raw spend; delivered spend is clipped at 0 and rounded | Score the clipped values, or measure the difference |
| R4 | The `clientes.csv` join does not check `customer_id` is unique | Merge with `validate="many_to_one"` |
| R5 | `client` crashes on non-numeric values or bad dates; saved models are not checked against their task | Validate input with clear messages |
| R6 | Train-validation gap is positive for F1 overfitting but negative for RMSE (LightGBM: about −790 EUR) | Report it so positive always means overfitting |

### Open questions

1. **Coherence:** keep "spend only for predicted returners", or show probability × conditional spend in the client?
2. **Customer's past outcomes as a feature:** whether a customer returned after their earlier purchases is not in the
   row. It could be computed from earlier training labels (only labels whose 90-day window closed before the purchase,
   inside each fold, with the same delay as in test). Ask the professor whether it is allowed before building it.
3. **Client input:** JSON file now; an interactive prompt or small UI could count toward the open-ended part.

### Coding rules

- Only the libraries the brief allows: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, LIME,
  imbalanced-learn, TensorFlow, Keras, PyTorch. No AutoML-style tools.
- No leakage: nothing in `prepare()` may use other test rows; nothing learned may sit outside the `Pipeline`.
- Every command must run on a fresh clone; run the tests before pushing.
- Keep code small and explainable.
- A new dependency goes into both `pyproject.toml` and `requirements.txt` with the same pin.
- Update the status in readme if you change anything
