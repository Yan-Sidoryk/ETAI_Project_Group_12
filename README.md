# ETAI Project - Group 12

"Everything & Then Some": predict whether a customer buys again within 90 days, and how much they spend.
See [docs/guidelines.md](docs/guidelines.md) for the full brief.

## Structure

```
main.py         entry point: compare | train | predict | client
config.yaml     paths, seed, validation, preprocessing, models, hyperparameters, outputs
diagnosis.yaml  data-cleaning rules (ranges, placeholder values, consistency checks and their fixes)
src/
  data.py           load config.yaml + diagnosis.yaml, read the CSVs
  cleaning.py       rule checks and repairs from diagnosis.yaml; training-only empty-row removal
  features.py       prepare(): clean, join clientes.csv, add_features(); split X and y
  preprocessing.py  imputation, scaling, one-hot encoding (fitted inside the model pipeline)
  model.py          model registry per task, baseline ("floor") models, build_pipeline()
  evaluate.py       time-ordered hold-out and CV folds with a calendar gap, metrics, CV reports
  inference.py      load saved models; shared classify / predict_spend for predict and client
  results.py        scoreboard and run reports
tests/          integration tests for the four commands
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
