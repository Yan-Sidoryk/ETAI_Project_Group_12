# Project Guidelines

**Deadline: 22 December, 18:00**

## The problem

*Everything & Then Some, Ltd.* is a wholesale company (tools, hardware, lighting, automotive care, household goods, etc.). For **any new purchase**, the tool must answer:

1. **Classification:** will this customer buy again within the **next 90 days**? (`repeat_purchase_90d`)
2. **Regression:** how much (EUR) will they spend in those 90 days? (`spend_next_90d`)

Business use: customers unlikely to return get incentives; high expected spenders get retention / account management.

## Objectives

1. **Data exploration & cleaning** — find inconsistencies, fix errors (invalid values, duplicates, ...).
2. **Classification pipeline optimization** — consistent model-assessment strategy; optimize the full pipeline: pipeline-dependent preprocessing (imputation, encoding, scaling, outliers), feature engineering, feature selection, algorithms, hyperparameters.
3. **Kaggle competition** — predict every row of `test.csv`, submit in `sample_submission.csv` format.
4. **Regression pipeline optimization** — for customers the classifier predicts will return, predict spend. Approach and how it combines with classification is our choice, but **both predictions must be coherent**.
5. **Deployment** — a terminal-runnable pipeline per task + a client interface that, for a new purchase, returns whether the customer buys again and how much.
6. **Explainability & open-ended** — explain both models (SHAP, LIME, ...), globally and per customer. Then something extra to stand out: dashboard, full UI, error analysis, ablation, stability over time, calibration, spend uncertainty, ...

## Data (`data/`)

| File | Content |
|---|---|
| `train.csv` | 6,534 purchase occasions with both targets |
| `test.csv` | 986 later purchase occasions, no targets |
| `clientes.csv` | Optional customer-level table, join on `customer_id` |
| `sample_submission.csv` | Kaggle format (`ID,repeat_purchase_90d`) + simple baseline |

Each row = one customer purchase occasion.

| Variable / group | Description |
|---|---|
| `ID` | Row id and Kaggle submission id |
| `customer_id` | Anonymized customer; join key for `clientes.csv` |
| `repeat_purchase_90d` | Classification target (1 = buys again within 90 days). Train only |
| `spend_next_90d` | Regression target, EUR spent in later purchases within 90 days. Train only. **Never use as a classification feature** |
| `purchase_date` | Anonymized date; order and intervals preserved |
| Current monetary values | Totals, tax, shipping, discounts, cost (`Total a Pagar`, `Total Bruto`, `Total Liquido`, `Total Imp_`, `Total Portes`, `Total Desc_ Global`, `Total Desc_ Linha`, `Custo`) |
| Current purchase details | `N_ Detalhes`, `invoice_count_current_purchase` |
| Current categoricals | `payment_type`, `salesperson`, `commercial_zone`, `transport_method`, `warehouse`, `document_series` |
| Historical timing | `days_since_previous_purchase`, `customer_tenure_days`, `prior_purchase_count`, `prior_mean_gap_days`, `prior_std_gap_days` |
| Historical value | `prior_total_spend`, `prior_mean/std/min/max_purchase_value` |
| Rolling frequency | `purchase_count_{30,90,180,365}d` |
| Rolling spend | `spend_{30,90,180,365}d` |
| `random_noise` | Extra numeric field — **assess whether it carries information** |

## Deliverables

1. **`Group12_ETAI_2627.zip`** with all code used to decide on and obtain the final solutions. Exploration notebooks only if relevant.
2. **GitHub repo** with the final pipelines, a `README.md` explaining how to run on **Mac and Windows**, a `config.yaml` and a requirements file.
3. **Final Kaggle submission**, matching the pipeline output.

### Pipeline requirements

Must run entirely from the terminal **without changing any code**, and contain:

- **Training per task**, raw data → saved model. Predicting never retrains.
- **Batch prediction** for `test.csv` writing:
  - Kaggle submission: `ID, repeat_purchase_90d`
  - Regression predictions: `ID, repeat_purchase_90d, spend_next_90d` for rows predicted to return
- **Client interface**: for one new purchase → repeat-purchase probability + decision + expected 90-day spend. Must work for **new and returning customers**.
- **`config.yaml`** with every choice that can change without editing code (paths, seed, models, hyperparameters).

## Grading (out of 20)

| Criterion | % | Points |
|---|---|---|
| Pipeline structure & code quality | 15 | 3 |
| Data exploration & cleaning | 15 | 3 |
| Classification pipeline optimization | 25 | 5 |
| Regression pipeline optimization | 20 | 4 |
| Explainability & open-ended | 15 | 3 |
| Deployment | 10 | 2 |

**Breakdown**

- **Pipeline structure & code quality (3):** repo organization, readability, documentation, reproducibility.
- **Data exploration & cleaning (3):** meaningful insights; clear steps and rationale for cleaning (errors, inconsistencies, invalid values, duplicates).
- **Classification (5):**
  - Kaggle performance — 1
  - Model assessment strategy, **min. 5 algorithms** — 1
  - Pipeline-dependent preprocessing & feature engineering — 1
  - Feature selection strategy & implementation — 0.75
  - Hyperparameter tuning — 0.75
  - Choice of metric(s) & interpretation — 0.5
- **Regression (4):**
  - Target exploration & modelling approach — 0.75
  - Model assessment strategy, **min. 4 algorithms** — 0.75
  - Preprocessing, feature engineering, feature selection & tuning — 1.5
  - Coherence between classification and regression — 0.5
  - Test performance (RMSE, MAE, R² on customers who actually returned) — 0.5
- **Explainability & open-ended (3):**
  - Explainability of both models, global + individual — 1.5
  - Formulation & difficulty of additional objectives — 0.5
  - Correctness of implementation — 0.5
  - Discussion of results & business meaning — 0.5
- **Deployment (2):**
  - Pipeline runs from terminal as described — 1
  - Client interface works for new and returning customers — 1

**Timing:** late up to 3 days at −1 point/day; after that not accepted. Early delivery: +0.15/day, max +1.

## Rules (Final Notes)

1. **No leakage:** features must be computable at purchase time. Each test row is predicted using **only that row, the training data and `clientes.csv`** — never other test rows.
2. Automated profiling (e.g. ydata-profiling) only counts if an insight is visibly acted upon.
3. **Allowed libraries:** scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, LIME, imbalanced-learn, TensorFlow, Keras, PyTorch. **Forbidden:** Lazy Predict and AutoML-like tools (Featuretools, TSFresh, ...).
4. Anything not in the zip/repo is not considered.
5. No theoretical explanations of topics covered in class.
6. Cite peer-reviewed sources only (no Medium, TowardsDataScience, ...). Never cite ChatGPT / LLMs.
7. **They will clone and run the repo — every command must run without interruption.** Failing this is severely penalized.
8. Code goes through plagiarism and AI-generation checks.
9. Must submit to Kaggle to get those points.
10. Grading has a comparative component against other groups.

## Defense

- 40 minutes per group. Presentation order of members is random.
- Every member must explain what was done and why; any member can be asked about any part of the code.
- **Not attending = 0.** Individual grades can differ without limit.

**Reminders:** don't include anything you can't explain. Finished is better than perfect.
