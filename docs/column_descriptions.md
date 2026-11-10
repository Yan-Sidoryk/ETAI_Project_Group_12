# Data column descriptions

This dictionary describes the raw CSVs in `data/`, inspected on 10 November 2026. Business definitions follow [the project guidelines](guidelines.md). Where the brief does not define a field precisely, the description is an interpretation of its name and observed values. No data has been cleaned for this reference.

## Files and relationships

| File | Rows | Columns | Row represents |
|---|---:|---:|---|
| `train.csv` | 6,534 | 40 | One customer purchase occasion, with classification and regression targets. |
| `test.csv` | 986 | 38 | One later purchase occasion to predict; contains the same 38 input columns as train. |
| `clientes.csv` | 1,143 | 13 | One customer, optionally joined to purchases using `customer_id`. |
| `sample_submission.csv` | 986 | 2 | One classification prediction per test `ID`. |

- `ID` is unique within each purchase file and does not overlap between train and test. It identifies a purchase row, not a customer.
- `customer_id` repeats across purchases and is unique in `clientes.csv`. Use a many-to-one left join on this key. Every customer in train and test has a matching customer record.
- Train contains 610 distinct customers; test contains 206, including 22 absent from train.
- Nonmissing purchase dates span **2015-07-11 to 2025-02-15** in train and **2025-05-24 to 2026-10-06** in test. Dates are anonymized, with order and intervals preserved.

## Purchase columns (`train.csv` and `test.csv`)

Missing counts below refer to the raw files and are shown as **train / test**. Types are logical types: counts may load as floating-point numbers because the CSV contains blanks. Money is in **EUR**. Historical fields describe purchases before the current purchase; rolling fields refer to the preceding number of days. Exact window boundary inclusivity and statistical conventions are not specified in the brief.

### Identifiers and date

| Column | Logical type | Description | Missing: train / test |
|---|---|---|---:|
| `ID` | Integer identifier | Unique purchase-row identifier; preserve for prediction output and submission alignment. | 0 / 0 |
| `customer_id` | Integer identifier | Anonymized customer identifier and join key for `clientes.csv`. Its numeric magnitude has no stated business meaning. | 0 / 0 |
| `purchase_date` | Date | Date of the current purchase, stored as `YYYY-MM-DD` text. Parse as a date for time-based features and validation. | 65 / 0 |

### Current purchase amounts and details

Portuguese column names are retained exactly as supplied. The translations below describe their apparent accounting meaning; the brief does not provide the full accounting calculation for each total.

| Column | Logical type / unit | Description | Missing: train / test |
|---|---|---|---:|
| `Total a Pagar` | Decimal, EUR | Total amount payable for the current purchase. | 65 / 0 |
| `Total Bruto` | Decimal, EUR | Gross purchase amount before discounts and tax. | 65 / 0 |
| `Total Liquido` | Decimal, EUR | Net purchase amount after discounts, before tax. | 65 / 0 |
| `Total Imp_` | Decimal, EUR | Total tax amount, not a tax percentage. | 65 / 0 |
| `Total Portes` | Decimal, EUR | Shipping or carriage charge. | 65 / 0 |
| `Total Desc_ Global` | Decimal, EUR | Overall purchase discount amount. | 65 / 0 |
| `Total Desc_ Linha` | Decimal, EUR | Total discount amount applied at line level. | 65 / 0 |
| `Custo` | Decimal, EUR | Recorded cost associated with the current purchase. Exact cost basis is unspecified; zero values occur. | 65 / 0 |
| `N_ Detalhes` | Integer count | Number of detail lines in the current purchase; does not necessarily equal the number of units bought. | 65 / 0 |
| `invoice_count_current_purchase` | Integer count | Number of invoices grouped into the current purchase occasion. Observed range: 1–5 in train and 1–4 in test. | 65 / 0 |

### Current purchase categories

These fields contain anonymized text labels. Numbers embedded in labels identify categories and should not be interpreted as an ordering.

| Column | Logical type | Description | Missing: train / test |
|---|---|---|---:|
| `payment_type` | Category | Payment method or payment terms, e.g. `Maybe Tomorrow Payment 10`. Exact underlying methods are not disclosed. | 260 / 30 |
| `salesperson` | Category | Sales representative associated with the purchase, e.g. `VLOOKUP Master 9`. | 259 / 30 |
| `commercial_zone` | Category | Sales territory associated with the purchase, e.g. `Decorative Dashboard Zone 4`. Distinct from the customer-table field `commercial_region`. | 65 / 0 |
| `transport_method` | Category | Delivery or transport method, e.g. `Maybe Tomorrow Delivery 61`. | 261 / 30 |
| `warehouse` | Category | Warehouse associated with the purchase, e.g. `Final Excel Warehouse 1`. | 260 / 30 |
| `document_series` | Category | Document or invoice series, e.g. `Ctrl Z Series 2`. | 65 / 0 |

### Customer history: timing and lifetime value

| Column | Logical type / unit | Description | Missing: train / test |
|---|---|---|---:|
| `days_since_previous_purchase` | Integer, days | Time from the customer's preceding purchase to the current purchase. Undefined for a first purchase. Negative values occur and need validation. | 861 / 52 |
| `customer_tenure_days` | Integer, days | Customer tenure at the current purchase, interpreted as elapsed days since the first recorded purchase. Zero can represent a first purchase. | 65 / 0 |
| `prior_purchase_count` | Integer count | Number of purchases before the current purchase. Zero indicates no prior purchases in the recorded history. | 65 / 0 |
| `prior_total_spend` | Decimal, EUR | Cumulative spending across the customer's prior purchases. | 65 / 0 |
| `prior_mean_purchase_value` | Decimal, EUR | Mean value of prior purchases. For valid records with a positive prior count, expected to equal `prior_total_spend / prior_purchase_count`. | 862 / 52 |
| `prior_std_purchase_value` | Decimal, EUR | Standard deviation of prior purchase values. Can be undefined when history is insufficient; the standard-deviation convention is unspecified. | 1,113 / 46 |
| `prior_min_purchase_value` | Decimal, EUR | Smallest prior purchase value. Undefined when there are no prior purchases. | 668 / 22 |
| `prior_max_purchase_value` | Decimal, EUR | Largest prior purchase value. Undefined when there are no prior purchases. | 668 / 22 |
| `prior_mean_gap_days` | Decimal, days | Mean interval between prior purchases. Requires at least two prior purchases. Negative values occur and need validation. | 1,308 / 76 |
| `prior_std_gap_days` | Decimal, days | Standard deviation of intervals between prior purchases. Can be undefined when there are too few intervals; the standard-deviation convention is unspecified. | 1,478 / 69 |

### Customer history: rolling windows

Zero is a valid count or spend when the customer made no purchases in the window. Missing values should be distinguished from zero.

| Column | Logical type / unit | Description | Missing: train / test |
|---|---|---|---:|
| `purchase_count_30d` | Integer count | Number of prior purchases in the preceding 30 days. | 65 / 0 |
| `spend_30d` | Decimal, EUR | Spending on prior purchases in the preceding 30 days. | 65 / 0 |
| `purchase_count_90d` | Integer count | Number of prior purchases in the preceding 90 days. | 65 / 0 |
| `spend_90d` | Decimal, EUR | Spending on prior purchases in the preceding 90 days. Historical input, distinct from the future target `spend_next_90d`. | 65 / 0 |
| `purchase_count_180d` | Integer count | Number of prior purchases in the preceding 180 days. | 65 / 0 |
| `spend_180d` | Decimal, EUR | Spending on prior purchases in the preceding 180 days. | 65 / 0 |
| `purchase_count_365d` | Integer count | Number of prior purchases in the preceding 365 days. | 65 / 0 |
| `spend_365d` | Decimal, EUR | Spending on prior purchases in the preceding 365 days. Contains missing, negative and extreme values requiring validation. | 260 / 30 |

### Additional numeric input

| Column | Logical type | Description | Missing: train / test |
|---|---|---|---:|
| `random_noise` | Decimal, unitless | Extra numeric field with no specified business meaning. The brief asks the team to assess whether it carries predictive information. | 65 / 0 |

### Targets (train only)

| Column | Logical type / unit | Description | Missing: train |
|---|---|---|---:|
| `repeat_purchase_90d` | Binary integer | Classification target: `1` means the customer buys again within the next 90 days; `0` means they do not. | 0 |
| `spend_next_90d` | Decimal, EUR | Regression target: total spending on later purchases within the next 90 days, excluding the current purchase. | 3,109 |

The raw training file contains 3,654 positive and 2,880 negative classification labels. All 2,880 non-returners have a blank spend target. An additional **229 returners** have a blank spend target. Therefore, a missing spend target cannot universally be interpreted as zero. The 3,425 observed spend targets are positive, ranging from EUR 1.27 to EUR 25,654.33.

Both targets describe future outcomes and must be excluded from input features. In particular, never use `spend_next_90d` to predict `repeat_purchase_90d`.

## Customer columns (`clientes.csv`)

Geography, address and company fields use anonymized labels. Geographic descriptions below reflect their names; real locations cannot be recovered from these labels. These are customer-level attributes, not current-purchase details.

| Column | Logical type | Description | Missing / 1,143 |
|---|---|---|---:|
| `customer_id` | Integer identifier | Unique anonymized customer identifier; joins to both purchase files. | 0 |
| `country_or_region` | Category | Customer country or region, e.g. `Country CT11: Land of The Final Version`. | 0 |
| `commercial_region` | Category | Customer commercial region, e.g. `Region RG37: We Will See Later`. Distinct from purchase-level `commercial_zone`. | 114 |
| `island` | Category | Customer island label, where recorded. A blank does not establish whether the customer lives on the mainland. | 578 |
| `municipality` | Category | Customer municipality label. | 159 |
| `city` | Category | Customer city label. | 546 |
| `postal_area` | Category | Customer postal-area label. | 71 |
| `postal_code` | Category / text | Anonymized postal-code label, e.g. `Postal code PC828: #REF! Delivery Route`. Treat as text, not a numeric postal code or spreadsheet formula. | 170 |
| `street` | Category / text | Anonymized street label. | 598 |
| `address` | Category / text | Anonymized address label; 1,120 distinct nonmissing values, so it is almost customer-specific. | 6 |
| `customer_type` | Unknown category | Intended customer-type field; all values are blank, so neither its categories nor intended coding can be determined. Some readers infer a floating-point type solely because it is empty. | 1,143 |
| `company_name` | Category / text | Anonymized company name; 1,141 distinct values, so it is almost customer-specific. | 0 |
| `customer_since_year` | Integer year | Year the customer relationship began, as indicated by the field name. Observed values span 2015–2026; the brief does not specify whether this means registration or first purchase. | 0 |

## Submission columns (`sample_submission.csv`)

| Column | Logical type | Description | Missing / 986 |
|---|---|---|---:|
| `ID` | Integer identifier | Test purchase-row identifier. The supplied template matches test IDs in the same order. | 0 |
| `repeat_purchase_90d` | Binary integer | Example baseline prediction to replace with the model's predicted class (`0` or `1`). These values are not ground-truth test labels. | 0 |

The Kaggle submission has exactly these two columns. Regression predictions belong in a separate output as described in the project guidelines.

## Interpretation and data-quality notes

- Many train input columns have 65 missing values. There are 65 train rows with all 38 inputs except `ID` and `customer_id` missing. These are incomplete purchase records, rather than ordinary missing history.
- Missing history can also be structural: no prior purchases means no prior minimum, maximum or mean; too little history can leave standard deviations and gap statistics undefined. Separate this case from missing values on customers with sufficient history.
- `days_since_previous_purchase` and `prior_mean_gap_days` each contain `-4` in 16 train rows and 2 test rows. Negative elapsed time is invalid under these definitions; treat this code as requiring cleaning rather than as a valid interval.
- `prior_mean_purchase_value` and `spend_365d` contain negative amounts and values of `1,000,000,000`. `prior_max_purchase_value` also has unusually large values. Validate these against prior counts, totals, minima and maxima before using them; observed raw ranges are not valid business limits.
- Categorical levels differ between train and test. Encoders must handle previously unseen labels as well as blanks.
- `customer_type` provides no observed information. Almost customer-specific fields such as `company_name` and `address` need care because they can encourage memorization.
- For modeling, use only information available at the current purchase. The project explicitly restricts each test prediction to that row, training data and `clientes.csv`; do not build features from other test rows.

## Sources

- Raw data: [`train.csv`](../data/train.csv), [`test.csv`](../data/test.csv), [`clientes.csv`](../data/clientes.csv), and [`sample_submission.csv`](../data/sample_submission.csv). Dimensions, missing counts, date ranges and target counts above were checked directly against these files.
- Business definitions and prediction constraints: [project guidelines](guidelines.md).
- Further exploration: [data audit notebook](../notebooks/01_data_audit.ipynb). This dictionary reports raw-file counts independently; notebook figures may use a filtered population.
