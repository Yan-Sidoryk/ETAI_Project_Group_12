# ETAI Project - Group 12

"Everything & Then Some": predict whether a customer buys again within 90 days, and how much they spend.
See [docs/guidelines.md](docs/guidelines.md) for the full brief.

## Structure

```
data/        raw input CSVs (train, test, clientes, sample_submission)
docs/        project brief (PDF + guidelines.md)
notebooks/   exploration and model-selection notebooks
src/         pipeline code (training, batch prediction, client interface)
models/      saved models (generated, git-ignored)
outputs/     submissions and predictions (generated, git-ignored)
config.yaml  paths, seed, models, hyperparameters
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

Each `train` run prints its cross-validation and hold-out scores against the floor model and adds a row to `outputs/runs.csv`.
Models, metrics and the threshold are set in `config.yaml`; no code edits are needed.
Runs are reproducible: the seed is `seed` in `config.yaml`, and two runs from the raw data give identical output files.
