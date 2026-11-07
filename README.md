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
