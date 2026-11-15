"""Saving each run: its scores to the scoreboard, and its full report to disk."""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, report_text: str) -> str:
    """Writes the run's full report to `results_dir/run_<run>.txt` and returns the path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return path
