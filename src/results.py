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


def selection_report(fold_scores: pd.DataFrame, pipeline) -> str:
    """The feature selection of a `train` run as text: how often each feature was kept across the CV folds,
    the reference column's rank per fold, and the saved model's full importance table.
    Empty when the pipeline does not select features."""
    if "select" not in pipeline.named_steps:
        return ""
    final = pipeline.named_steps["select"]
    table = final.importances_
    n_folds = len(fold_scores)
    kept_in = pd.Series({name: sum(name in kept for kept in fold_scores["kept"]) for name in table.index})
    stable, never = kept_in[kept_in == n_folds].index, kept_in[kept_in == 0].index
    unstable = kept_in[(kept_in > 0) & (kept_in < n_folds)].sort_values(ascending=False)
    lines = [
        f"Feature selection (fitted on each fold's training rows only; final model on all rows)",
        f"  kept per fold: {', '.join(str(len(kept)) for kept in fold_scores['kept'])} of {len(table)}"
        f"   |   final model: {len(final.selected_)}",
        f"  kept in every fold ({len(stable)}): {', '.join(stable)}",
        f"  kept in some folds ({len(unstable)}): " + ", ".join(f"{n} {k}/{n_folds}" for n, k in unstable.items()),
        f"  never kept ({len(never)}): {', '.join(never)}",
    ]
    if final.reference in table.index:
        lines.append(f"  {final.reference}: rank {table.loc[final.reference, 'rank']} of {len(table)} in the final "
                     f"model, kept in {kept_in[final.reference]} of {n_folds} folds")
    return "\n".join(lines) + "\n\nFinal model importance:\n" + table.to_string(float_format=lambda v: f"{v:.4g}")
