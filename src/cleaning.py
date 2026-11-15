"""
Rule-based cleaning, driven by diagnosis.yaml.

`check_rules` counts rule violations; `clean_dataset` repairs them. Both work row by row and
learn nothing from the data, so they give the same result on the training file, the test file
or a single new purchase. `drop_empty_rows` is the only function that removes rows, and is
for training data only.
"""
import numpy as np
import pandas as pd


def _violations(df: pd.DataFrame, rule: dict) -> pd.Series:
    """True for the rows of `df` that break a consistency rule (among the rows its `where` selects)."""
    bad = df.eval(rule["violation"], engine="python").fillna(False).astype(bool)
    if rule.get("where"):
        bad &= df.eval(rule["where"], engine="python").fillna(False).astype(bool)
    return bad


def check_rules(df: pd.DataFrame, diagnosis: dict) -> pd.Series:
    """Number of rows breaking each range, sentinel and consistency rule."""
    counts = {}

    for col, bounds in diagnosis.get("ranges", {}).items():
        if col not in df.columns:
            continue
        if bounds.get("min") is not None:
            counts[f"{col}_below_min"] = df[col].lt(bounds["min"]).sum()
        if bounds.get("max") is not None:
            counts[f"{col}_above_max"] = df[col].gt(bounds["max"]).sum()

    for col, rule in diagnosis.get("sentinels", {}).items():
        if col in df.columns:
            counts[f"{col}_sentinel"] = df[col].ge(rule["min_inclusive"]).sum()

    for name, rule in diagnosis.get("consistency", {}).items():
        counts[name] = _violations(df, rule).sum()

    return pd.Series(counts, dtype="Int64")


def clean_dataset(df: pd.DataFrame, diagnosis: dict) -> pd.DataFrame:
    """
    Repairs the errors found in the data audit and returns a copy with the same rows.
    The order matters: sign flips are repaired (step 2) before negative values are removed (step 3).

      1. placeholder values (1e9)            -> NaN
      2. sign-flipped values                 -> absolute value
      3. values outside their allowed range  -> NaN   (this catches the -4 code)
      4. values contradicting other columns  -> NaN   (rules with `set_nan`)
      5. columns that follow from others     -> recomputed
    """
    out = df.copy()

    for col, rule in diagnosis.get("sentinels", {}).items():
        if col in out.columns:
            out[col] = out[col].mask(out[col] >= rule["min_inclusive"])

    for col in diagnosis.get("abs_value", []):
        if col in out.columns:
            out[col] = out[col].abs()

    for col, bounds in diagnosis.get("ranges", {}).items():
        if col not in out.columns:
            continue
        if bounds.get("min") is not None:
            out[col] = out[col].mask(out[col] < bounds["min"])
        if bounds.get("max") is not None:
            out[col] = out[col].mask(out[col] > bounds["max"])

    for rule in diagnosis.get("consistency", {}).values():
        if "set_nan" in rule:
            out.loc[_violations(out, rule), rule["set_nan"]] = np.nan

    for col, rule in diagnosis.get("recompute", {}).items():
        rows = out.eval(rule["where"], engine="python").fillna(False).astype(bool)
        out.loc[rows, col] = out.eval(rule["value"], engine="python")[rows]

    return out


def drop_empty_rows(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Training data only: drops the rows where every feature is missing.
    Never call it on data to predict, where every row needs a prediction.
    """
    return df[~df[feature_cols].isna().all(axis=1)]
