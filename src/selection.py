"""
Feature selection, the first step of the model Pipeline. It is fitted on each fold's training rows only, so the
validation rows never influence which features are used, and it is cached with the pipeline (see build_pipeline).
notebooks/02_feature_selection.ipynb shows the same method step by step on the development rows.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

_PROBE = {"classification": (RandomForestClassifier, "roc_auc"),
          "regression": (RandomForestRegressor, "neg_root_mean_squared_error")}


class FeatureSelector(TransformerMixin, BaseEstimator):
    """
    Keeps the raw columns that carry signal and are not near-copies of a stronger column.

    fit(X, y), on the rows it is given only:
      1. importance: the rows are split `n_splits` ways; for each part a probe (the project's preprocessing + a
         random forest) is fitted on the other parts, and each raw column of the held-out part is shuffled
         `n_repeats` times. Importance = mean drop in score (ROC-AUC, or RMSE in EUR). A categorical column is
         shuffled as a whole, and every column is judged after preprocessing (imputation, missing flags, encoding).
      2. no signal: dropped when importance <= max(importance of `reference`, `min_importance`). `reference`
         (random_noise) is a known-useless column, and `min_importance` is the smallest score drop worth keeping
         (in ROC-AUC, or EUR of RMSE), so a kept feature must beat pure noise and a minimum real effect.
      3. duplicates: numeric columns with Spearman |r| >= `corr_threshold` form a family (complete linkage);
         only the most important member is kept.
    """

    def __init__(self, task, preprocessing, forest_params=None, n_splits=5, n_repeats=2,
                 corr_threshold=0.9, reference="random_noise", min_importance=0.0, random_state=None):
        self.task, self.preprocessing, self.forest_params = task, preprocessing, forest_params
        self.n_splits, self.n_repeats, self.corr_threshold = n_splits, n_repeats, corr_threshold
        self.reference, self.min_importance, self.random_state = reference, min_importance, random_state

    def fit(self, X: pd.DataFrame, y):
        forest, scoring = _PROBE[self.task]
        probe = Pipeline([("prep", build_preprocessor(self.preprocessing)),
                          ("model", forest(**(self.forest_params or {}), random_state=self.random_state))])
        y = pd.Series(np.asarray(y))
        drops = []
        # ponytail: shuffled inner folds, only to rank features; a time-ordered inner split would be stricter
        for fit_rows, val_rows in KFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X):
            fitted = clone(probe).fit(X.iloc[fit_rows], y.iloc[fit_rows])
            drops.append(permutation_importance(fitted, X.iloc[val_rows], y.iloc[val_rows], scoring=scoring,
                                                n_repeats=self.n_repeats, random_state=self.random_state).importances)
        drops = np.hstack(drops)
        table = pd.DataFrame({"importance": drops.mean(axis=1), "std": drops.std(axis=1)}, index=X.columns)

        noise = table.loc[self.reference, "importance"] if self.reference in table.index else -np.inf
        cutoff = max(noise, self.min_importance)
        table["kept"] = table["importance"] > cutoff
        table["reason"] = np.where(table["kept"], "", f"no signal (not above {cutoff:.4g})")
        if not table["kept"].any():  # never return an empty feature set
            best = table["importance"].idxmax()
            table.loc[best, ["kept", "reason"]] = [True, "kept: best of a no-signal set"]

        categorical = set(self.preprocessing["categorical_features"])
        numeric = [c for c in table.index[table["kept"]] if c not in categorical]
        if len(numeric) > 1:
            corr = X[numeric].corr(method="spearman").abs().fillna(0)
            distance = squareform(1 - corr.values, checks=False)
            family = fcluster(linkage(distance, method="complete"), t=1 - self.corr_threshold, criterion="distance")
            for members in pd.Series(numeric).groupby(family):
                best = table.loc[members[1], "importance"].idxmax()
                for name in members[1]:
                    if name != best:
                        table.loc[name, ["kept", "reason"]] = [False, f"duplicate of {best}"]

        self.importances_ = table.sort_values("importance", ascending=False)
        self.importances_.insert(0, "rank", range(1, len(table) + 1))
        self.selected_ = [c for c in X.columns if table.loc[c, "kept"]]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_]
