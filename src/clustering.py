"""K-means groups with training-only repeat-purchase probabilities."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted, validate_data


class ClusterClassifier(ClassifierMixin, BaseEstimator):
    """Predict binary outcomes from the observed positive rate in each cluster.

    Labels do not affect the K-means geometry. Probabilities are empirical rates,
    not calibrated estimates; a cluster without training assignments falls back
    to the overall training rate. Put learned preprocessing ahead of this estimator
    in a Pipeline so it is refitted inside each validation fold.
    """

    def __init__(self, n_clusters=3, n_init=20, random_state=42, threshold=0.5):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("ClusterClassifier requires binary labels 0 and 1.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        self.classes_ = np.array([0, 1])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init,
                              random_state=self.random_state).fit(X)
        # Use the same nearest-centroid rule at training and prediction time.
        assignments = self.kmeans_.predict(X)
        self.cluster_counts_ = np.bincount(assignments, minlength=self.n_clusters)
        positives = np.bincount(assignments, weights=y, minlength=self.n_clusters)
        self.global_rate_ = float(np.mean(y))
        self.cluster_rates_ = np.full(self.n_clusters, self.global_rate_)
        np.divide(positives, self.cluster_counts_, out=self.cluster_rates_,
                  where=self.cluster_counts_ > 0)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "cluster_rates_")
        X = validate_data(self, X, reset=False)
        positive = self.cluster_rates_[self.kmeans_.predict(X)]
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
