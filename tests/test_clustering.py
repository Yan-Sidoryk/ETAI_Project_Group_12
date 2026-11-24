"""Checks for the notebook's training-only cluster classifier."""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.pipeline import Pipeline

from src.clustering import ClusterClassifier
from src.preprocessing import build_preprocessor


class ClusterClassifierTest(unittest.TestCase):
    def test_rates_and_nearest_centroid_predictions(self):
        X = np.array([[0.], [0.1], [0.2], [10.], [10.1], [10.2]])
        y = np.array([0, 0, 1, 1, 1, 1])
        model = ClusterClassifier(n_clusters=2).fit(X, y)
        future = np.array([[0.15], [9.9]])
        probabilities = model.predict_proba(future)
        np.testing.assert_allclose(probabilities[:, 1], [1 / 3, 1])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(future), [0, 1])
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        # Relabeling changes rates but must never change the learned geometry.
        relabeled = clone(model).fit(X, 1 - y)
        np.testing.assert_allclose(model.kmeans_.cluster_centers_, relabeled.kmeans_.cluster_centers_)
        np.testing.assert_allclose(relabeled.predict_proba(future)[:, 1], 1 - probabilities[:, 1])

    def test_empty_clusters_and_single_class(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = ClusterClassifier(n_clusters=3).fit([[0.], [0.], [10.], [10.]], [0, 1, 1, 1])
        empty = model.cluster_counts_ == 0
        self.assertTrue(empty.any())
        np.testing.assert_allclose(model.cluster_rates_[empty], 0.75)
        model = ClusterClassifier(n_clusters=1).fit([[0.], [1.]], [1, 1])
        np.testing.assert_allclose(model.predict_proba([[2.]]), [[0, 1]])

    def test_pipeline_cloning_unseen_categories_and_inference_is_read_only(self):
        X = pd.DataFrame({"category": ["a", "a", "b", "b"], "value": [0., 1., 9., 10.]})
        pipeline = Pipeline([
            ("prep", build_preprocessor({"categorical_features": ["category"], "min_frequency": None})),
            ("model", ClusterClassifier(n_clusters=2)),
        ])
        fitted = clone(pipeline).fit(X, [0, 0, 1, 1])
        rates = fitted.named_steps["model"].cluster_rates_.copy()
        means = fitted.named_steps["prep"].named_transformers_["remainder"].named_steps["scale"].mean_.copy()
        future = pd.DataFrame({"category": ["unseen"], "value": [100.]})
        result = fitted.predict_proba(future)
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(fitted.named_steps["model"].cluster_rates_, rates)
        np.testing.assert_array_equal(
            fitted.named_steps["prep"].named_transformers_["remainder"].named_steps["scale"].mean_, means)
        self.assertFalse(hasattr(pipeline.named_steps["model"], "cluster_rates_"))

    def test_invalid_labels_and_unfitted_prediction(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            ClusterClassifier(n_clusters=1).fit([[0.], [1.]], [0, 2])
        with self.assertRaises(NotFittedError):
            ClusterClassifier().predict([[0.]])


if __name__ == "__main__":
    unittest.main()
