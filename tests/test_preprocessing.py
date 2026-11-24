"""Behavior checks for configurable preprocessing, including inference and persistence."""
import copy
import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.preprocessing import build_preprocessor


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.config = {"categorical_features": ["category"], "scaler": "standard", "min_frequency": 2}
        self.train = pd.DataFrame({
            "category": ["a", "a", "b", "b", np.nan, "c"],
            "number": [1., 2., np.nan, 4., 8., 10.],
            "other": [3., 5., 6., 8., 10., 12.],
        })
        self.future = pd.DataFrame({"category": ["unseen", np.nan],
                                    "number": [np.nan, 1000.], "other": [7., 9.]})

    def test_legacy_defaults_and_training_only_statistics(self):
        prep = build_preprocessor(self.config).fit(self.train)
        numeric = prep.named_transformers_["remainder"]
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_, [4., 7.])
        means = numeric.named_steps["scale"].mean_.copy()
        transformed = prep.transform(self.future)
        np.testing.assert_allclose(numeric.named_steps["scale"].mean_, means)
        encoder = prep.named_transformers_["categorical"].named_steps["encode"]
        self.assertFalse(transformed[0, :len(encoder.get_feature_names_out())].any())
        self.assertTrue(numeric.named_steps["impute"].add_indicator)
        self.assertEqual(prep.named_transformers_["categorical"].named_steps["impute"].fill_value,
                         "missing")

    def test_all_scalers_fit_and_transform_unseen_rows(self):
        for name in ("none", "standard", "minmax", "robust", "maxabs", "normalize", "power", "quantile"):
            with self.subTest(scaler=name):
                setting = {"type": name, "params": {"n_quantiles": 4}} if name == "quantile" else name
                prep = build_preprocessor({**self.config, "scaler": setting})
                train = prep.fit_transform(self.train)
                future = prep.transform(self.future)
                self.assertEqual(train.shape[1], future.shape[1])
                self.assertTrue(np.isfinite(future).all())

    def test_imputer_encoder_combinations_and_persistence(self):
        for numeric in ("median", "mean", "most_frequent", "constant", "knn"):
            for categorical in ("constant", "most_frequent"):
                for encoder in ("onehot", "ordinal"):
                    with self.subTest(numeric=numeric, categorical=categorical, encoder=encoder):
                        config = {**self.config, "numeric_imputer": numeric,
                                  "categorical_imputer": categorical, "encoder": encoder}
                        prep = clone(build_preprocessor(config)).fit(self.train)
                        result = prep.transform(self.future)
                        self.assertTrue(np.isfinite(result).all())
                        if encoder == "ordinal":
                            self.assertEqual(result[0, 0], -1)
                        saved = io.BytesIO()
                        joblib.dump(prep, saved)
                        saved.seek(0)
                        np.testing.assert_allclose(joblib.load(saved).transform(self.future), result)

    def test_parameters_override_defaults_without_mutating_config(self):
        config = {**self.config,
                  "scaler": {"type": "minmax", "params": {"feature_range": [-1, 1]}},
                  "numeric_imputer": {"type": "constant", "params": {
                      "fill_value": -99, "add_indicator": False, "keep_empty_features": True}},
                  "categorical_imputer": {"type": "constant", "params": {"fill_value": "blank"}},
                  "encoder": {"type": "onehot", "params": {"min_frequency": None}}}
        before = copy.deepcopy(config)
        prep = build_preprocessor(config).fit(self.train)
        numeric = prep.named_transformers_["remainder"]
        self.assertEqual(numeric.named_steps["impute"].transform(self.future[["number", "other"]])[0, 0], -99)
        self.assertEqual(numeric.named_steps["scale"].feature_range, (-1, 1))
        categorical = prep.named_transformers_["categorical"]
        self.assertIn("blank", categorical.named_steps["encode"].categories_[0])
        self.assertIsNone(categorical.named_steps["encode"].min_frequency)
        self.assertEqual(config, before)

    def test_none_imputers_pass_through_missing_values(self):
        prep = build_preprocessor({**self.config, "numeric_imputer": "none",
                                   "categorical_imputer": "none", "scaler": "none"}).fit(self.train)
        result = prep.transform(self.future)
        self.assertTrue(np.isnan(result[0, -2]))
        self.assertEqual(prep.named_transformers_["categorical"].named_steps["impute"], "passthrough")

    def test_invalid_settings_give_config_context(self):
        for key, setting in (
            ("scaler", "typo"), ("numeric_imputer", "invalid"),
            ("categorical_imputer", "mean"), ("encoder", "invalid"),
            ("scaler", None), ("scaler", {"params": {}}),
            ("scaler", {"type": "standard", "params": []}),
            ("scaler", {"type": "none", "params": {"unused": 1}}),
            ("scaler", {"type": "standard", "params": {"typo": 1}}),
            ("encoder", {"type": "onehot", "param": {}}),
        ):
            with self.subTest(key=key, setting=setting):
                with self.assertRaisesRegex(ValueError, f"preprocessing.{key}"):
                    build_preprocessor({**self.config, key: setting})


if __name__ == "__main__":
    unittest.main()
