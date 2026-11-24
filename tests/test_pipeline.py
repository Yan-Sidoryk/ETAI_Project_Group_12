"""Integration checks for the four routes, using temporary model/output paths.

Run from the repository root: python -m unittest discover -s tests -v
The real datasets are used with small models so the checks stay quick.
"""
from argparse import Namespace
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import main
from src.data import load_config
from src.inference import classify, load_model, predict_spend


class PipelineRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="etai-pipeline-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.config = load_config()
        # These route checks use fixed stand-in models without search spaces.
        cls.config["tuning"] = {"enabled": False}
        cls.config["paths"]["models_dir"] = str(Path(cls.directory.name) / "models")
        cls.config["paths"]["outputs_dir"] = str(Path(cls.directory.name) / "outputs")
        cls.config["outputs"] = {
            "submission": str(Path(cls.directory.name) / "outputs/submission.csv"),
            "regression_predictions": str(Path(cls.directory.name) / "outputs/regression.csv"),
            "scoreboard": str(Path(cls.directory.name) / "outputs/runs.csv"),
        }
        cls.config["validation"]["n_splits"] = 2
        # Small stand-in models; select them explicitly so the tests do not depend on config.yaml's choice.
        cls.config["classification"]["model"] = "random_forest"
        cls.config["regression"]["model"] = "catboost"
        cls.config["classification"]["models"] = {
            "random_forest": {"params": {"n_estimators": 3, "min_samples_leaf": 5}}
        }
        cls.config["regression"]["models"] = {
            "catboost": {"params": {
                "iterations": 3, "depth": 3, "verbose": 0,
                "thread_count": 1, "allow_writing_files": False,
            }}
        }
        with patch.object(main, "load_config", return_value=cls.config), redirect_stdout(io.StringIO()):
            main.train(Namespace(task="classification"))
            main.train(Namespace(task="regression"))

    def run_route(self, route, **arguments):
        output = io.StringIO()
        with patch.object(main, "load_config", return_value=self.config), redirect_stdout(output):
            route(Namespace(**arguments))
        return output.getvalue()

    def test_training_saves_both_tasks_and_finite_assessment(self):
        runs = pd.read_csv(self.config["outputs"]["scoreboard"])
        self.assertEqual(list(runs["task"]), ["classification", "regression"])
        self.assertTrue(np.isfinite(runs[["cv_mean", "cv_std", "floor", "holdout"]]).all().all())
        for task in ("classification", "regression"):
            model = load_model(self.config, task)
            self.assertEqual(model["features"], list(model["pipeline"].feature_names_in_))
            self.assertTrue(set(self.config["targets"].values()).isdisjoint(model["features"]))

    def test_comparison_reports_candidates_and_does_not_overwrite_models(self):
        for task, filename in (("classification", "classifier.joblib"), ("regression", "regressor.joblib")):
            path = Path(self.config["paths"]["models_dir"]) / filename
            before = path.read_bytes()
            self.run_route(main.compare, task=task)
            table = pd.read_csv(Path(self.config["paths"]["outputs_dir"]) / f"comparison_{task}.csv")
            self.assertEqual(set(table["model"]), {"floor", self.config[task]["model"]})
            self.assertEqual(before, path.read_bytes())

    def test_batch_export_preserves_ids_and_coherent_spend(self):
        # A prediction route must succeed without access to the training CSV.
        config = copy.deepcopy(self.config)
        config["paths"]["train"] = "unused-training-file.csv"
        with patch.object(main, "load_config", return_value=config), redirect_stdout(io.StringIO()):
            main.predict(Namespace())
        test = pd.read_csv(config["paths"]["test"])
        clients = pd.read_csv(config["paths"]["clients"])
        submission = pd.read_csv(config["outputs"]["submission"])
        regression = pd.read_csv(config["outputs"]["regression_predictions"])
        self.assertEqual(list(submission.columns), ["ID", "repeat_purchase_90d"])
        self.assertEqual(list(regression.columns), ["ID", "repeat_purchase_90d", "spend_next_90d"])
        self.assertEqual(list(submission["ID"]), list(test["ID"]))
        self.assertTrue(submission["repeat_purchase_90d"].isin([0, 1]).all())
        selected = submission["repeat_purchase_90d"] == 1
        self.assertEqual(list(regression["ID"]), list(submission.loc[selected, "ID"]))
        self.assertTrue((regression["spend_next_90d"] >= 0).all())
        spend = predict_spend(load_model(config, "regression"), test, clients)
        np.testing.assert_allclose(regression["spend_next_90d"], spend[selected])
        classifier = load_model(config, "classification")
        batch_probability, _ = classify(classifier, test, clients)
        single_probability, _ = classify(classifier, test.iloc[[0]], clients)
        np.testing.assert_allclose(single_probability, batch_probability[:1])

    def test_client_examples_and_negative_decision_gate(self):
        for name in ("new_customer", "returning_customer"):
            output = self.run_route(main.client, input=f"examples/{name}.json")
            self.assertIn("Probability of buying again within 90 days:", output)
            self.assertIn("Expected spend in the next 90 days:", output)

        path = Path(self.config["paths"]["models_dir"]) / "classifier.joblib"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        classifier = joblib.load(path)
        classifier["threshold"] = 1.1  # Force a negative decision for this check.
        joblib.dump(classifier, path)
        with patch.object(main, "predict_spend", side_effect=AssertionError("Spend must be gated")):
            output = self.run_route(main.client, input="examples/new_customer.json")
        self.assertIn("does not return (0)", output)
        self.assertIn("0.00 EUR", output)

    def test_client_rejects_unknown_fields_and_missing_models(self):
        invalid = Path(self.directory.name) / "invalid.json"
        invalid.write_text(json.dumps({"unknown_field": 1}), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "Unknown fields"):
            self.run_route(main.client, input=str(invalid))
        config = copy.deepcopy(self.config)
        config["paths"]["models_dir"] = str(Path(self.directory.name) / "absent")
        with self.assertRaisesRegex(SystemExit, "train --task regression"):
            load_model(config, "regression")


if __name__ == "__main__":
    unittest.main()
