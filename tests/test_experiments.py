"""Experiment ranking, isolation, reproducibility, and command integration."""
from argparse import Namespace
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import main
from src.data import load_config
from src.experiments import preprocessing_candidates, run_preprocessing_experiment


class PreprocessingExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config()
        # These grid assertions require the reference to be one of its candidates.
        # Keep that fixture independent of the user's selected production scaler.
        self.config["preprocessing"]["scaler"] = "standard"
        self.config["paths"]["outputs_dir"] = self.temp.name
        self.config["preprocessing_experiment"] = {
            "mode": "grid", "candidates": {"scaler": ["standard", "none"], "encoder": ["onehot", "ordinal"]}}

    def test_grid_keeps_baseline_and_deduplicates(self):
        self.assertEqual(len(preprocessing_candidates(self.config)), 4)
        choices = preprocessing_candidates(self.config)
        self.assertEqual(len(choices), 4)
        self.assertEqual(choices[0], self.config["preprocessing"])
        self.config["preprocessing_experiment"]["candidates"]["scaler"] = []
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            preprocessing_candidates(self.config)

    def test_ranking_errors_identical_splits_and_csv_columns(self):
        class CV:
            calls = 0

            def split(self, X):
                self.calls += 1
                return iter([(np.array([0, 1]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))])

        for task, metric, expected in (("classification", "f1", 0.9), ("regression", "rmse", 0.1),
                                       ("regression", "r2", 0.9)):
            with self.subTest(task=task, metric=metric):
                config = copy.deepcopy(self.config)
                config[task]["metrics"] = [metric]
                config[task]["selection_metric"] = metric
                config["preprocessing_experiment"]["candidates"] = {
                    "scaler": ["standard", "none", "invalid"]}
                cv = CV()
                observed = []

                def score(pipeline, X, y, frozen, task, task_config):
                    observed.append([(train.tolist(), val.tolist()) for train, val in frozen.split(X)])
                    score = 0.9 if pipeline.get_params()["prep__remainder__scale"] == "passthrough" else 0.1
                    return pd.DataFrame({metric: [score, score], "validation": [score, score],
                                         "gap": [0., 0.], "fold": [1, 2]})

                before = copy.deepcopy(config)
                X = pd.DataFrame({"unused": range(4)})
                with patch("src.experiments.cross_validate_pipeline", side_effect=score), redirect_stdout(io.StringIO()):
                    directory, table = run_preprocessing_experiment(config, task, X, X["unused"], cv)
                    archive = (directory / "summary.csv").read_bytes()
                    latest = Path(config["paths"]["outputs_dir"]) / f"preprocessing_{task}_summary.csv"
                    latest.write_text("stale results", encoding="utf-8")
                    second, second_table = run_preprocessing_experiment(config, task, X, X["unused"], cv)
                self.assertNotEqual(directory, second)
                self.assertEqual((directory / "summary.csv").read_bytes(), archive)
                self.assertEqual(latest.read_bytes(), (second / "summary.csv").read_bytes())
                self.assertEqual(len(pd.read_csv(latest)), len(second_table))
                self.assertEqual(cv.calls, 2)  # Once per experiment, never once per candidate.
                self.assertTrue(all(splits == observed[0] for splits in observed))
                self.assertAlmostEqual(table.iloc[0]["cv_mean"], expected)
                self.assertEqual(table.iloc[-1]["status"], "failed")
                self.assertIn("Unknown preprocessing.scaler", table.iloc[-1]["error"])
                required = {"candidate", "task", "model", "scaler", "numeric_imputer",
                            "categorical_imputer", "encoder"}
                for csv in (directory / "summary.csv", directory / "folds.csv", latest):
                    columns = set(pd.read_csv(csv).columns)
                    self.assertTrue(required <= columns)
                    self.assertTrue({"preprocessing", "model_config"}.isdisjoint(columns))
                self.assertEqual(table.iloc[0]["model"], config[task]["model"])
                self.assertEqual(set(table["scaler"]), {"standard", "none", "invalid"})
                self.assertEqual({path.name for path in directory.iterdir()}, {"summary.csv", "folds.csv"})
                self.assertEqual(len(pd.read_csv(directory / "folds.csv")), 4)
                self.assertEqual(len(pd.read_csv(directory / "summary.csv")), 3)
                self.assertEqual(config, before)

    def test_real_route_both_tasks_without_generated_configs_or_models(self):
        config = copy.deepcopy(self.config)
        config["validation"]["n_splits"] = 2
        config["paths"]["models_dir"] = str(Path(self.temp.name) / "models")
        config["outputs"]["scoreboard"] = str(Path(self.temp.name) / "runs.csv")
        config["classification"]["model"] = "logistic_regression"
        config["classification"]["models"] = {"logistic_regression": {"params": {"max_iter": 2000}}}
        config["regression"]["model"] = "ridge"
        config["regression"]["models"] = {"ridge": {"params": {"alpha": 1.0}}}
        config["preprocessing_experiment"]["candidates"] = {"encoder": ["onehot", "ordinal"]}
        snapshot = Path(self.temp.name) / "input.yaml"
        snapshot.write_text(yaml.safe_dump(config))
        for task in ("classification", "regression"):
            with redirect_stdout(io.StringIO()), patch.object(main, "score_model", side_effect=AssertionError("Holdout scored")):
                main.experiment(Namespace(task=task, config=str(snapshot), models=None))
            directory, = (path for path in Path(self.temp.name).glob(f"preprocessing_{task}_*") if path.is_dir())
            self.assertFalse(Path(config["paths"]["models_dir"]).exists())
            table = pd.read_csv(directory / "summary.csv")
            self.assertEqual(list(table["status"]), ["ok", "ok"])
            self.assertEqual({path.name for path in directory.iterdir()}, {"summary.csv", "folds.csv"})
            self.assertEqual(yaml.safe_load(snapshot.read_text()), config)

    def test_all_failed_retains_error_table(self):
        config = copy.deepcopy(self.config)
        config["preprocessing"]["scaler"] = "invalid"
        config["preprocessing_experiment"]["candidates"] = {}
        X = pd.DataFrame({"unused": range(4)})

        class CV:
            def split(self, X):
                return iter([(np.array([0]), np.array([1])), (np.array([0, 1]), np.array([2, 3]))])

        with redirect_stdout(io.StringIO()):
            directory, table = run_preprocessing_experiment(config, "classification", X, X["unused"], CV())
        self.assertEqual(table.iloc[0]["status"], "failed")
        latest = Path(config["paths"]["outputs_dir"]) / "preprocessing_classification_summary.csv"
        self.assertEqual(pd.read_csv(latest).iloc[0]["status"], "failed")
        self.assertTrue((directory / "summary.csv").exists())
        self.assertFalse((directory / "best_config.yaml").exists())


if __name__ == "__main__":
    unittest.main()
