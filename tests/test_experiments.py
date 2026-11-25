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
        self.config["paths"]["outputs_dir"] = self.temp.name
        # A fixed baseline and a tiny selector, so the tests do not depend on config.yaml's current choices.
        self.config["preprocessing"].update(scaler="standard", numeric_imputer="median",
                                            categorical_imputer="constant", encoder="onehot")
        self.config["feature_selection"].update(n_splits=2, n_repeats=1,
                                                params={"n_estimators": 3, "min_samples_leaf": 5})
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

    def test_soft_voting_builds_its_members_from_their_entries(self):
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from xgboost import XGBClassifier
        from src.model import build_estimator
        models = self.config["classification"]["models"]
        models["rf"] = {"params": {"n_estimators": 7}, "type": "random_forest"}
        models["xgb"] = {"params": {"max_depth": 2}, "type": "xgboost"}
        voting = build_estimator("classification", self.config,
                                 {"type": "voting", "params": {"members": ["rf", "xgb"]}})
        self.assertIsInstance(voting, VotingClassifier)
        self.assertEqual(voting.voting, "soft")
        (rf_name, rf), (xgb_name, xgb) = voting.estimators
        self.assertEqual((rf_name, xgb_name), ("rf", "xgb"))
        self.assertIsInstance(rf, RandomForestClassifier)
        self.assertEqual((rf.n_estimators, rf.random_state), (7, self.config["seed"]))
        self.assertIsInstance(xgb, XGBClassifier)
        with self.assertRaisesRegex(ValueError, "classification only"):
            build_estimator("regression", self.config, {"type": "voting", "params": {"members": ["rf"]}})

    def test_task_preprocessing_overrides_shared_settings(self):
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from src.model import build_pipeline
        self.config["preprocessing"]["encoder"] = "ordinal"
        self.config["classification"].pop("preprocessing", None)
        self.config["regression"]["preprocessing"] = {"encoder": "onehot"}
        for task, name, encoder in (("classification", "ordinal", OrdinalEncoder),
                                    ("regression", "onehot", OneHotEncoder)):
            pipeline = build_pipeline(task, self.config, {"type": "random_forest"})
            categorical = pipeline.named_steps["prep"].transformers[0][1]
            self.assertIsInstance(categorical.named_steps["encode"], encoder)
            self.assertEqual(pipeline.named_steps["select"].preprocessing["encoder"], name)
        self.assertEqual(preprocessing_candidates(self.config, "regression")[0]["encoder"], "onehot")
        # A model entry's own preprocessing changes its "prep" step only, never the shared selection
        pipeline = build_pipeline("classification", self.config,
                                  {"type": "mlp", "preprocessing": {"encoder": "onehot"}})
        self.assertIsInstance(pipeline.named_steps["prep"].transformers[0][1].named_steps["encode"], OneHotEncoder)
        self.assertEqual(pipeline.named_steps["select"].preprocessing["encoder"], "ordinal")

    def test_feature_selection_is_shared_by_every_candidate(self):
        self.config["classification"]["model"] = "random_forest"
        seen = []

        def score(pipeline, X, y, cv, task, task_config):
            params = pipeline.get_params()
            seen.append((params["select__preprocessing"], params["prep__remainder__scale"]))
            return pd.DataFrame({"f1": [0.5, 0.5], "validation": [0.5, 0.5], "gap": [0., 0.], "fold": [1, 2],
                                 **{m: [0.5, 0.5] for m in self.config["classification"]["metrics"]}})

        X = pd.DataFrame({"unused": range(6)})
        cv = type("CV", (), {"split": lambda self, X: iter([([0, 1], [2, 3]), ([0, 1, 2], [4, 5])])})()
        with patch("src.experiments.cross_validate_pipeline", side_effect=score), redirect_stdout(io.StringIO()):
            run_preprocessing_experiment(self.config, "classification", X, X["unused"], cv)
        self.assertTrue(all(select == self.config["preprocessing"] for select, _ in seen))
        self.assertGreater(len({repr(scale) for _, scale in seen}), 1)  # the model's own preprocessing does vary

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
