"""Optuna search validation, ranking, reproducibility and route integration."""
from argparse import Namespace
from contextlib import redirect_stdout
import copy
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml

import main
from src.data import load_config
from src.features import split_features_target
from src.inference import MODEL_FILES
from src.model import build_pipeline
from src.tuning import run_tuning, suggest_parameters, validate_search_space


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config()
        self.config["paths"]["outputs_dir"] = self.temp.name
        self.config["tuning"] = {"n_trials": 2, "timeout_seconds": None}
        self.X = pd.DataFrame({"unused": range(6)})
        self.y = pd.Series([0, 1, 0, 1, 0, 1])

    @staticmethod
    def cv():
        class CV:
            calls = 0

            def split(self, X):
                self.calls += 1
                return iter([(np.array([0, 1]), np.array([2, 3])),
                             (np.array([0, 1, 2, 3]), np.array([4, 5]))])
        return CV()

    def test_distributions_and_validation(self):
        space = {"a": {"type": "int", "low": 1, "high": 5, "step": 2},
                 "b": {"type": "float", "low": 0.01, "high": 1., "log": True},
                 "c": {"type": "categorical", "choices": [None, "balanced"]}}
        validate_search_space(space)
        self.assertEqual(suggest_parameters(optuna.trial.FixedTrial({"a": 3, "b": .1, "c": None}), space),
                         {"a": 3, "b": .1, "c": None})
        for invalid in ({}, {"a": {"type": "wrong"}},
                        {"a": {"type": "int", "low": 5, "high": 1}},
                        {"a": {"type": "float", "low": 0, "high": 1, "log": True}},
                        {"a": {"type": "categorical", "choices": []}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_search_space(invalid)
        # Every configured model has a search space accepted by its estimator.
        for task in ("classification", "regression"):
            for name, spec in self.config[task]["models"].items():
                validate_search_space(spec["search_space"])
                model = build_pipeline(task, self.config, {"type": name, **spec}).named_steps["model"]
                known = set(model.get_params()) | set(inspect.signature(type(model).__init__).parameters)
                self.assertTrue(set(spec["search_space"]) <= known)

    def test_ranking_fixed_splits_and_best_config(self):
        for task, metric in (("classification", "f1"), ("regression", "rmse"), ("regression", "r2")):
            with self.subTest(task=task, metric=metric):
                config = copy.deepcopy(self.config)
                config[task]["metrics"] = [metric]
                config[task]["selection_metric"] = metric
                config[task]["model"] = "random_forest"
                config[task]["models"]["random_forest"]["search_space"] = {
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 20}}
                cv, observed = self.cv(), []

                def evaluate(pipeline, X, y, fixed, task, settings):
                    observed.append([(a.tolist(), b.tolist()) for a, b in fixed.split(X)])
                    value = pipeline.named_steps["model"].min_samples_leaf / 20
                    return pd.DataFrame({metric: [value, value], "validation": [value, value],
                                         "gap": [0., 0.], "fold": [1, 2]})

                before = copy.deepcopy(config)
                with patch("src.tuning.cross_validate_pipeline", side_effect=evaluate), redirect_stdout(io.StringIO()):
                    directory, summary = run_tuning(config, task, self.X, self.y, cv)
                trials = pd.read_csv(directory / "trials.csv")
                self.assertEqual(trials.columns[-2:].tolist(), ["params", "sampled_params"])
                exported_summary = pd.read_csv(directory / "summary.csv")
                self.assertEqual(exported_summary.columns[-2:].tolist(), ["params", "sampled_params"])
                self.assertTrue(trials.cv_mean.is_monotonic_increasing if metric == "rmse"
                                else trials.cv_mean.is_monotonic_decreasing)
                exported_folds = pd.read_csv(directory / "folds.csv")
                self.assertTrue(exported_folds[metric].is_monotonic_increasing if metric == "rmse"
                                else exported_folds[metric].is_monotonic_decreasing)
                optimum = trials.cv_mean.min() if metric == "rmse" else trials.cv_mean.max()
                self.assertAlmostEqual(summary.iloc[0].cv_mean, optimum)
                self.assertEqual(cv.calls, 1)
                self.assertEqual(observed[0], observed[1])
                best = yaml.safe_load((directory / "best_config.yaml").read_text())
                self.assertNotIn("diagnosis", best)
                self.assertEqual(best["preprocessing"], config["preprocessing"])
                self.assertEqual(best[task]["models"]["random_forest"]["params"], json.loads(summary.iloc[0].params))
                self.assertEqual(config, before)
                self.assertEqual(len(pd.read_csv(directory / "folds.csv")), 4)

    def test_failure_recording_and_no_successful_config(self):
        config = self.config
        with patch("src.tuning.cross_validate_pipeline", side_effect=ValueError("invalid trial")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "All tuning trials failed"):
                run_tuning(config, "classification", self.X, self.y, self.cv())
        directory, = Path(self.temp.name).iterdir()
        trials = pd.read_csv(directory / "trials.csv")
        self.assertTrue((trials.status == "failed").all())
        self.assertTrue(trials.error.str.contains("invalid trial").all())
        self.assertFalse((directory / "best_config.yaml").exists())

    def test_failed_trial_does_not_prevent_later_success(self):
        metric = self.config["classification"]["selection_metric"]
        success = pd.DataFrame({key: [0.7, 0.8] for key in self.config["classification"]["metrics"]})
        success = success.assign(validation=success[metric], gap=0., fold=[1, 2])
        with patch("src.tuning.cross_validate_pipeline", side_effect=[ValueError("bad combination"), success]), \
                redirect_stdout(io.StringIO()):
            directory, summary = run_tuning(self.config, "classification", self.X, self.y, self.cv())
        trials = pd.read_csv(directory / "trials.csv")
        self.assertEqual(trials.status.tolist(), ["ok", "failed"])
        self.assertEqual(int(summary.iloc[0].trial), 1)
        self.assertTrue((directory / "best_config.yaml").exists())

    def test_real_route_and_generated_config_train_both_tasks(self):
        config = self.config
        config["validation"]["n_splits"] = 2
        config["classification"]["model"] = "logistic_regression"
        config["regression"]["model"] = "ridge"
        config["paths"]["models_dir"] = str(Path(self.temp.name) / "models")
        config["outputs"]["scoreboard"] = str(Path(self.temp.name) / "runs.csv")
        config["tuning"]["enabled"] = True
        snapshot = Path(self.temp.name) / "input.yaml"
        serializable = copy.deepcopy(config)
        serializable.pop("diagnosis")
        snapshot.write_text(yaml.safe_dump(serializable))
        for task in ("classification", "regression"):
            _, dev, _, cv = main.load_training_splits(config, task)
            X, y = split_features_target(dev, config, task)
            with redirect_stdout(io.StringIO()), patch.object(main, "score_model", side_effect=AssertionError("Holdout scored")):
                run_tuning(config, task, X, y, cv)
                run_tuning(config, task, X, y, cv)
            directories = sorted(Path(self.temp.name).glob(f"tuning_{task}_*"))
            self.assertEqual(len(directories), 2)
            first, second = [pd.read_csv(d / "trials.csv") for d in directories]
            self.assertEqual(first.params.tolist(), second.params.tolist())
            np.testing.assert_allclose(first.cv_mean, second.cv_mean)
            self.assertTrue((first.status == "ok").all())
            if task == "classification":
                self.assertFalse(Path(config["paths"]["models_dir"]).exists())
            # Config-enabled training tunes, assesses and saves the selected params.
            with redirect_stdout(io.StringIO()):
                main.train(Namespace(task=task, config=str(snapshot)))
            directories = sorted(Path(self.temp.name).glob(f"tuning_{task}_*"))
            self.assertEqual(len(directories), 3)
            best = yaml.safe_load((directories[-1] / "best_config.yaml").read_text())
            self.assertFalse(best["tuning"]["enabled"])
            artifact = joblib.load(Path(config["paths"]["models_dir"]) / MODEL_FILES[task])
            name = config[task]["model"]
            self.assertEqual(artifact["config"][task]["models"][name]["params"], best[task]["models"][name]["params"])
            # A generated snapshot trains directly without triggering another search.
            with redirect_stdout(io.StringIO()), patch("src.tuning.run_tuning", side_effect=AssertionError("Tuning repeated")):
                main.train(Namespace(task=task, config=str(directories[0] / "best_config.yaml")))
        self.assertEqual(yaml.safe_load(snapshot.read_text()), serializable)

    def test_disabled_or_missing_flag_skips_tuning(self):
        config = self.config
        config["classification"]["model"] = "logistic_regression"
        config["validation"]["n_splits"] = 2
        config["paths"]["models_dir"] = str(Path(self.temp.name) / "models")
        config["outputs"]["scoreboard"] = str(Path(self.temp.name) / "runs.csv")
        for options in ({"enabled": False}, {}):
            config["tuning"] = options
            with patch.object(main, "load_config", return_value=config), redirect_stdout(io.StringIO()), \
                    patch("src.tuning.run_tuning", side_effect=AssertionError("Tuning should be disabled")):
                main.train(Namespace(task="classification"))
        config["tuning"]["enabled"] = "false"
        with patch.object(main, "load_config", return_value=config), self.assertRaisesRegex(ValueError, "true or false"):
            main.train(Namespace(task="classification"))

    def test_invalid_options_fail_before_creating_outputs(self):
        for trials in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_tuning(self.config, "classification", self.X, self.y, self.cv(), n_trials=trials)
        self.config["classification"]["models"]["random_forest"]["search_space"] = {
            "typo": {"type": "int", "low": 1, "high": 2}}
        with self.assertRaisesRegex(ValueError, "unknown search_space"):
            run_tuning(self.config, "classification", self.X, self.y, self.cv())
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
