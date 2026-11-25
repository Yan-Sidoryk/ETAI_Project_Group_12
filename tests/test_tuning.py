"""Optuna tuning: config specs and a tiny end-to-end run on development folds."""
import ast
from argparse import Namespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optuna
import pandas as pd

import main
from src.data import load_config
from src.tuning import suggest


class TuningTest(unittest.TestCase):
    def test_suggest_reads_every_spec_type(self):
        trial = optuna.trial.FixedTrial({"a": 5, "b": 0.05, "c": None})
        self.assertEqual(suggest(trial, "a", ["int", 1, 10]), 5)
        self.assertEqual(suggest(trial, "b", ["float", 0.01, 0.3, "log"]), 0.05)
        self.assertIsNone(suggest(trial, "c", ["choice", None, 8, 12]))

    def test_tune_command_scores_configured_params_and_keeps_int_types(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["paths"]["outputs_dir"] = directory
            config["validation"]["n_splits"] = 2
            config["feature_selection"].update(n_splits=2, n_repeats=1, params={"n_estimators": 3, "min_samples_leaf": 5})
            config["classification"]["models"]["random_forest"] = {"params": {"n_estimators": 5, "min_samples_leaf": 5}}
            config["tuning"]["spaces"]["random_forest"] = {"n_estimators": ["int", 3, 8],
                                                          "min_samples_leaf": ["int", 1, 20]}
            output = io.StringIO()
            with patch.object(main, "load_config", return_value=config), redirect_stdout(output):
                main.tune_command(Namespace(task="classification", model="random_forest", trials=2))
            saved = list(Path(directory).glob("tuning_classification_random_forest_*.csv"))
            self.assertEqual(len(saved), 1)
            trials = pd.read_csv(saved[0])
            self.assertEqual(len(trials), 2)
            printed = output.getvalue()
            self.assertIn("configured params: 0.", printed)
            # the params to copy keep integer types, so they can go straight into config.yaml
            params = ast.literal_eval(printed.split("params: ")[-1].strip())
            self.assertIsInstance(params["n_estimators"], int)
            self.assertIsInstance(params["min_samples_leaf"], int)


if __name__ == "__main__":
    unittest.main()
