"""Threshold search used by compare and train."""
import unittest

import numpy as np

from src.evaluate import best_threshold


class BestThresholdTest(unittest.TestCase):
    def test_finds_the_threshold_that_separates_the_classes(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        probability = np.array([0.10, 0.20, 0.30, 0.35, 0.60, 0.90])
        threshold, f1 = best_threshold(y, probability)
        self.assertTrue(0.30 < threshold <= 0.35)  # the lowest threshold that still rejects every 0
        self.assertEqual(f1, 1.0)

    def test_low_threshold_when_almost_everyone_returns(self):
        y = np.array([1, 1, 1, 1, 0])
        probability = np.array([0.20, 0.25, 0.30, 0.40, 0.45])  # uninformative, all below 0.5
        threshold, f1 = best_threshold(y, probability)
        self.assertEqual(threshold, 0.05)  # predicting "returns" for everyone beats any cut-off
        self.assertAlmostEqual(f1, 2 * 4 / (2 * 4 + 1))


if __name__ == "__main__":
    unittest.main()
