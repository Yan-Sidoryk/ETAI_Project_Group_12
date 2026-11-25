"""Threshold search used by compare and train."""
import unittest

import numpy as np

from scipy import stats

from src.evaluate import best_threshold, corrected_paired_ttest


class CorrectedPairedTtestTest(unittest.TestCase):
    def test_matches_nadeau_bengio_formula(self):
        a, b = np.array([0.75, 0.76, 0.74, 0.77, 0.73]), np.array([0.73, 0.75, 0.74, 0.74, 0.72])
        t, p = corrected_paired_ttest(a, b, train_rows=3000, val_rows=800)
        d = a - b
        expected_t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 5 + 800 / 3000))
        self.assertAlmostEqual(t, expected_t)
        self.assertAlmostEqual(p, 2 * stats.t.sf(abs(expected_t), df=4))
        # the correction makes it more conservative than a plain paired t-test
        self.assertGreater(p, stats.ttest_rel(a, b).pvalue)

    def test_identical_and_constant_differences(self):
        a = np.array([0.7, 0.8, 0.75])
        self.assertEqual(corrected_paired_ttest(a, a, 100, 50), (0.0, 1.0))
        self.assertEqual(corrected_paired_ttest(a + 0.01, a, 100, 50)[1], 0.0)


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
