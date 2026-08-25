"""Tests for the statistical controls and the leakage fixes.

These tests exist to lock in the corrections, not just to exercise the code. Each
one fails if a specific defect is reintroduced.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import robustness_analysis as robustness  # noqa: E402
import statistical_controls as controls  # noqa: E402


def toy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "latitude": 43.65 + rng.normal(0, 0.05, n),
            "longitude": -79.38 + rng.normal(0, 0.05, n),
            "neighbourhood_cleansed": rng.choice(
                [f"Area {index}" for index in range(12)], n
            ),
        }
    )


class AlignmentControlTests(unittest.TestCase):
    def test_random_partition_preserves_sizes(self) -> None:
        labels = controls.random_partition(10, [3, 3, 4], random_state=1)
        counts = sorted(np.bincount(labels).tolist())
        self.assertEqual(counts, [3, 3, 4])

    def test_random_partition_rejects_mismatched_sizes(self) -> None:
        with self.assertRaises(ValueError):
            controls.random_partition(10, [3, 3], random_state=1)

    def test_aggregate_neighbourhoods_reaches_target_granularity(self) -> None:
        frame = toy_frame()
        labels = controls.aggregate_neighbourhoods(frame, n_regions=4)
        self.assertEqual(len(labels), len(frame))
        self.assertEqual(len(np.unique(labels)), 4)

    def test_aggregate_neighbourhoods_is_a_function_of_neighbourhood(self) -> None:
        """Super-regions must be a strict coarsening: one region per neighbourhood."""
        frame = toy_frame()
        labels = controls.aggregate_neighbourhoods(frame, n_regions=5)
        grouped = pd.DataFrame(
            {"area": frame["neighbourhood_cleansed"], "region": labels}
        )
        self.assertTrue((grouped.groupby("area")["region"].nunique() == 1).all())

    def test_alignment_report_random_floor_is_near_zero(self) -> None:
        """A chance-corrected score must sit at ~0 for a random partition."""
        frame = toy_frame()
        rng = np.random.default_rng(3)
        detected = rng.integers(0, 5, len(frame))
        report = controls.alignment_report(frame, detected)
        self.assertAlmostEqual(report["ami_random_floor"], 0.0, delta=0.05)
        self.assertEqual(report["detected_communities"], 5)
        self.assertEqual(report["official_neighbourhoods"], 12)

    def test_alignment_report_is_perfect_for_identical_partitions(self) -> None:
        frame = toy_frame()
        detected = frame["neighbourhood_cleansed"].to_numpy()
        report = controls.alignment_report(frame, detected)
        self.assertAlmostEqual(report["nmi_vs_official"], 1.0, places=6)
        self.assertAlmostEqual(report["ami_vs_official"], 1.0, places=6)


class PairedTestTests(unittest.TestCase):
    def test_consistent_improvement_gives_a_significant_sign_test(self) -> None:
        result = controls.paired_fold_test(
            baseline=[0.60, 0.61, 0.62, 0.63, 0.64],
            expanded=[0.601, 0.611, 0.621, 0.631, 0.641],
            metric="test_r2",
        )
        self.assertEqual(result.sign_test_wins, 5)
        self.assertAlmostEqual(result.sign_test_pvalue, 0.03125, places=5)
        self.assertGreater(result.mean_delta, 0)

    def test_lower_is_better_flips_the_sign(self) -> None:
        """For MAE, a decrease is an improvement and must score as a win."""
        result = controls.paired_fold_test(
            baseline=[55.0, 55.0, 55.0],
            expanded=[54.0, 54.0, 54.0],
            metric="mae",
            higher_is_better=False,
        )
        self.assertEqual(result.sign_test_wins, 3)
        self.assertGreater(result.mean_delta, 0)

    def test_no_difference_is_not_significant(self) -> None:
        result = controls.paired_fold_test(
            baseline=[0.6] * 5, expanded=[0.6] * 5, metric="test_r2"
        )
        self.assertEqual(result.sign_test_wins, 0)
        self.assertGreater(result.sign_test_pvalue, 0.5)


class LeakageTests(unittest.TestCase):
    def test_winsorization_bounds_ignore_the_test_fold(self) -> None:
        """An extreme value confined to the test fold must not move the bounds."""
        price = np.concatenate([np.full(98, 100.0), [1.0, 100_000.0]])
        train_index = np.arange(98)
        result = robustness.winsorize_target_in_fold(price, train_index)
        # Every training value is 100, so both bounds collapse to 100 and the two
        # outliers, which live only in the test fold, are clipped to it.
        self.assertAlmostEqual(float(np.exp(result[-1])), 100.0, places=6)
        self.assertAlmostEqual(float(np.exp(result[-2])), 100.0, places=6)

    def test_prepare_price_frame_does_not_impute(self) -> None:
        """Imputation must happen inside the pipeline, not before the split."""
        frame = pd.DataFrame(
            {
                "log_price_w": np.log([100.0, 120.0, 150.0, 200.0]),
                "price": [100.0, 120.0, 150.0, 200.0],
                "host_id": [1, 2, 3, 4],
                "latitude": [43.6, 43.7, 43.65, 43.62],
                "longitude": [-79.4, -79.38, -79.36, -79.39],
                "community": ["0", "1", "0", "1"],
                "accommodates": [2.0, np.nan, 4.0, 3.0],
                "room_type": ["Entire home/apt"] * 4,
            }
        )
        work, numeric, _ = robustness.prepare_price_frame(frame, "community")
        self.assertIn("accommodates", numeric)
        self.assertTrue(work["accommodates"].isna().any())

    def test_pipeline_imputes_missing_numerics(self) -> None:
        """The pipeline must still handle NaNs now that the frame keeps them."""
        pipeline = robustness.make_price_pipeline(["accommodates"], ["room_type"])
        X = pd.DataFrame(
            {
                "accommodates": [2.0, np.nan, 4.0, 6.0],
                "room_type": ["Entire home/apt", "Private room"] * 2,
            }
        )
        y = np.log([100.0, 120.0, 150.0, 200.0])
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        self.assertEqual(len(predictions), 4)
        self.assertFalse(np.isnan(predictions).any())


if __name__ == "__main__":
    unittest.main()
