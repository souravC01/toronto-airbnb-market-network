from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import airbnb_final_experiments as experiments  # noqa: E402
import robustness_analysis as robustness  # noqa: E402
from clean_dataset import clean_price_column  # noqa: E402


class CleaningTests(unittest.TestCase):
    def test_clean_price_column(self) -> None:
        result = clean_price_column(pd.Series(["$1,250.00", "$99", None]))
        self.assertEqual(result.iloc[0], 1250.0)
        self.assertEqual(result.iloc[1], 99.0)
        self.assertTrue(np.isnan(result.iloc[2]))


class NetworkMetricTests(unittest.TestCase):
    def test_variation_of_information_identity_and_symmetry(self) -> None:
        labels_a = [0, 0, 1, 1]
        labels_b = [0, 1, 0, 1]
        self.assertAlmostEqual(experiments.variation_of_information(labels_a, labels_a), 0.0)
        self.assertAlmostEqual(
            experiments.variation_of_information(labels_a, labels_b),
            experiments.variation_of_information(labels_b, labels_a),
        )

    def test_spatial_graph_connects_only_nearby_listings(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "host_id": [10, 20, 30],
                "latitude": [43.6500, 43.6501, 43.7500],
                "longitude": [-79.3800, -79.3801, -79.4800],
                "neighbourhood_cleansed": ["A", "A", "B"],
                "room_type": ["Entire home/apt"] * 3,
                "property_type": ["Entire condo"] * 3,
                "price_w": [100.0, 110.0, 120.0],
                "log_price_w": np.log([100.0, 110.0, 120.0]),
            }
        )
        config = experiments.GraphConfig("spatial", True, False, False)
        graph = experiments.build_graph_variant(frame, config, 500, 2, 0.6, 0.25, 0.15)

        self.assertTrue(graph.has_edge(1, 2))
        self.assertFalse(graph.has_edge(1, 3))
        self.assertEqual(graph[1][2]["types"], {"spatial"})


class RobustnessSplitTests(unittest.TestCase):
    def test_grouped_and_spatial_splits_have_expected_separation(self) -> None:
        frame = pd.DataFrame(
            {
                "host_id": np.repeat(np.arange(10), 2),
                "latitude": np.repeat([43.60, 43.64, 43.68, 43.72, 43.76], 4)
                + np.tile([0.0000, 0.0002, 0.0004, 0.0006], 5),
                "longitude": np.repeat([-79.60, -79.50, -79.40, -79.30, -79.20], 4)
                + np.tile([0.0000, 0.0002, 0.0004, 0.0006], 5),
            }
        )

        schemes, spatial_blocks = robustness.make_validation_splits(frame, n_splits=5)

        for train_indices, test_indices in schemes["host_grouped_5fold"]:
            train_hosts = set(frame.iloc[train_indices]["host_id"])
            test_hosts = set(frame.iloc[test_indices]["host_id"])
            self.assertFalse(train_hosts & test_hosts)

        held_out = []
        for train_indices, test_indices in schemes["spatial_block_5fold"]:
            train_blocks = set(spatial_blocks[train_indices])
            test_blocks = set(spatial_blocks[test_indices])
            self.assertEqual(len(test_blocks), 1)
            self.assertFalse(train_blocks & test_blocks)
            held_out.extend(test_indices.tolist())

        self.assertEqual(sorted(held_out), list(range(len(frame))))


if __name__ == "__main__":
    unittest.main()
