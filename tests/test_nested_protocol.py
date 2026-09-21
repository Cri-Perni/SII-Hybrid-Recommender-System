"""Test di regressione per il protocollo nested e la fusione batch-invariant."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import evaluate_top_n_detailed, perform_statistical_test
from src.experiment_config import ExperimentConfig
from src.hybrid import HybridRecommender
from src.nested_experiment import run_nested_experiment


class FixedModel:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def predict_batch(self, frame):
        self.calls.append(frame.copy())
        return np.asarray([self.values.get(item, 3.0) for item in frame["item_id"]], dtype=float)


class NestedProtocolTests(unittest.TestCase):
    def test_config_is_validated_and_json_ready(self):
        config = ExperimentConfig(outer_splits=2, inner_splits=2, alpha_grid=(0.0, 0.5, 1.0), theta_grid=(3.0,))
        self.assertEqual(config.to_dict()["alpha_grid"], [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            ExperimentConfig(outer_splits=1)
        with self.assertRaises(ValueError):
            ExperimentConfig(alpha_grid=(1.1,))

    def test_hybrid_endpoints_and_batch_invariance(self):
        cf = FixedModel({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
        cb = FixedModel({1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0})
        frame = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3]})
        np.testing.assert_allclose(HybridRecommender(cf, cb, alpha=1.0).predict_batch(frame), [1, 2, 3])
        np.testing.assert_allclose(HybridRecommender(cf, cb, alpha=0.0).predict_batch(frame), [5, 4, 3])
        hybrid = HybridRecommender(cf, cb, alpha=0.25)
        initial = hybrid.predict_batch(frame)
        extended = hybrid.predict_batch(pd.concat([frame, pd.DataFrame({"user_id": [1], "item_id": [4]})], ignore_index=True))
        np.testing.assert_allclose(initial, extended[:3])

    def test_top_n_excludes_training_items_and_is_seeded(self):
        model = FixedModel({1: 100.0, 2: 2.0, 3: 3.0})
        train = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]})
        test = pd.DataFrame({"user_id": [1], "item_id": [2], "rating": [5.0]})
        result = evaluate_top_n_detailed(model, train, test, [1, 2, 3], k_list=(1,), max_users=1, sample_seed=7)
        self.assertEqual(result["support"]["evaluated_users"], 1)
        self.assertNotIn(1, model.calls[0]["item_id"].tolist())
        self.assertEqual(result["metrics"]["Precision@1"], 0.0)

    def test_nested_runner_returns_only_outer_aggregate_structure(self):
        records = []
        for user_id in range(1, 7):
            for item_id in range(1, 7):
                records.append({"user_id": user_id, "item_id": item_id, "rating": float(1 + (user_id + item_id) % 5)})
        ratings = pd.DataFrame(records)
        feature_matrix = np.eye(6, dtype=float)
        config = ExperimentConfig(
            outer_splits=2,
            inner_splits=2,
            alpha_grid=(0.0, 0.5, 1.0),
            theta_grid=(3.0, 4.0),
            ranking_k=(1,),
            ranking_max_users=10,
            cold_item_max_train_ratings=2,
            n_factors=3,
        )
        result = run_nested_experiment(ratings, feature_matrix, {item_id: item_id - 1 for item_id in range(1, 7)}, range(1, 7), config)
        self.assertEqual(len(result["outer_folds"]), 2)
        self.assertEqual(sum(fold["test_rows"] for fold in result["outer_folds"]), len(ratings))
        self.assertEqual(result["pointwise"]["hybrid_selected"]["rmse"]["n"], 2)
        for fold in result["outer_folds"]:
            self.assertIn(fold["tuning"]["selected"]["alpha"], config.alpha_grid)
            self.assertIn(fold["tuning"]["selected"]["theta"], config.theta_grid)
            self.assertEqual(len(fold["tuning"]["configurations"]), len(config.alpha_grid) * len(config.theta_grid))

    def test_paired_statistics_require_aligned_outer_folds(self):
        with self.assertRaises(ValueError):
            perform_statistical_test([1.0, 1.1], [1.0])
        result = perform_statistical_test([0.9, 1.05], [1.0, 1.1])
        self.assertEqual(result["paired_folds"], 2)
        self.assertLess(result["mean_rmse_difference_a_minus_b"], 0)

    def test_outer_test_labels_do_not_change_its_inner_selection(self):
        from sklearn.model_selection import KFold

        records = [
            {"user_id": user_id, "item_id": item_id, "rating": float(1 + (user_id + item_id) % 5)}
            for user_id in range(1, 7)
            for item_id in range(1, 7)
        ]
        ratings = pd.DataFrame(records)
        config = ExperimentConfig(
            outer_splits=2, inner_splits=2, alpha_grid=(0.0, 0.5, 1.0), theta_grid=(3.0, 4.0),
            ranking_k=(1,), ranking_max_users=10, n_factors=3,
        )
        feature_matrix = np.eye(6, dtype=float)
        item_index = {item_id: item_id - 1 for item_id in range(1, 7)}
        baseline = run_nested_experiment(ratings, feature_matrix, item_index, range(1, 7), config)
        _, first_outer_test_indices = next(KFold(n_splits=2, shuffle=True, random_state=config.random_state).split(ratings))
        mutated = ratings.copy()
        mutated.loc[first_outer_test_indices, "rating"] = mutated.loc[first_outer_test_indices, "rating"] + 10.0
        altered = run_nested_experiment(mutated, feature_matrix, item_index, range(1, 7), config)
        self.assertEqual(
            baseline["outer_folds"][0]["tuning"]["selected"],
            altered["outer_folds"][0]["tuning"]["selected"],
        )

    def test_zero_training_interactions_are_included_in_cold_item_support(self):
        from sklearn.model_selection import KFold

        records = [
            {"user_id": user_id, "item_id": item_id, "rating": float(1 + (user_id + item_id) % 5)}
            for user_id in range(1, 7)
            for item_id in range(1, 7)
        ]
        ratings = pd.DataFrame(records)
        config = ExperimentConfig(
            outer_splits=2, inner_splits=2, alpha_grid=(0.0, 1.0), theta_grid=(3.0,),
            ranking_k=(1,), ranking_max_users=10, cold_item_max_train_ratings=0, n_factors=3,
        )
        _, first_outer_test_indices = next(KFold(n_splits=2, shuffle=True, random_state=config.random_state).split(ratings))
        ratings.loc[first_outer_test_indices[0], "item_id"] = 7
        result = run_nested_experiment(
            ratings, np.eye(7), {item_id: item_id - 1 for item_id in range(1, 8)}, range(1, 8), config
        )
        support = result["outer_folds"][0]["cold_start"]["support"]
        self.assertGreaterEqual(support["zero_train_test_items"], 1)
        self.assertGreaterEqual(support["test_ratings"], 1)

    def test_report_is_rendered_from_canonical_json_results(self):
        import json
        import tempfile
        from pathlib import Path

        import run_pipeline

        project_root = Path(__file__).resolve().parents[1]
        results = json.loads((project_root / "report" / "experiment_results.json").read_text(encoding="utf-8"))
        expected = results["datasets"]["ml-1m"]["nested_5x3"]["pointwise"]["hybrid_selected"]["rmse"]["mean"]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "relation.md"
            run_pipeline.write_report(results, destination)
            self.assertIn(f"{expected:.4f}", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
