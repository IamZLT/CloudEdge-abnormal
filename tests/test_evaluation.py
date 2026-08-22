from __future__ import annotations

import math
import unittest

import numpy as np

from src.evaluation import (
    count_model_parameters,
    evaluate_binary,
    fit_threshold,
    qwen_anomaly_score,
    summarize_inference,
    summarize_latency,
)
from src.metrics import binary_detection_metrics, latency_stats


class QwenScoreTest(unittest.TestCase):
    def test_score_direction(self):
        self.assertAlmostEqual(qwen_anomaly_score("NG", 0.9)["score"], 0.9)
        self.assertAlmostEqual(qwen_anomaly_score("OK", 0.9)["score"], 0.1)

    def test_parse_failure_abstains(self):
        result = qwen_anomaly_score("OK", 0.9, parse_ok=False)
        self.assertFalse(result["valid"])
        self.assertEqual(result["decision"], "REVIEW")
        self.assertTrue(math.isnan(result["score"]))


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([0, 0, 1, 1])
        self.scores = np.asarray([0.1, 0.4, 0.6, 0.9])

    def test_perfect_metrics_and_schema(self):
        result = evaluate_binary(self.labels, self.scores, threshold=0.5)
        self.assertEqual(result["image_auroc"], 1.0)
        self.assertEqual(result["image_auprc"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["valid_rate"], 1.0)
        self.assertIn("industrial_point", result["evaluation_v2"])

    def test_target_recall_point(self):
        result = evaluate_binary(self.labels, self.scores, threshold=0.5, target_recall=0.99)
        self.assertEqual(result["fpr_at_target_recall"], 0.0)
        self.assertAlmostEqual(result["threshold_at_target_recall"], 0.6)

    def test_valid_mask_excludes_failed_response(self):
        result = evaluate_binary(
            self.labels,
            [0.1, float("nan"), 0.6, 0.9],
            threshold=0.5,
            valid_mask=[True, False, True, True],
        )
        self.assertEqual(result["n_total"], 4)
        self.assertEqual(result["n_valid"], 3)
        self.assertEqual(result["valid_rate"], 0.75)

    def test_external_predictions(self):
        result = evaluate_binary(
            self.labels,
            self.scores,
            threshold=0.5,
            predictions=[0, 1, 1, 1],
        )
        self.assertEqual(result["fp"], 1)
        self.assertAlmostEqual(result["fp_rate"], 0.5)

    def test_fit_threshold(self):
        self.assertAlmostEqual(fit_threshold(self.labels, self.scores), 0.6)
        self.assertAlmostEqual(
            fit_threshold(self.labels, self.scores, strategy="target_recall"),
            0.6,
        )

    def test_single_class_returns_nan_ranking(self):
        result = evaluate_binary([0, 0], [0.1, 0.2], threshold=0.5)
        self.assertTrue(math.isnan(result["image_auroc"]))
        self.assertTrue(math.isnan(result["image_auprc"]))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_binary([0], [0.1, 0.2], threshold=0.5)

    def test_latency_summary_ignores_nan(self):
        result = summarize_latency([1.0, 2.0, float("nan")])
        self.assertAlmostEqual(result["mean_ms"], 1.5)

    def test_legacy_metrics_wrapper_exposes_new_and_old_fields(self):
        result = binary_detection_metrics(self.labels, self.scores, 0.5)
        self.assertEqual(result["image_auroc"], 1.0)
        self.assertEqual(result["image_auprc"], 1.0)
        self.assertEqual(result["fn_rate"], 0.0)
        self.assertIn("evaluation_v2", result)
        self.assertEqual(latency_stats([1.0, 3.0])["mean_ms"], 2.0)

    def test_cloud_runtime_summary_and_parameter_count(self):
        class FakeParameter:
            def __init__(self, n, trainable):
                self.n = n
                self.requires_grad = trainable

            def numel(self):
                return self.n

        class FakeModel:
            def parameters(self):
                return iter([FakeParameter(8, True), FakeParameter(2, False)])

        params = count_model_parameters(FakeModel())
        self.assertEqual(params["total"], 10)
        self.assertEqual(params["trainable"], 8)

        runtime = summarize_inference([10.0, 20.0, float("nan")], FakeModel())
        self.assertEqual(runtime["n_inferences"], 3)
        self.assertEqual(runtime["n_valid_latency"], 2)
        self.assertAlmostEqual(runtime["inference_latency_ms"]["mean_ms"], 15.0)
        self.assertEqual(runtime["parameters"]["total"], 10)

    def test_missing_model_has_explicit_unknown_parameter_count(self):
        params = count_model_parameters(None)
        self.assertIsNone(params["total"])
        self.assertIsNone(params["total_m"])


if __name__ == "__main__":
    unittest.main()
