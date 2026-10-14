"""tests/test_train.py — Smoke tests for src/train.py"""
import numpy as np
import pytest


class TestTrainOnData:
    def test_returns_pipeline_and_metadata(self, trained_payload):
        assert "pipeline" in trained_payload
        assert "metadata" in trained_payload

    def test_metadata_has_required_keys(self, trained_payload):
        meta = trained_payload["metadata"]
        for key in ["best_model_name", "threshold", "numeric_features",
                    "categorical_features", "test_metrics"]:
            assert key in meta, f"Missing metadata key: {key}"

    def test_threshold_in_valid_range(self, trained_payload):
        t = trained_payload["metadata"]["threshold"]
        assert 0.0 <= t <= 1.0

    def test_best_model_name_is_string(self, trained_payload):
        assert isinstance(trained_payload["metadata"]["best_model_name"], str)
        assert len(trained_payload["metadata"]["best_model_name"]) > 0

    def test_test_metrics_have_pr_auc(self, trained_payload):
        metrics = trained_payload["metadata"]["test_metrics"]
        assert "test_pr_auc" in metrics
        assert 0.0 <= metrics["test_pr_auc"] <= 1.0

    def test_pipeline_can_predict_proba(self, trained_payload, split_data):
        X_train, X_val, X_test, _, _, _ = split_data
        probs = trained_payload["pipeline"].predict_proba(X_test)[:, 1]
        assert probs.shape[0] == len(X_test)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_numeric_features_non_empty(self, trained_payload):
        assert len(trained_payload["metadata"]["numeric_features"]) > 0

    def test_categorical_features_non_empty(self, trained_payload):
        assert len(trained_payload["metadata"]["categorical_features"]) > 0


class TestBuildPreprocessor:
    def test_preprocessor_fits_and_transforms(self, split_data):
        from src.features import get_feature_lists
        from src.train import build_preprocessor

        X_train, _, _, _, _, _ = split_data
        feature_lists = get_feature_lists(X_train)
        preprocessor = build_preprocessor(
            feature_lists["numeric"],
            feature_lists["categorical"],
        )
        transformed = preprocessor.fit_transform(X_train)
        assert transformed.shape[0] == len(X_train)
        assert transformed.shape[1] > 0
        assert not np.isnan(transformed).any()
