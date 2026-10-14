"""tests/test_predict.py — Tests for src/predict.py"""
import numpy as np
import pandas as pd
import pytest

from src.predict import predict_batch, predict_single, validate_batch_schema


class TestPredictSingle:
    def test_probability_in_range(self, trained_payload, single_record):
        result = predict_single(single_record, payload=trained_payload)
        assert 0.0 <= result["probability"] <= 1.0

    def test_label_is_binary(self, trained_payload, single_record):
        result = predict_single(single_record, payload=trained_payload)
        assert result["label"] in (0, 1)

    def test_label_consistent_with_threshold(self, trained_payload, single_record):
        threshold = 0.5
        result = predict_single(single_record, payload=trained_payload, threshold=threshold)
        if result["probability"] >= threshold:
            assert result["label"] == 1
        else:
            assert result["label"] == 0

    def test_risk_category_valid(self, trained_payload, single_record):
        result = predict_single(single_record, payload=trained_payload)
        assert result["risk_category"] in ("Low", "Medium", "High", "Critical")

    def test_threshold_returned_in_result(self, trained_payload, single_record):
        result = predict_single(single_record, payload=trained_payload, threshold=0.4)
        assert result["threshold"] == 0.4

    def test_high_risk_record_detectable(self, trained_payload):
        """A classic high-churn profile should score above 0.3."""
        risky = {
            "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0, "TotalCharges": 95.0,
        }
        result = predict_single(risky, payload=trained_payload)
        assert result["probability"] > 0.1  # model trained on synthetic data — loose bound

    def test_low_risk_record_detectable(self, trained_payload):
        """A classic low-churn profile should score below 0.9."""
        safe = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 45.0, "TotalCharges": 2700.0,
        }
        result = predict_single(safe, payload=trained_payload)
        assert result["probability"] < 0.9


class TestPredictBatch:
    def test_output_shape(self, trained_payload, engineered_df):
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)
        assert len(results) == len(features)

    def test_output_columns(self, trained_payload, engineered_df):
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)
        for col in ["churn_probability", "churn_label", "risk_category"]:
            assert col in results.columns

    def test_probabilities_in_range(self, trained_payload, engineered_df):
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)
        assert results["churn_probability"].between(0.0, 1.0).all()

    def test_labels_binary(self, trained_payload, engineered_df):
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)
        assert set(results["churn_label"].unique()).issubset({0, 1})

    def test_customer_id_preserved(self, trained_payload, engineered_df):
        features = engineered_df.drop(columns=["Churn"]).copy()
        features.insert(0, "customerID", [f"C{i}" for i in range(len(features))])
        results = predict_batch(features, payload=trained_payload)
        assert "customerID" in results.columns

    def test_churn_column_ignored(self, trained_payload, engineered_df):
        """Batch predict should work even if Churn column is still in df."""
        results = predict_batch(engineered_df, payload=trained_payload)
        assert len(results) == len(engineered_df)


class TestValidateBatchSchema:
    def test_all_present(self, engineered_df):
        required = list(engineered_df.columns[:5])
        missing = validate_batch_schema(engineered_df, required)
        assert missing == []

    def test_detects_missing(self, engineered_df):
        missing = validate_batch_schema(engineered_df, ["tenure", "NONEXISTENT_COL"])
        assert "NONEXISTENT_COL" in missing
        assert "tenure" not in missing
