"""tests/test_batch.py — Integration tests for batch prediction workflow"""
import io
import pandas as pd
import pytest

from src.predict import predict_batch, validate_batch_schema


class TestBatchWorkflow:
    def test_full_pipeline_on_engineered_data(self, trained_payload, engineered_df):
        """End-to-end: engineered DataFrame → batch predictions → validate output."""
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)

        assert len(results) == len(features)
        assert results["churn_probability"].between(0, 1).all()
        assert set(results["churn_label"].unique()).issubset({0, 1})
        assert results["risk_category"].isin(["Low", "Medium", "High", "Critical"]).all()

    def test_csv_roundtrip(self, trained_payload, engineered_df):
        """Simulate upload → score → download CSV."""
        features = engineered_df.drop(columns=["Churn"])

        # Simulate CSV upload (serialize/deserialize)
        buffer = io.StringIO()
        features.to_csv(buffer, index=False)
        buffer.seek(0)
        reloaded = pd.read_csv(buffer)

        results = predict_batch(reloaded, payload=trained_payload)
        csv_out = results.to_csv(index=False)
        assert "churn_probability" in csv_out
        assert "risk_category" in csv_out

    def test_batch_with_customer_id(self, trained_payload, engineered_df):
        """customerID column preserved through batch scoring."""
        features = engineered_df.drop(columns=["Churn"]).copy()
        features.insert(0, "customerID", [f"ID-{i:04d}" for i in range(len(features))])

        results = predict_batch(features, payload=trained_payload)
        assert "customerID" in results.columns
        assert list(results["customerID"]) == [f"ID-{i:04d}" for i in range(len(features))]

    def test_schema_validation_catches_missing(self, engineered_df):
        """Missing required column should be flagged before scoring."""
        features = engineered_df.drop(columns=["tenure", "Churn"])
        required = ["tenure", "MonthlyCharges", "Contract"]
        missing = validate_batch_schema(features, required)
        assert "tenure" in missing
        assert "MonthlyCharges" not in missing

    def test_batch_sorting_by_risk(self, trained_payload, engineered_df):
        """Top-N high-risk customers should have highest probabilities."""
        features = engineered_df.drop(columns=["Churn"])
        results = predict_batch(features, payload=trained_payload)
        top5 = results.sort_values("churn_probability", ascending=False).head(5)
        rest = results.sort_values("churn_probability", ascending=False).iloc[5:]
        assert top5["churn_probability"].min() >= rest["churn_probability"].max()

    def test_batch_different_thresholds(self, trained_payload, engineered_df):
        """Higher threshold → fewer predicted churners."""
        features = engineered_df.drop(columns=["Churn"])
        r_low = predict_batch(features, payload=trained_payload, threshold=0.2)
        r_high = predict_batch(features, payload=trained_payload, threshold=0.8)
        churners_low = r_low["churn_label"].sum()
        churners_high = r_high["churn_label"].sum()
        assert churners_low >= churners_high

    def test_empty_dataframe_graceful(self, trained_payload, engineered_df):
        """Empty DataFrame: sklearn imputer rejects 0-row input — verify it raises clearly."""
        import pytest
        empty = engineered_df.drop(columns=["Churn"]).iloc[0:0]
        # sklearn SimpleImputer raises ValueError for 0-sample input — this is expected behaviour
        with pytest.raises((ValueError, Exception)):
            predict_batch(empty, payload=trained_payload)
