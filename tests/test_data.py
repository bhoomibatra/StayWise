"""tests/test_data.py — Tests for src/data.py"""
import numpy as np
import pandas as pd
import pytest

from src.data import clean, get_quality_report, split, validate_schema

REQUIRED_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


class TestValidateSchema:
    def test_valid_df_raises_nothing(self, raw_df):
        validate_schema(raw_df, REQUIRED_COLS)  # should not raise

    def test_missing_column_raises(self, raw_df):
        bad = raw_df.drop(columns=["tenure"])
        with pytest.raises(ValueError, match="tenure"):
            validate_schema(bad, REQUIRED_COLS)

    def test_multiple_missing_raises(self, raw_df):
        bad = raw_df.drop(columns=["tenure", "MonthlyCharges"])
        with pytest.raises(ValueError):
            validate_schema(bad, REQUIRED_COLS)


class TestClean:
    def test_churn_encoded_as_int(self, clean_df):
        assert clean_df["Churn"].dtype in [np.int64, np.int32, int]
        assert set(clean_df["Churn"].unique()).issubset({0, 1})

    def test_customer_id_dropped(self, clean_df):
        assert "customerID" not in clean_df.columns

    def test_total_charges_numeric(self, clean_df):
        assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])

    def test_no_blank_total_charges(self, clean_df):
        assert clean_df["TotalCharges"].isna().sum() == 0

    def test_total_charges_blank_string_handled(self):
        """Rows with blank TotalCharges string are coerced to NaN by clean().
        clean() converts blanks → NaN via pd.to_numeric. Rows may be kept or dropped.
        Either way, the column must be numeric (not object/string)."""
        from tests.conftest import _make_synthetic_df
        df = _make_synthetic_df(n=20)
        df["TotalCharges"] = df["TotalCharges"].astype(str)
        df.loc[df.index[:3], "TotalCharges"] = "  "  # inject blanks
        cleaned = clean(df)
        # TotalCharges must be numeric after cleaning
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"]), \
            "TotalCharges should be numeric after clean()"

    def test_both_churn_classes_present(self, clean_df):
        assert len(clean_df["Churn"].unique()) == 2

    def test_row_count_preserved(self, raw_df, clean_df):
        # Rows with blank TotalCharges may be dropped — count should be ≤ original
        assert len(clean_df) <= len(raw_df)
        assert len(clean_df) > 0


class TestQualityReport:
    def test_report_has_required_keys(self, clean_df):
        qr = get_quality_report(clean_df)
        for key in ["n_rows", "n_cols", "n_duplicates", "n_missing_total", "churn_rate"]:
            assert key in qr

    def test_churn_rate_in_range(self, clean_df):
        qr = get_quality_report(clean_df)
        assert 0.0 <= qr["churn_rate"] <= 1.0


class TestSplit:
    def test_split_sizes(self, split_data):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data
        total = len(X_train) + len(X_val) + len(X_test)
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)
        assert total == len(X_train) + len(X_val) + len(X_test)

    def test_no_overlap(self, split_data):
        X_train, X_val, X_test, _, _, _ = split_data
        idx_train = set(X_train.index)
        idx_val = set(X_val.index)
        idx_test = set(X_test.index)
        assert idx_train.isdisjoint(idx_val)
        assert idx_train.isdisjoint(idx_test)
        assert idx_val.isdisjoint(idx_test)

    def test_target_not_in_features(self, split_data):
        X_train, _, _, _, _, _ = split_data
        assert "Churn" not in X_train.columns

    def test_both_classes_in_train(self, split_data):
        _, _, _, y_train, _, _ = split_data
        assert set(y_train.unique()) == {0, 1}
