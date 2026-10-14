"""tests/test_features.py — Tests for src/features.py"""
import numpy as np
import pandas as pd
import pytest

from src.features import engineer


class TestTenureGroup:
    @pytest.mark.parametrize("tenure, expected", [
        (0,  "0-12"),
        (1,  "0-12"),
        (12, "0-12"),
        (13, "13-24"),
        (24, "13-24"),
        (25, "25-48"),
        (48, "25-48"),
        (49, "49-60"),
        (60, "49-60"),
        (61, "61+"),
        (72, "61+"),
    ])
    def test_tenure_group_boundaries(self, clean_df, tenure, expected):
        df = clean_df.copy()
        df["tenure"] = tenure
        out = engineer(df)
        assert (out["tenure_group"] == expected).all(), \
            f"tenure={tenure} → expected {expected}, got {out['tenure_group'].unique()}"


class TestNumAddons:
    def test_num_addons_zero(self, clean_df):
        df = clean_df.copy()
        addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
        for col in addon_cols:
            df[col] = "No"
        out = engineer(df)
        assert (out["num_addons"] == 0).all()

    def test_num_addons_max(self, clean_df):
        df = clean_df.copy()
        addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
        for col in addon_cols:
            df[col] = "Yes"
        out = engineer(df)
        assert (out["num_addons"] == 6).all()

    def test_num_addons_range(self, engineered_df):
        assert engineered_df["num_addons"].between(0, 6).all()


class TestChargesPerMonth:
    def test_charges_per_month_formula(self, clean_df):
        df = clean_df.copy()
        df["MonthlyCharges"] = 50.0
        df["TotalCharges"] = 100.0
        df["tenure"] = 2
        out = engineer(df)
        # charges_per_month = TotalCharges / max(tenure, 1)
        expected = 100.0 / 2
        assert np.allclose(out["charges_per_month"], expected)

    def test_zero_tenure_no_div_zero(self, clean_df):
        df = clean_df.copy()
        df["tenure"] = 0
        df["TotalCharges"] = 50.0
        out = engineer(df)
        assert not out["charges_per_month"].isna().any()
        assert (out["charges_per_month"] == 50.0).all()

    def test_charges_positive(self, engineered_df):
        assert (engineered_df["charges_per_month"] >= 0).all()


class TestIsMonthToMonth:
    def test_month_to_month_flag_true(self, clean_df):
        df = clean_df.copy()
        df["Contract"] = "Month-to-month"
        out = engineer(df)
        assert (out["is_month_to_month"] == 1).all()

    def test_other_contract_flag_false(self, clean_df):
        df = clean_df.copy()
        df["Contract"] = "Two year"
        out = engineer(df)
        assert (out["is_month_to_month"] == 0).all()


class TestEngineerIdempotent:
    def test_column_count_increases(self, clean_df):
        out = engineer(clean_df.copy())
        assert len(out.columns) > len(clean_df.columns)

    def test_new_columns_present(self, engineered_df):
        for col in ["tenure_group", "charges_per_month", "num_addons", "is_month_to_month"]:
            assert col in engineered_df.columns, f"Missing: {col}"
