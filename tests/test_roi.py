"""tests/test_roi.py — Tests for src/roi.py"""
import pytest

from src.roi import customer_value, expected_saving, format_roi_summary


class TestCustomerValue:
    def test_returns_positive_float(self):
        val = customer_value(monthly_charges=65.0, current_tenure=12)
        assert isinstance(val, float)
        assert val > 0

    def test_higher_monthly_higher_value(self):
        v1 = customer_value(monthly_charges=50.0, current_tenure=12)
        v2 = customer_value(monthly_charges=100.0, current_tenure=12)
        assert v2 > v1

    def test_longer_tenure_higher_value(self):
        v1 = customer_value(monthly_charges=65.0, current_tenure=60)
        v2 = customer_value(monthly_charges=65.0, current_tenure=6)
        # shorter tenure → more remaining months → higher value
        assert v2 > v1

    def test_zero_tenure_still_positive(self):
        val = customer_value(monthly_charges=65.0, current_tenure=0)
        assert val > 0


class TestExpectedSaving:
    def test_returns_dict_with_required_keys(self):
        cval = customer_value(65.0, current_tenure=12)
        result = expected_saving(p_churn=0.8, cust_value=cval)
        for key in ["expected_saving", "offer_cost", "success_rate", "recommend_action"]:
            assert key in result

    def test_high_churn_prob_positive_saving(self):
        """High churn probability → expected saving should be positive (worth acting)."""
        cval = customer_value(65.0, current_tenure=6)
        result = expected_saving(p_churn=0.9, cust_value=cval)
        assert result["expected_saving"] > 0
        assert result["recommend_action"] is True

    def test_zero_churn_no_action(self):
        """Zero churn probability → no expected saving."""
        cval = customer_value(65.0, current_tenure=12)
        result = expected_saving(p_churn=0.0, cust_value=cval)
        assert result["expected_saving"] <= 0
        assert result["recommend_action"] is False

    def test_expected_saving_formula(self):
        """
        expected_saving = p_churn × success_rate × cust_value − offer_cost
        """
        cval = 1000.0
        p = 0.8
        result = expected_saving(p_churn=p, cust_value=cval)
        sr = result["success_rate"]
        oc = result["offer_cost"]
        computed = p * sr * cval - oc
        assert abs(result["expected_saving"] - computed) < 0.01


class TestFormatRoiSummary:
    def test_returns_string(self):
        cval = customer_value(65.0, current_tenure=12)
        roi = expected_saving(0.7, cust_value=cval)
        summary = format_roi_summary(roi)
        assert isinstance(summary, str)
        assert len(summary) > 0
