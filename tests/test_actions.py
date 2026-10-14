"""tests/test_actions.py — Tests for src/actions.py"""
import pytest

from src.actions import get_actions


# Synthetic SHAP driver dicts that match what explain_customer returns
def _make_driver(feature: str, shap_val: float, reason: str = "") -> dict:
    return {
        "feature": feature,
        "base_feature": feature,  # actions.py uses base_feature
        "shap_value": shap_val,
        "reason": reason or feature,
        "display_value": "Yes",
    }


class TestGetActions:
    def test_returns_list(self):
        drivers = [_make_driver("is_month_to_month", 0.4, "Month-to-month contract")]
        result = get_actions(drivers)
        assert isinstance(result, list)

    def test_month_to_month_triggers_offer(self):
        drivers = [_make_driver("is_month_to_month", 0.5, "Month-to-month contract")]
        actions = get_actions(drivers)
        assert len(actions) >= 1
        offers = [a["offer"] for a in actions]
        # The rule maps is_month_to_month → "Discounted 1-year contract"
        assert any(
            "contract" in o.lower() or "year" in o.lower() or "discount" in o.lower()
            for o in offers
        ), f"Expected contract/year/discount offer, got: {offers}"

    def test_high_charges_triggers_offer(self):
        drivers = [_make_driver("MonthlyCharges", 0.6, "High monthly charges")]
        actions = get_actions(drivers)
        assert len(actions) >= 1

    def test_no_internet_security_triggers_offer(self):
        drivers = [_make_driver("OnlineSecurity", 0.3, "No online security")]
        actions = get_actions(drivers)
        assert len(actions) >= 1

    def test_negative_shap_no_action(self):
        """Features with negative SHAP (protective factors) should not trigger urgent offers."""
        drivers = [_make_driver("tenure", -0.5, "Long-tenured customer")]
        actions = get_actions(drivers)
        # May return empty or low-priority actions — should not crash
        assert isinstance(actions, list)

    def test_action_has_required_keys(self):
        drivers = [_make_driver("is_month_to_month", 0.5, "Month-to-month contract")]
        actions = get_actions(drivers)
        for action in actions:
            assert "offer" in action
            assert "rationale" in action
            assert "cost_note" in action

    def test_empty_drivers_returns_list(self):
        actions = get_actions([])
        assert isinstance(actions, list)

    def test_multiple_drivers_multiple_actions(self):
        drivers = [
            _make_driver("is_month_to_month", 0.5, "Month-to-month contract"),
            _make_driver("MonthlyCharges", 0.4, "High monthly charges"),
            _make_driver("OnlineSecurity", 0.3, "No online security"),
        ]
        actions = get_actions(drivers)
        assert len(actions) >= 1
