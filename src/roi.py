"""
src/roi.py
----------
ROI and expected-saving computation for StayWise retention offers.

All business assumptions are read from config.yaml and are editable in the UI.

Formula
-------
    customer_value   = MonthlyCharges × expected_remaining_months
    expected_saving  = P(churn) × success_rate × customer_value − offer_cost
    recommend_action = expected_saving > 0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def customer_value(
    monthly_charges: float,
    current_tenure: int,
    config: dict | None = None,
    config_path: Path | None = None,
) -> float:
    """
    Estimate the remaining lifetime value of a customer.

    Parameters
    ----------
    monthly_charges : float
        Customer's current monthly bill.
    current_tenure : int
        Months already with the company.
    config : dict | None
        Pre-loaded config dict (avoids repeated disk reads).
    config_path : Path | None

    Returns
    -------
    float — estimated remaining customer value (currency units from config).
    """
    if config is None:
        config = _load_config(config_path)

    biz = config["business"]
    max_months = biz["max_remaining_months"]
    currency_factor = biz["currency_conversion"]

    # Simple model: assume a typical lifecycle of 72 months (6 years)
    # remaining = min(max_months, max(0, 72 - current_tenure))
    TYPICAL_LIFECYCLE = 72
    remaining_months = min(max_months, max(0, TYPICAL_LIFECYCLE - current_tenure))

    return round(monthly_charges * currency_factor * remaining_months, 2)


def expected_saving(
    p_churn: float,
    cust_value: float,
    config: dict | None = None,
    config_path: Path | None = None,
    offer_cost_override: float | None = None,
    success_rate_override: float | None = None,
) -> dict[str, Any]:
    """
    Compute the expected saving from offering a retention incentive.

    Parameters
    ----------
    p_churn : float
        Predicted churn probability in [0, 1].
    cust_value : float
        Estimated remaining customer value.
    config : dict | None
    config_path : Path | None
    offer_cost_override : float | None
        Override config offer_cost (for What-If UI).
    success_rate_override : float | None
        Override config success_rate (for What-If UI).

    Returns
    -------
    dict with keys:
        offer_cost, success_rate, customer_value, expected_saving,
        recommend_action (bool), breakeven_probability (float)
    """
    if config is None:
        config = _load_config(config_path)

    biz = config["business"]
    offer_cost = offer_cost_override if offer_cost_override is not None else biz["offer_cost"]
    success_rate = success_rate_override if success_rate_override is not None else biz["success_rate"]

    saving = p_churn * success_rate * cust_value - offer_cost
    recommend = saving > 0.0

    # Break-even: minimum p_churn to justify the offer
    if success_rate * cust_value > 0:
        breakeven_p = offer_cost / (success_rate * cust_value)
    else:
        breakeven_p = 1.0

    return {
        "offer_cost": round(offer_cost, 2),
        "success_rate": round(success_rate, 4),
        "customer_value": round(cust_value, 2),
        "expected_saving": round(saving, 2),
        "recommend_action": recommend,
        "breakeven_probability": round(min(breakeven_p, 1.0), 4),
        "currency_symbol": biz.get("currency_symbol", "$"),
    }


def format_roi_summary(roi: dict[str, Any]) -> str:
    """Return a human-readable ROI summary string."""
    sym = roi["currency_symbol"]
    if roi["recommend_action"]:
        return (
            f"✅ Recommended — Expected saving: {sym}{roi['expected_saving']:.2f}\n"
            f"   (P(churn)×{roi['success_rate']:.0%} success × {sym}{roi['customer_value']:.0f} value "
            f"− {sym}{roi['offer_cost']:.0f} offer cost)"
        )
    else:
        return (
            f"❌ No action recommended — Expected saving: {sym}{roi['expected_saving']:.2f} (negative)\n"
            f"   Breakeven churn probability: {roi['breakeven_probability']:.1%}"
        )
