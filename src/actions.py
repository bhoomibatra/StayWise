"""
src/actions.py
--------------
Rule-based action engine for StayWise.

Maps top SHAP churn drivers to specific retention offers.
Rules are priority-ordered; the first matching rule is applied.

All actions are illustrative business recommendations.
Effectiveness depends on context not captured in this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetentionAction:
    """A single retention offer recommendation."""
    offer: str
    rationale: str
    estimated_cost_note: str
    priority: int  # lower = higher priority


# ─── Rule definitions ─────────────────────────────────────────────────────────

# Rules map a tuple of (base_feature_name, shap_direction) to an action.
# shap_direction: "positive" = feature increases churn risk
_RULES: list[tuple[str, str, RetentionAction]] = [
    (
        "is_month_to_month", "positive",
        RetentionAction(
            offer="Discounted 1-year contract",
            rationale="Month-to-month contract is the strongest churn signal. "
                      "A locked-in term reduces likelihood of leaving.",
            estimated_cost_note="Discount ~10–15% off annual total",
            priority=1,
        ),
    ),
    (
        "Contract", "positive",
        RetentionAction(
            offer="Upgrade to longer-term contract with loyalty discount",
            rationale="Short or flexible contract increases churn probability.",
            estimated_cost_note="10% annual discount",
            priority=2,
        ),
    ),
    (
        "MonthlyCharges", "positive",
        RetentionAction(
            offer="Plan optimisation review or targeted discount",
            rationale="High monthly bill is a top churn driver. "
                      "A review may find a better-fit plan or justify a loyalty discount.",
            estimated_cost_note="Offer cost varies; typical $10–20/month reduction",
            priority=3,
        ),
    ),
    (
        "charges_per_month", "positive",
        RetentionAction(
            offer="Billing review and potential plan restructure",
            rationale="Effective monthly charge is high relative to tenure.",
            estimated_cost_note="Discount or service bundle",
            priority=4,
        ),
    ),
    (
        "TechSupport", "positive",
        RetentionAction(
            offer="Free TechSupport trial (3 months)",
            rationale="Customers without tech support churn more. "
                      "A free trial increases stickiness and perceived value.",
            estimated_cost_note="~$10/month for 3 months = $30 offer cost",
            priority=5,
        ),
    ),
    (
        "OnlineSecurity", "positive",
        RetentionAction(
            offer="Free Online Security bundle (3 months)",
            rationale="No security subscription is associated with higher churn.",
            estimated_cost_note="~$5/month for 3 months = $15 offer cost",
            priority=6,
        ),
    ),
    (
        "OnlineBackup", "positive",
        RetentionAction(
            offer="Free Online Backup trial (3 months)",
            rationale="Backup adds perceived value and switching cost.",
            estimated_cost_note="~$5/month for 3 months",
            priority=7,
        ),
    ),
    (
        "PaymentMethod", "positive",
        RetentionAction(
            offer="Autopay incentive ($5/month discount)",
            rationale="Electronic-check payers have higher churn. "
                      "Auto-pay reduces friction and churn intent.",
            estimated_cost_note="$5/month discount for 12 months = $60",
            priority=8,
        ),
    ),
    (
        "num_addons", "positive",
        RetentionAction(
            offer="Service bundle promotion (add 2 services at 50% off)",
            rationale="Low add-on count indicates low engagement. "
                      "A bundle increases perceived value and switching cost.",
            estimated_cost_note="~$15/month discount on bundle",
            priority=9,
        ),
    ),
    (
        "tenure", "positive",
        RetentionAction(
            offer="Early loyalty bonus or welcome incentive",
            rationale="Short tenure customers churn at higher rates. "
                      "An early loyalty reward improves retention.",
            estimated_cost_note="One-time $25–50 credit",
            priority=10,
        ),
    ),
    (
        "InternetService", "positive",
        RetentionAction(
            offer="Fiber upgrade package review",
            rationale="Fiber optic customers show higher churn — "
                      "possibly due to competition. Review package competitiveness.",
            estimated_cost_note="Competitive pricing adjustment",
            priority=11,
        ),
    ),
]

# Default fallback action
_DEFAULT_ACTION = RetentionAction(
    offer="General loyalty outreach",
    rationale="No specific high-priority driver identified. "
              "Proactive account manager contact is recommended.",
    estimated_cost_note="Internal labour cost only",
    priority=99,
)


def get_actions(
    drivers: list[dict[str, Any]],
    max_actions: int = 2,
) -> list[dict[str, Any]]:
    """
    Map top SHAP drivers to retention offers.

    Parameters
    ----------
    drivers : list of dicts from explain.explain_customer()
        Each dict has keys: base_feature, shap_value, reason, direction
    max_actions : int
        Maximum number of offers to return (default 2).

    Returns
    -------
    list of dicts: [{"offer": str, "rationale": str, "cost_note": str, "matched_on": str}]
    """
    if not drivers:
        return [_action_to_dict(_DEFAULT_ACTION, "no drivers")]

    matched: list[tuple[int, RetentionAction, str]] = []  # (priority, action, matched_feature)

    for driver in drivers:
        base_feat = driver.get("base_feature", "")
        shap_val = driver.get("shap_value", 0.0)
        direction = "positive" if shap_val > 0 else "negative"

        for feat_key, required_dir, action in _RULES:
            # Match on feature name (exact or prefix) and direction
            if feat_key == base_feat and required_dir == direction:
                matched.append((action.priority, action, base_feat))
                break

    # Sort by priority and take top N
    matched.sort(key=lambda x: x[0])
    selected = matched[:max_actions]

    if not selected:
        return [_action_to_dict(_DEFAULT_ACTION, "no matching rules")]

    return [_action_to_dict(a, feat) for _, a, feat in selected]


def _action_to_dict(action: RetentionAction, matched_on: str) -> dict[str, Any]:
    return {
        "offer": action.offer,
        "rationale": action.rationale,
        "cost_note": action.estimated_cost_note,
        "matched_on": matched_on,
    }


def get_action_for_reason(reason_key: str) -> RetentionAction:
    """Look up the action for a specific feature key (for testing)."""
    for feat_key, _, action in _RULES:
        if feat_key == reason_key:
            return action
    return _DEFAULT_ACTION
