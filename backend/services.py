"""Service Layer for StayWise Enterprise API.

Handles:
- Dataset pre-scoring using the trained Scikit-Learn pipeline
- Customer risk driver extraction & protective factor identification
- Filtering, sorting, and pagination across 7,043 customers
- Aggregated executive metrics, churn distributions, and analytics
- Model performance benchmarks and test evaluation data
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.preprocess import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from backend.database import list_tasks, list_notes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_JSON_PATH = MODELS_DIR / "metrics.json"
EVAL_DETAILS_PATH = MODELS_DIR / "evaluation_details.json"

# In-memory cached dataset state
_CACHED_DF: pd.DataFrame | None = None
_CACHED_METRICS: dict[str, Any] | None = None
_CACHED_PIPELINE: Any | None = None
_CACHED_THRESHOLDS = {"high": 0.60, "medium": 0.35}


def get_pipeline():
    """Load and cache the trained Scikit-learn pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")
        _CACHED_PIPELINE = joblib.load(BEST_MODEL_PATH)
    return _CACHED_PIPELINE


def get_metrics_data() -> dict[str, Any]:
    """Load model evaluation metrics JSON."""
    global _CACHED_METRICS
    if _CACHED_METRICS is None:
        if METRICS_JSON_PATH.exists():
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                _CACHED_METRICS = json.load(f)
        else:
            _CACHED_METRICS = {}
    return _CACHED_METRICS


def get_scored_dataset(high_thresh: float = 0.60, med_thresh: float = 0.35, force_refresh: bool = False) -> pd.DataFrame:
    """Load the dataset and score all 7,043 customer records with the trained pipeline."""
    global _CACHED_DF, _CACHED_THRESHOLDS

    if (
        not force_refresh
        and _CACHED_DF is not None
        and _CACHED_THRESHOLDS["high"] == high_thresh
        and _CACHED_THRESHOLDS["medium"] == med_thresh
    ):
        return _CACHED_DF

    pipeline = get_pipeline()
    raw_df = load_raw_data(DEFAULT_DATA_PATH)

    # Validate feature inputs
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_input = raw_df[feature_cols].copy()

    # Model inference
    probs = pipeline.predict_proba(X_input)[:, 1]
    preds = pipeline.predict(X_input)

    df = raw_df.copy()
    df["churn_probability"] = np.round(probs, 4)
    df["churn_probability_pct"] = np.round(probs * 100, 1)
    df["predicted_churn"] = np.where(preds == 1, "Churn", "Stay")

    # Risk categorization
    conditions = [
        df["churn_probability"] >= high_thresh,
        df["churn_probability"] >= med_thresh,
    ]
    choices = ["High Risk", "Medium Risk"]
    df["risk_category"] = np.select(conditions, choices, default="Low Risk")

    # Create tenure bracket
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df["tenure_band"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    _CACHED_DF = df
    _CACHED_THRESHOLDS = {"high": high_thresh, "medium": med_thresh}
    return _CACHED_DF


def extract_risk_drivers(row: pd.Series) -> dict[str, list[dict[str, str]]]:
    """Generate empirical, non-causal risk indicators and protective factors for a customer."""
    risk_factors: list[dict[str, str]] = []
    protective_factors: list[dict[str, str]] = []

    contract = str(row.get("Contract", ""))
    if contract == "Month-to-month":
        risk_factors.append({
            "title": "Month-to-Month Contract",
            "severity": "High",
            "detail": "Associated with highest historical attrition (~42.7% churn rate in dataset) due to absence of commitment terms."
        })
    elif contract in ["One year", "Two year"]:
        protective_factors.append({
            "title": f"{contract} Term Commitment",
            "detail": "Long-term agreement associated with significantly lower churn rates (< 11%)."
        })

    tenure = float(row.get("tenure", 0))
    if tenure <= 6:
        risk_factors.append({
            "title": f"Early Tenure Hazard Window ({int(tenure)} mo)",
            "severity": "High",
            "detail": "Customers in months 1-6 experience highest voluntary dropout during onboarding."
        })
    elif tenure <= 18:
        risk_factors.append({
            "title": f"Mid-Onboarding Phase ({int(tenure)} mo)",
            "severity": "Medium",
            "detail": "Brand habituation still forming; sensitive to service quality disruptions."
        })
    elif tenure >= 48:
        protective_factors.append({
            "title": f"Established Loyalty ({int(tenure)} mo)",
            "detail": "Multi-year customer tenure indicates high operational inertia and brand trust."
        })

    internet = str(row.get("InternetService", ""))
    tech_support = str(row.get("TechSupport", ""))
    online_sec = str(row.get("OnlineSecurity", ""))

    if internet == "Fiber optic":
        if tech_support == "No" or online_sec == "No":
            risk_factors.append({
                "title": "Unbundled Fiber Optic Service",
                "severity": "High",
                "detail": "High-speed line without complementary technical support or security add-ons correlates with elevated dissatisfaction."
            })
        else:
            protective_factors.append({
                "title": "Protected Fiber Optic Bundle",
                "detail": "Subscriber has bundled support & security, increasing product stickiness."
            })

    payment = str(row.get("PaymentMethod", ""))
    if "Electronic check" in payment:
        risk_factors.append({
            "title": "Manual Electronic Check Billing",
            "severity": "Medium",
            "detail": "Manual payment methods exhibit higher attrition (~45%) than automated recurring billing channels."
        })
    elif "automatic" in payment.lower():
        protective_factors.append({
            "title": f"Automated Payment ({payment})",
            "detail": "Recurring auto-pay eliminates monthly payment friction and involuntary churn."
        })

    monthly = float(row.get("MonthlyCharges", 0))
    if monthly >= 80.0:
        risk_factors.append({
            "title": f"Premium Monthly Charge (${monthly:.2f}/mo)",
            "severity": "Medium",
            "detail": "Above-average monthly expenditure increases price sensitivity and competitor comparison."
        })

    partner = str(row.get("Partner", ""))
    dependents = str(row.get("Dependents", ""))
    if partner == "Yes" and dependents == "Yes":
        protective_factors.append({
            "title": "Family Account Ecosystem",
            "detail": "Multi-person households exhibit greater switching friction."
        })

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }


def get_recommended_actions(row: pd.Series, risk_cat: str) -> list[dict[str, str]]:
    """Generate business-aligned retention recommendations."""
    actions = []
    contract = str(row.get("Contract", ""))
    payment = str(row.get("PaymentMethod", ""))
    tech_support = str(row.get("TechSupport", ""))
    online_sec = str(row.get("OnlineSecurity", ""))
    tenure = float(row.get("tenure", 0))

    if contract == "Month-to-month":
        actions.append({
            "type": "Contract Upgrade Offer",
            "title": "Offer 1-Year Commitment Incentive",
            "description": "Propose a 15% promotional bill credit or speed tier boost in return for signing an annual agreement.",
            "priority": "High" if risk_cat == "High Risk" else "Medium",
        })

    if tech_support == "No" or online_sec == "No":
        actions.append({
            "type": "Value Add-on Onboarding",
            "title": "Deploy Premium Support Trial",
            "description": "Provision 6 months of complimentary Tech Support & Online Security to reduce frustration.",
            "priority": "Medium",
        })

    if "Electronic check" in payment:
        actions.append({
            "type": "Auto-Pay Migration Bonus",
            "title": "Automated Billing Incentive",
            "description": "Offer a recurring $5 monthly statement credit upon enrolling in automatic ACH or credit card payment.",
            "priority": "Low",
        })

    if tenure <= 6:
        actions.append({
            "type": "Proactive Onboarding Outreach",
            "title": "CSM Early Lifecycle Health Check",
            "description": "Schedule a proactive customer success onboarding check-in call to ensure hardware & speed satisfaction.",
            "priority": "High",
        })

    if not actions:
        actions.append({
            "type": "Loyalty Review",
            "title": "Account Health & Relationship Review",
            "description": "Perform standard customer satisfaction check and provide loyalty milestone recognition.",
            "priority": "Low",
        })

    return actions


def get_overview_data(high_thresh: float = 0.60, med_thresh: float = 0.35) -> dict[str, Any]:
    """Calculate executive metrics and aggregate distributions."""
    df = get_scored_dataset(high_thresh, med_thresh)
    metrics_info = get_metrics_data()
    best_perf = metrics_info.get("metrics_summary", {}).get("Logistic Regression", {})

    total_customers = len(df)
    pred_churn_count = int((df["predicted_churn"] == "Churn").sum())
    pred_stay_count = total_customers - pred_churn_count
    pred_churn_rate = round((pred_churn_count / total_customers) * 100, 2)

    has_actuals = "Churn" in df.columns
    if has_actuals:
        actual_churn_count = int((df["Churn"].astype(str).str.lower() == "yes").sum())
        actual_churn_rate = round((actual_churn_count / total_customers) * 100, 2)
    else:
        actual_churn_count = 0
        actual_churn_rate = 0.0

    high_risk_count = int((df["risk_category"] == "High Risk").sum())
    med_risk_count = int((df["risk_category"] == "Medium Risk").sum())
    low_risk_count = int((df["risk_category"] == "Low Risk").sum())

    # Churn distribution by Contract
    contract_data = []
    for contract, group in df.groupby("Contract"):
        c_total = len(group)
        c_pred_churn = int((group["predicted_churn"] == "Churn").sum())
        c_act_churn = int((group["Churn"].astype(str).str.lower() == "yes").sum()) if has_actuals else 0
        contract_data.append({
            "contract": contract,
            "total": c_total,
            "pred_churn": c_pred_churn,
            "pred_churn_rate": round((c_pred_churn / c_total) * 100, 1),
            "actual_churn": c_act_churn,
            "actual_churn_rate": round((c_act_churn / c_total) * 100, 1) if has_actuals else 0,
        })

    # Churn distribution by Tenure Band
    tenure_data = []
    for band, group in df.groupby("tenure_band", observed=False):
        t_total = len(group)
        if t_total == 0:
            continue
        t_pred_churn = int((group["predicted_churn"] == "Churn").sum())
        t_act_churn = int((group["Churn"].astype(str).str.lower() == "yes").sum()) if has_actuals else 0
        tenure_data.append({
            "band": str(band),
            "total": t_total,
            "pred_churn": t_pred_churn,
            "pred_churn_rate": round((t_pred_churn / t_total) * 100, 1),
            "actual_churn": t_act_churn,
            "actual_churn_rate": round((t_act_churn / t_total) * 100, 1) if has_actuals else 0,
        })

    # Churn distribution by Payment Method
    payment_data = []
    for pay, group in df.groupby("PaymentMethod"):
        p_total = len(group)
        p_pred_churn = int((group["predicted_churn"] == "Churn").sum())
        payment_data.append({
            "payment": pay.replace(" (automatic)", " (Auto)"),
            "total": p_total,
            "pred_churn": p_pred_churn,
            "pred_churn_rate": round((p_pred_churn / p_total) * 100, 1),
        })

    # Tasks stats from SQLite
    all_tasks = list_tasks()
    open_tasks = sum(1 for t in all_tasks if t["status"] == "Open")
    in_progress_tasks = sum(1 for t in all_tasks if t["status"] == "In Progress")
    completed_tasks = sum(1 for t in all_tasks if t["status"] == "Completed")

    return {
        "kpis": {
            "total_customers": total_customers,
            "predicted_churners": pred_churn_count,
            "predicted_non_churners": pred_stay_count,
            "predicted_churn_rate": pred_churn_rate,
            "high_risk_count": high_risk_count,
            "high_risk_pct": round((high_risk_count / total_customers) * 100, 1),
            "medium_risk_count": med_risk_count,
            "medium_risk_pct": round((med_risk_count / total_customers) * 100, 1),
            "low_risk_count": low_risk_count,
            "low_risk_pct": round((low_risk_count / total_customers) * 100, 1),
            "actual_churn_count": actual_churn_count,
            "actual_churn_rate": actual_churn_rate,
            "model_accuracy": round(float(best_perf.get("accuracy", 0.8055)) * 100, 1),
            "model_roc_auc": round(float(best_perf.get("roc_auc", 0.8411)), 4),
            "model_f1": round(float(best_perf.get("f1", 0.6040)), 4),
            "model_recall": round(float(best_perf.get("recall", 0.5588)) * 100, 1),
            "active_tasks_count": open_tasks + in_progress_tasks,
            "completed_tasks_count": completed_tasks,
        },
        "risk_distribution": [
            {"category": "High Risk", "count": high_risk_count, "pct": round((high_risk_count / total_customers) * 100, 1), "color": "#ef4444"},
            {"category": "Medium Risk", "count": med_risk_count, "pct": round((med_risk_count / total_customers) * 100, 1), "color": "#f59e0b"},
            {"category": "Low Risk", "count": low_risk_count, "pct": round((low_risk_count / total_customers) * 100, 1), "color": "#10b981"},
        ],
        "prediction_distribution": [
            {"label": "Predicted Churn", "count": pred_churn_count, "pct": pred_churn_rate, "color": "#f43f5e"},
            {"label": "Predicted Stay", "count": pred_stay_count, "pct": round(100 - pred_churn_rate, 2), "color": "#0ea5e9"},
        ],
        "contract_breakdown": contract_data,
        "tenure_breakdown": tenure_data,
        "payment_breakdown": payment_data,
        "system_status": {
            "model_name": "Logistic Regression (C=10.0)",
            "pipeline_status": "Active & In Sync",
            "last_scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "data_source": "IBM Telco Dataset (7,043 Records)",
            "calibration_status": "Empirical logistic scores mapped to configurable triage thresholds",
        },
    }


def query_customers(
    search: str | None = None,
    predicted_churn: str | None = None,
    risk_category: str | None = None,
    contract: str | None = None,
    sort_by: str = "churn_probability",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 25,
    high_thresh: float = 0.60,
    med_thresh: float = 0.35,
) -> dict[str, Any]:
    """Search, filter, sort, and paginate customers."""
    df = get_scored_dataset(high_thresh, med_thresh)

    filtered = df.copy()

    # Search by customerID
    if search:
        s = search.strip().upper()
        filtered = filtered[filtered["customerID"].str.upper().str.contains(s, na=False)]

    # Filters
    if predicted_churn and predicted_churn != "All":
        filtered = filtered[filtered["predicted_churn"] == predicted_churn]

    if risk_category and risk_category != "All":
        filtered = filtered[filtered["risk_category"] == risk_category]

    if contract and contract != "All":
        filtered = filtered[filtered["Contract"] == contract]

    # Sorting
    ascending = (sort_order.lower() == "asc")
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered = filtered.sort_values(by="churn_probability", ascending=False)

    total_records = len(filtered)
    total_pages = max(1, (total_records + page_size - 1) // page_size)
    current_page = max(1, min(page, total_pages))

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    sliced = filtered.iloc[start_idx:end_idx]

    # Check which customers have active retention tasks in SQLite
    tasks = list_tasks()
    active_task_customer_ids = {t["customer_id"] for t in tasks if t["status"] in ("Open", "In Progress")}

    results = []
    for _, row in sliced.iterrows():
        cid = row["customerID"]
        results.append({
            "customer_id": cid,
            "predicted_churn": row["predicted_churn"],
            "churn_probability": float(row["churn_probability"]),
            "churn_probability_pct": float(row["churn_probability_pct"]),
            "risk_category": row["risk_category"],
            "contract": row["Contract"],
            "tenure": int(row["tenure"]),
            "monthly_charges": float(row["MonthlyCharges"]),
            "total_charges": float(row["TotalCharges"]) if not pd.isna(row["TotalCharges"]) else None,
            "payment_method": row["PaymentMethod"],
            "internet_service": row["InternetService"],
            "historical_churn": row.get("Churn", None),
            "has_active_retention_task": cid in active_task_customer_ids,
        })

    return {
        "items": results,
        "pagination": {
            "total": total_records,
            "page": current_page,
            "page_size": page_size,
            "total_pages": total_pages,
        },
    }


def export_customers_csv(
    search: str | None = None,
    predicted_churn: str | None = None,
    risk_category: str | None = None,
    contract: str | None = None,
    high_thresh: float = 0.60,
    med_thresh: float = 0.35,
) -> str:
    """Export filtered customer list as CSV string."""
    df = get_scored_dataset(high_thresh, med_thresh)
    filtered = df.copy()

    if search:
        s = search.strip().upper()
        filtered = filtered[filtered["customerID"].str.upper().str.contains(s, na=False)]
    if predicted_churn and predicted_churn != "All":
        filtered = filtered[filtered["predicted_churn"] == predicted_churn]
    if risk_category and risk_category != "All":
        filtered = filtered[filtered["risk_category"] == risk_category]
    if contract and contract != "All":
        filtered = filtered[filtered["Contract"] == contract]

    export_cols = [
        "customerID",
        "predicted_churn",
        "churn_probability",
        "risk_category",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod",
        "InternetService",
        "Churn",
    ]
    available_cols = [c for c in export_cols if c in filtered.columns]
    return filtered[available_cols].to_csv(index=False)


def get_customer_profile(customer_id: str, high_thresh: float = 0.60, med_thresh: float = 0.35) -> dict[str, Any] | None:
    """Retrieve complete 360 customer profile with empirical risk drivers and retention history."""
    df = get_scored_dataset(high_thresh, med_thresh)
    match = df[df["customerID"] == customer_id]
    if match.empty:
        return None

    row = match.iloc[0]
    drivers = extract_risk_drivers(row)
    actions = get_recommended_actions(row, row["risk_category"])
    tasks = list_tasks(customer_id=customer_id)
    notes = list_notes(customer_id=customer_id)

    return {
        "customer_id": customer_id,
        "model_output": {
            "predicted_churn": row["predicted_churn"],
            "churn_probability": float(row["churn_probability"]),
            "churn_probability_pct": float(row["churn_probability_pct"]),
            "risk_category": row["risk_category"],
            "historical_actual_churn": row.get("Churn", None),
        },
        "account_details": {
            "tenure": int(row["tenure"]),
            "contract": row["Contract"],
            "paperless_billing": row["PaperlessBilling"],
            "payment_method": row["PaymentMethod"],
            "monthly_charges": float(row["MonthlyCharges"]),
            "total_charges": float(row["TotalCharges"]) if not pd.isna(row["TotalCharges"]) else None,
        },
        "demographics": {
            "gender": row["gender"],
            "senior_citizen": "Yes" if row["SeniorCitizen"] == 1 else "No",
            "partner": row["Partner"],
            "dependents": row["Dependents"],
        },
        "services": {
            "phone_service": row["PhoneService"],
            "multiple_lines": row["MultipleLines"],
            "internet_service": row["InternetService"],
            "online_security": row["OnlineSecurity"],
            "online_backup": row["OnlineBackup"],
            "device_protection": row["DeviceProtection"],
            "tech_support": row["TechSupport"],
            "streaming_tv": row["StreamingTV"],
            "streaming_movies": row["StreamingMovies"],
        },
        "risk_drivers": drivers["risk_factors"],
        "protective_factors": drivers["protective_factors"],
        "recommended_actions": actions,
        "retention_history": {
            "tasks": tasks,
            "notes": notes,
        },
    }


def get_analytics_data() -> dict[str, Any]:
    """Compute deep-dive analytics aggregations for churn patterns."""
    df = get_scored_dataset()

    def _calc_group_stats(col_name: str) -> list[dict[str, Any]]:
        results = []
        for val, group in df.groupby(col_name):
            total = len(group)
            act_churn = int((group["Churn"].astype(str).str.lower() == "yes").sum())
            pred_churn = int((group["predicted_churn"] == "Churn").sum())
            results.append({
                "segment": str(val),
                "total": total,
                "historical_churn": act_churn,
                "historical_churn_rate": round((act_churn / total) * 100, 1),
                "predicted_churn": pred_churn,
                "predicted_churn_rate": round((pred_churn / total) * 100, 1),
            })
        return results

    contract_stats = _calc_group_stats("Contract")
    tenure_stats = _calc_group_stats("tenure_band")
    payment_stats = _calc_group_stats("PaymentMethod")
    internet_stats = _calc_group_stats("InternetService")
    tech_support_stats = _calc_group_stats("TechSupport")
    online_sec_stats = _calc_group_stats("OnlineSecurity")
    paperless_stats = _calc_group_stats("PaperlessBilling")

    # Financial comparison between churners vs non-churners
    churners = df[df["Churn"].astype(str).str.lower() == "yes"]
    retainers = df[df["Churn"].astype(str).str.lower() == "no"]

    financial_comparison = {
        "churners": {
            "count": len(churners),
            "avg_monthly_charges": round(float(churners["MonthlyCharges"].mean()), 2),
            "median_monthly_charges": round(float(churners["MonthlyCharges"].median()), 2),
            "avg_tenure": round(float(churners["tenure"].mean()), 1),
            "median_tenure": round(float(churners["tenure"].median()), 1),
            "avg_total_charges": round(float(churners["TotalCharges"].dropna().mean()), 2),
        },
        "retainers": {
            "count": len(retainers),
            "avg_monthly_charges": round(float(retainers["MonthlyCharges"].mean()), 2),
            "median_monthly_charges": round(float(retainers["MonthlyCharges"].median()), 2),
            "avg_tenure": round(float(retainers["tenure"].mean()), 1),
            "median_tenure": round(float(retainers["tenure"].median()), 1),
            "avg_total_charges": round(float(retainers["TotalCharges"].dropna().mean()), 2),
        },
    }

    return {
        "by_contract": contract_stats,
        "by_tenure": tenure_stats,
        "by_payment": payment_stats,
        "by_internet": internet_stats,
        "by_tech_support": tech_support_stats,
        "by_online_security": online_sec_stats,
        "by_paperless": paperless_stats,
        "financial_comparison": financial_comparison,
    }


def get_model_performance_data() -> dict[str, Any]:
    """Load complete technical model evaluation benchmarks."""
    metrics_info = get_metrics_data()
    
    # Load detailed holdout curve and confusion matrix data if present
    eval_details = {}
    if EVAL_DETAILS_PATH.exists():
        with open(EVAL_DETAILS_PATH, "r", encoding="utf-8") as f:
            eval_details = json.load(f)

    return {
        "summary": metrics_info.get("metrics_summary", {}),
        "best_model": metrics_info.get("best_model", "Logistic Regression"),
        "selection_rationale": metrics_info.get("selection_rationale", ""),
        "feature_importances": metrics_info.get("random_forest_features", []),
        "detailed_evaluations": eval_details,
        "holdout_test_size": 1409,
        "methodology": {
            "validation_scheme": "5-Fold Stratified Cross-Validation on Training Split (80%)",
            "benchmark_dataset": "Untouched Stratified Holdout Test Set (20% = 1,409 rows)",
            "leakage_prevention": "ColumnTransformer fitted strictly on training data",
            "calibration_note": "Probabilities are logistic posteriors; risk categories represent operational workflow thresholds.",
        },
    }
