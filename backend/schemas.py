"""
backend/schemas.py
------------------
Pydantic v2 schemas for StayWise FastAPI service.
Enforces strict typing, input validation, and clear JSON API serialization.
"""

from __future__ import annotations
from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    """Raw customer attributes received for single-record prediction."""
    model_config = ConfigDict(extra="ignore")

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Whether customer is a senior citizen (1) or not (0)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="Whether customer has multiple lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Customer's internet service provider")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Amount charged to customer monthly")
    TotalCharges: float = Field(..., ge=0.0, description="Total amount charged to customer")


class PredictionResponse(BaseModel):
    """Inference result returned for a single customer."""
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., description="Binary prediction: 1 for churn, 0 for retain")
    risk_tier: str = Field(..., description="Low, Medium, High, or Critical")
    decision_threshold: float = Field(..., description="Active classification cutoff")
    customer_lifetime_value: float = Field(..., description="Estimated CLV in USD")
    expected_savings: float = Field(..., description="Net expected savings from retention offer in USD")
    recommend_action: bool = Field(..., description="Whether retention intervention is cost-effective")


class RetentionAction(BaseModel):
    """Prescribed retention countermeasure based on SHAP feature attributions."""
    offer: str
    rationale: str
    cost_note: str


class SHAPDriver(BaseModel):
    """Individual feature contribution toward churn probability."""
    feature: str
    feature_value: Any
    shap_value: float
    impact: str = Field(..., description="Increases Churn Risk or Promotes Retention")
    reason: str


class ExplainResponse(BaseModel):
    """SHAP attributions and prescribed action plan for a customer."""
    churn_probability: float
    base_value: Optional[float] = None
    drivers: List[SHAPDriver]
    recommended_actions: List[RetentionAction]


class DataSummaryResponse(BaseModel):
    """Executive dataset metrics and historical churn breakdown."""
    total_customers: int
    churn_rate: float
    monthly_revenue_at_risk: float
    average_monthly_bill: float
    contract_breakdown: List[Dict[str, Any]]
    tenure_breakdown: List[Dict[str, Any]]
    payment_breakdown: List[Dict[str, Any]]
    class_distribution: Dict[str, int]


class ModelComparisonItem(BaseModel):
    """Benchmark metrics for a single algorithm."""
    model: str
    train_time_s: float
    cv_roc_auc: float
    cv_pr_auc: float
    cv_f1: float
    cv_precision: float
    cv_recall: float
    val_pr_auc: float
    val_roc_auc: float
    val_f1: float


class ActiveModelResponse(BaseModel):
    """Metadata regarding currently active, calibrated production pipeline."""
    best_model_name: str
    threshold: float
    calibration_method: str
    test_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
