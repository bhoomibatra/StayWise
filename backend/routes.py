"""
backend/routes.py
-----------------
API route handlers for StayWise FastAPI service.
Connects REST requests directly to the verified core Python ML pipeline in src/.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from src.data import clean, load_raw, get_quality_report
from src.features import engineer
from src.predict import (
    load_pipeline,
    predict_single,
    predict_batch,
    _apply_engineering,
    _drop_non_features,
)
from src.roi import customer_value, expected_saving
from src.explain import explain_customer
from src.actions import get_actions

from backend.schemas import (
    CustomerInput,
    PredictionResponse,
    ExplainResponse,
    SHAPDriver,
    RetentionAction,
    DataSummaryResponse,
    ModelComparisonItem,
    ActiveModelResponse,
)

router = APIRouter(prefix="/api", tags=["StayWise API"])

# Cache model in memory
_CACHED_PAYLOAD = None

def _get_model():
    global _CACHED_PAYLOAD
    if _CACHED_PAYLOAD is None:
        try:
            _CACHED_PAYLOAD = load_pipeline()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Trained ML model pipeline not found. Run model training first.",
            )
    return _CACHED_PAYLOAD


@router.get("/health")
def health_check():
    """Verify service health and model readiness."""
    try:
        payload = _get_model()
        model_name = payload["metadata"].get("best_model_name", "Unknown")
        return {"status": "healthy", "model_loaded": True, "active_model": model_name}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "error": str(e)}


@router.get("/data/summary", response_model=DataSummaryResponse)
def get_data_summary():
    """Retrieve executive dataset stats and historical cohort breakdowns."""
    try:
        raw = load_raw()
        df = clean(raw)
        df = engineer(df)
        qr = get_quality_report(df)

        n_churners = int(df["Churn"].sum())
        rev_at_risk = float(df[df["Churn"] == 1]["MonthlyCharges"].sum())

        # Contract Breakdown
        contract_grp = df.groupby("Contract")["Churn"].agg(["mean", "count"]).reset_index()
        contract_breakdown = [
            {"contract": row["Contract"], "churn_rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for _, row in contract_grp.iterrows()
        ]

        # Tenure Breakdown
        tenure_grp = df.groupby("tenure_group")["Churn"].agg(["mean", "count"]).reset_index()
        tenure_breakdown = [
            {"tenure_group": row["tenure_group"], "churn_rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for _, row in tenure_grp.iterrows()
        ]

        # Payment Breakdown
        pay_grp = df.groupby("PaymentMethod")["Churn"].agg(["mean", "count"]).reset_index()
        payment_breakdown = [
            {"payment_method": row["PaymentMethod"], "churn_rate": round(float(row["mean"]), 4), "count": int(row["count"])}
            for _, row in pay_grp.iterrows()
        ]

        return DataSummaryResponse(
            total_customers=qr["n_rows"],
            churn_rate=round(qr["churn_rate"], 4),
            monthly_revenue_at_risk=round(rev_at_risk, 2),
            average_monthly_bill=round(float(df["MonthlyCharges"].mean()), 2),
            contract_breakdown=contract_breakdown,
            tenure_breakdown=tenure_breakdown,
            payment_breakdown=payment_breakdown,
            class_distribution={"retained": int((df["Churn"] == 0).sum()), "churned": n_churners},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute data summary: {e}")


@router.get("/models/comparison", response_model=List[ModelComparisonItem])
def get_model_comparison():
    """Return benchmark metrics from the latest multi-algorithm training run."""
    csv_path = Path(__file__).resolve().parent.parent / "reports" / "model_comparison.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Model comparison report not found.")
    df_cmp = pd.read_csv(csv_path)
    items = []
    for _, row in df_cmp.iterrows():
        items.append(ModelComparisonItem(
            model=str(row["model"]),
            train_time_s=round(float(row["train_time_s"]), 2),
            cv_roc_auc=round(float(row["cv_roc_auc"]), 4),
            cv_pr_auc=round(float(row["cv_pr_auc"]), 4),
            cv_f1=round(float(row["cv_f1"]), 4),
            cv_precision=round(float(row["cv_precision"]), 4),
            cv_recall=round(float(row["cv_recall"]), 4),
            val_pr_auc=round(float(row["val_pr_auc"]), 4),
            val_roc_auc=round(float(row["val_roc_auc"]), 4),
            val_f1=round(float(row["val_f1"]), 4),
        ))
    return items


@router.get("/models/active", response_model=ActiveModelResponse)
def get_active_model():
    """Return metadata regarding the calibrated model currently deployed."""
    payload = _get_model()
    meta = payload["metadata"]
    return ActiveModelResponse(
        best_model_name=meta.get("best_model_name", "Logistic Regression"),
        threshold=meta.get("threshold", 0.06),
        calibration_method="isotonic",
        test_metrics=meta.get("test_metrics", {}),
        val_metrics=meta.get("val_metrics", {}),
    )


@router.post("/predict/single", response_model=PredictionResponse)
def predict_single_customer(customer: CustomerInput):
    """Generate real-time calibrated prediction and retention ROI for one customer."""
    payload = _get_model()
    threshold = payload["metadata"].get("threshold", 0.06)
    record = customer.model_dump()

    res = predict_single(record, payload=payload, threshold=threshold)
    prob = res["probability"]
    label = res["label"]
    risk = res["risk_category"]

    clv = customer_value(customer.MonthlyCharges, customer.tenure)
    roi = expected_saving(prob, clv)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=label,
        risk_tier=risk,
        decision_threshold=threshold,
        customer_lifetime_value=round(clv, 2),
        expected_savings=round(roi["expected_saving"], 2),
        recommend_action=roi["recommend_action"],
    )


@router.post("/explain/customer", response_model=ExplainResponse)
def explain_single_customer(customer: CustomerInput):
    """Compute local SHAP game-theoretic attributions and dynamic action offers."""
    payload = _get_model()
    record = customer.model_dump()
    threshold = payload["metadata"].get("threshold", 0.06)

    res = predict_single(record, payload=payload, threshold=threshold)
    prob = res["probability"]

    record_df = pd.DataFrame([record])
    record_df = _drop_non_features(record_df)
    record_df = _apply_engineering(record_df)

    raw_drivers = explain_customer(payload["pipeline"], record_df)
    drivers = [
        SHAPDriver(
            feature=d["feature"],
            feature_value=d.get("feature_value", ""),
            shap_value=round(float(d["shap_value"]), 4),
            impact="Increases Churn Risk" if d["shap_value"] > 0 else "Promotes Retention",
            reason=d["reason"],
        )
        for d in raw_drivers
    ]

    actions = [
        RetentionAction(
            offer=a["offer"],
            rationale=a["rationale"],
            cost_note=a["cost_note"],
        )
        for a in get_actions(raw_drivers)
    ]

    return ExplainResponse(
        churn_probability=round(prob, 4),
        drivers=drivers,
        recommended_actions=actions,
    )


@router.post("/predict/batch")
async def predict_batch_customers(file: UploadFile = File(...)):
    """Upload bulk CSV of customers, score all rows, and return high-risk summary."""
    payload = _get_model()
    try:
        contents = await file.read()
        import io
        df_upload = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {e}")

    try:
        scored = predict_batch(df_upload, payload=payload)
        threshold = payload["metadata"].get("threshold", 0.06)
        
        n_total = len(scored)
        n_churners = int((scored["churn_prediction"] == 1).sum())
        n_critical = int((scored["churn_probability"] >= 0.80).sum())
        avg_prob = float(scored["churn_probability"].mean())

        top_at_risk = scored.sort_values("churn_probability", ascending=False).head(50).to_dict(orient="records")

        return JSONResponse(content={
            "summary": {
                "total_customers": n_total,
                "predicted_churners": n_churners,
                "critical_risk_count": n_critical,
                "average_churn_probability": round(avg_prob, 4),
                "threshold_applied": threshold,
            },
            "top_records": top_at_risk,
        })
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction error: {e}")
