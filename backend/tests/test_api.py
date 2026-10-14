"""
backend/tests/test_api.py
-------------------------
Integration tests for StayWise FastAPI REST API endpoints.
Uses FastAPI TestClient to test all routes without needing a live network port.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "StayWise"
    assert "health_check" in data


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data


def test_data_summary():
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] > 0
    assert 0.0 <= data["churn_rate"] <= 1.0
    assert len(data["contract_breakdown"]) > 0
    assert len(data["tenure_breakdown"]) > 0


def test_model_comparison():
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3  # LR, RF, XGB, DT


def test_active_model():
    response = client.get("/api/models/active")
    assert response.status_code == 200
    data = response.json()
    assert "best_model_name" in data
    assert "threshold" in data


def test_predict_single_valid():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
    }
    response = client.post("/api/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert "customer_lifetime_value" in data
    assert "expected_savings" in data


def test_predict_single_invalid_missing_field():
    payload = {
        "gender": "Female",
        # missing tenure, MonthlyCharges, etc.
    }
    response = client.post("/api/predict/single", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)


def test_explain_customer():
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }
    response = client.post("/api/explain/customer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "drivers" in data
    assert "recommended_actions" in data
    assert isinstance(data["drivers"], list)
