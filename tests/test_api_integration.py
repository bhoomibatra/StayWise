"""Integration test verifying all StayWise enterprise API endpoints and static assets."""

import json
import urllib.request
import pytest

BASE_URL = "http://127.0.0.1:8000"


def test_overview_endpoint():
    res = urllib.request.urlopen(f"{BASE_URL}/api/overview")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["kpis"]["total_customers"] == 7043
    assert data["kpis"]["high_risk_count"] > 0
    assert "Logistic Regression" in data["system_status"]["model_name"]


def test_customers_search_and_pagination():
    res = urllib.request.urlopen(f"{BASE_URL}/api/customers?search=9237-HQITU")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert len(data["items"]) == 1
    assert data["items"][0]["customer_id"] == "9237-HQITU"
    assert data["items"][0]["risk_category"] == "High Risk"


def test_customer_360_profile():
    res = urllib.request.urlopen(f"{BASE_URL}/api/customers/9237-HQITU")
    assert res.status == 200
    profile = json.loads(res.read().decode("utf-8"))
    assert profile["customer_id"] == "9237-HQITU"
    assert len(profile["risk_drivers"]) > 0
    assert len(profile["recommended_actions"]) > 0


def test_at_risk_queue():
    res = urllib.request.urlopen(f"{BASE_URL}/api/at-risk?min_risk=High+Risk")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["pagination"]["total"] > 0
    for item in data["items"]:
        assert item["risk_category"] == "High Risk"


def test_retention_task_crud():
    payload = json.dumps({
        "customer_id": "9237-HQITU",
        "task_type": "Contract Upgrade Offer",
        "priority": "High",
        "assigned_to": "Sarah Chen (CSM)",
        "notes": "Automated integration test task",
        "due_date": "2026-10-31"
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/retention/tasks",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    res = urllib.request.urlopen(req)
    assert res.status == 201
    created = json.loads(res.read().decode("utf-8"))
    task_id = created["id"]
    assert created["customer_id"] == "9237-HQITU"
    assert created["status"] == "Open"

    # Patch status to In Progress
    patch_payload = json.dumps({"status": "In Progress"}).encode("utf-8")
    patch_req = urllib.request.Request(
        f"{BASE_URL}/api/retention/tasks/{task_id}",
        data=patch_payload,
        headers={"Content-Type": "application/json"},
        method="PATCH"
    )
    patch_res = urllib.request.urlopen(patch_req)
    assert patch_res.status == 200
    updated = json.loads(patch_res.read().decode("utf-8"))
    assert updated["status"] == "In Progress"


def test_analytics_endpoint():
    res = urllib.request.urlopen(f"{BASE_URL}/api/analytics")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert len(data["by_contract"]) == 3
    assert data["financial_comparison"]["churners"]["count"] == 1869


def test_model_performance_endpoint():
    res = urllib.request.urlopen(f"{BASE_URL}/api/model-performance")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["holdout_test_size"] == 1409
    assert "Logistic Regression" in data["summary"]
    assert "Random Forest" in data["summary"]
    assert "K-Nearest Neighbors" in data["summary"]
    assert "Support Vector Machine" in data["summary"]


def test_static_frontend_served():
    res = urllib.request.urlopen(BASE_URL)
    assert res.status == 200
    html = res.read().decode("utf-8")
    assert "StayWise" in html
    assert "root" in html
