"""Pydantic schemas for the StayWise Enterprise API."""

from typing import Any
from pydantic import BaseModel, Field


class RetentionTaskCreate(BaseModel):
    customer_id: str
    task_type: str
    priority: str = "Medium"
    assigned_to: str = "Sarah Chen (CSM)"
    notes: str = ""
    due_date: str = ""


class RetentionTaskUpdate(BaseModel):
    status: str | None = None
    priority: str | None = None
    assigned_to: str | None = None
    notes: str | None = None
    due_date: str | None = None


class RetentionNoteCreate(BaseModel):
    customer_id: str
    author: str = "Current User"
    note: str


class ThresholdConfig(BaseModel):
    high_threshold: float = Field(0.60, ge=0.50, le=0.95)
    medium_threshold: float = Field(0.35, ge=0.15, le=0.59)


class FilterParams(BaseModel):
    search: str | None = None
    predicted_churn: str | None = None
    risk_category: str | None = None
    contract: str | None = None
    payment_method: str | None = None
    internet_service: str | None = None
    sort_by: str = "churn_probability"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 25
