"""StayWise Enterprise Application — FastAPI Backend Server.

Provides robust, production-grade REST APIs for customer churn intelligence,
retention task workflows, customer 360 profiles, exploratory analytics,
and real Scikit-learn model performance evaluation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.database import (
    add_note,
    create_task,
    delete_task,
    get_task,
    init_db,
    list_notes,
    list_tasks,
    update_task,
)
from backend.schemas import (
    FilterParams,
    RetentionNoteCreate,
    RetentionTaskCreate,
    RetentionTaskUpdate,
    ThresholdConfig,
)
from backend.services import (
    export_customers_csv,
    get_analytics_data,
    get_customer_profile,
    get_model_performance_data,
    get_overview_data,
    get_scored_dataset,
    query_customers,
)

# App state for active operational thresholds
threshold_state = {
    "high_threshold": 0.60,
    "medium_threshold": 0.35,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database and pre-warm dataset scoring cache
    print("[StayWise Backend] Initializing SQLite database...")
    init_db()
    print("[StayWise Backend] Pre-scoring dataset with trained Scikit-learn pipeline...")
    get_scored_dataset(threshold_state["high_threshold"], threshold_state["medium_threshold"])
    print("[StayWise Backend] Startup complete. Engine is ready.")
    yield


app = FastAPI(
    title="StayWise Retention Intelligence API",
    description="Enterprise API powering customer churn prediction, 360 drilldown, and retention workflows.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Vite dev server and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# EXECUTIVE OVERVIEW API
# ==============================================================================
@app.get("/api/overview")
def get_overview():
    """Retrieve executive KPIs, risk distributions, and system health status."""
    try:
        return get_overview_data(
            high_thresh=threshold_state["high_threshold"],
            med_thresh=threshold_state["medium_threshold"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate overview: {exc}")


# ==============================================================================
# CUSTOMER WORKSPACE APIS
# ==============================================================================
@app.get("/api/customers")
def get_customers(
    search: str | None = Query(None, description="Search by customer ID"),
    predicted_churn: str | None = Query(None, description="Filter by 'Churn' or 'Stay'"),
    risk_category: str | None = Query(None, description="Filter by 'High Risk', 'Medium Risk', or 'Low Risk'"),
    contract: str | None = Query(None, description="Filter by contract type"),
    sort_by: str = Query("churn_probability", description="Column to sort by"),
    sort_order: str = Query("desc", description="'asc' or 'desc'"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=5, le=100, description="Items per page"),
):
    """Retrieve paginated, filtered, and sorted customer records."""
    try:
        return query_customers(
            search=search,
            predicted_churn=predicted_churn,
            risk_category=risk_category,
            contract=contract,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
            high_thresh=threshold_state["high_threshold"],
            med_thresh=threshold_state["medium_threshold"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query customers: {exc}")


@app.get("/api/customers/export")
def export_customers(
    search: str | None = Query(None),
    predicted_churn: str | None = Query(None),
    risk_category: str | None = Query(None),
    contract: str | None = Query(None),
):
    """Export the filtered customer dataset as downloadable CSV."""
    csv_content = export_customers_csv(
        search=search,
        predicted_churn=predicted_churn,
        risk_category=risk_category,
        contract=contract,
        high_thresh=threshold_state["high_threshold"],
        med_thresh=threshold_state["medium_threshold"],
    )
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=staywise_customers_export.csv"},
    )


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: str):
    """Retrieve complete 360 profile for a single customer."""
    profile = get_customer_profile(
        customer_id=customer_id,
        high_thresh=threshold_state["high_threshold"],
        med_thresh=threshold_state["medium_threshold"],
    )
    if not profile:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")
    return profile


# ==============================================================================
# AT-RISK ACCOUNTS QUEUE
# ==============================================================================
@app.get("/api/at-risk")
def get_at_risk(
    min_risk: str = Query("Medium Risk", description="'High Risk' only or 'Medium Risk' and above"),
    contract: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100),
):
    """Retrieve prioritized queue of at-risk customers for triage."""
    risk_filter = "High Risk" if min_risk == "High Risk" else None
    res = query_customers(
        risk_category=risk_filter,
        contract=contract,
        sort_by="churn_probability",
        sort_order="desc",
        page=page,
        page_size=page_size,
        high_thresh=threshold_state["high_threshold"],
        med_thresh=threshold_state["medium_threshold"],
    )
    if min_risk != "High Risk":
        # Keep High Risk and Medium Risk only
        res["items"] = [item for item in res["items"] if item["risk_category"] in ("High Risk", "Medium Risk")]
    return res


# ==============================================================================
# RETENTION ACTION WORKFLOW APIS
# ==============================================================================
@app.get("/api/retention/tasks")
def get_retention_tasks(
    status: str | None = Query(None),
    priority: str | None = Query(None),
    customer_id: str | None = Query(None),
):
    """List persistent retention tasks with optional status and priority filters."""
    return list_tasks(status=status, priority=priority, customer_id=customer_id)


@app.post("/api/retention/tasks", status_code=201)
def post_retention_task(task_in: RetentionTaskCreate):
    """Create a new retention follow-up task."""
    created = create_task(
        customer_id=task_in.customer_id,
        task_type=task_in.task_type,
        priority=task_in.priority,
        assigned_to=task_in.assigned_to,
        notes=task_in.notes,
        due_date=task_in.due_date,
    )
    return created


@app.patch("/api/retention/tasks/{task_id}")
def patch_retention_task(task_id: int, update_in: RetentionTaskUpdate):
    """Update task status, priority, notes, or assigned team member."""
    updated = update_task(
        task_id=task_id,
        status=update_in.status,
        priority=update_in.priority,
        assigned_to=update_in.assigned_to,
        notes=update_in.notes,
        due_date=update_in.due_date,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
    return updated


@app.delete("/api/retention/tasks/{task_id}", status_code=204)
def delete_retention_task(task_id: int):
    """Delete a retention task."""
    success = delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
    return Response(status_code=204)


@app.get("/api/retention/notes/{customer_id}")
def get_customer_notes(customer_id: str):
    """Get interaction notes for a customer."""
    return list_notes(customer_id)


@app.post("/api/retention/notes", status_code=201)
def post_customer_note(note_in: RetentionNoteCreate):
    """Log an interaction note for a customer."""
    return add_note(
        customer_id=note_in.customer_id,
        author=note_in.author,
        note=note_in.note,
    )


# ==============================================================================
# ANALYTICS & INSIGHTS API
# ==============================================================================
@app.get("/api/analytics")
def get_analytics():
    """Retrieve exploratory analytics distributions and financial comparisons."""
    try:
        return get_analytics_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics: {exc}")


# ==============================================================================
# MODEL PERFORMANCE API
# ==============================================================================
@app.get("/api/model-performance")
def get_model_performance():
    """Retrieve holdout evaluation benchmarks, ROC curves, and confusion matrices."""
    try:
        return get_model_performance_data()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model metrics: {exc}")


# ==============================================================================
# SYSTEM SETTINGS & THRESHOLD CONFIGURATION
# ==============================================================================
@app.get("/api/settings/thresholds")
def get_thresholds():
    """Get active operational triage thresholds."""
    return threshold_state


@app.post("/api/settings/thresholds")
def set_thresholds(cfg: ThresholdConfig):
    """Update active operational triage thresholds."""
    if cfg.medium_threshold >= cfg.high_threshold:
        raise HTTPException(status_code=400, detail="Medium threshold must be lower than High threshold.")
    threshold_state["high_threshold"] = cfg.high_threshold
    threshold_state["medium_threshold"] = cfg.medium_threshold
    # Refresh cache
    get_scored_dataset(cfg.high_threshold, cfg.medium_threshold, force_refresh=True)
    return threshold_state


# ==============================================================================
# STATIC FRONTEND SERVING
# ==============================================================================
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
