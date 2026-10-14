"""
backend/main.py
---------------
Main FastAPI application entry point for StayWise.
Configures CORS, global lifespan preloading, and REST routers.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router as api_router, _get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("staywise.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload ML model on startup to ensure instant zero-cold-start inference."""
    logger.info("Initializing StayWise backend service ...")
    try:
        _get_model()
        logger.info("ML model pipeline preloaded successfully.")
    except Exception as e:
        logger.warning("Failed to preload ML model on startup: %s", e)
    yield
    logger.info("Shutting down StayWise backend.")


app = FastAPI(
    title="StayWise — Churn Intelligence API",
    description="Explainable Customer Churn Prediction and Retention Decision Support Service",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Configuration ───────────────────────────────────────────────────────
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Routes ───────────────────────────────────────────────────────────
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "project": "StayWise",
        "description": "Customer Churn Prediction & Retention Platform",
        "docs_url": "/docs",
        "health_check": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
