# StayWise — Master Full-Stack Architecture & Implementation Contract
**Full Title:** Explainable Customer Churn Prediction & Retention Intelligence Platform  
**System Architecture:** Decoupled Full-Stack Application (React + TypeScript + Tailwind CSS Frontend | FastAPI + Pydantic Backend | Scikit-Learn + XGBoost ML Engine)  
**Author:** Senior ML & Full-Stack Architect  
**Status:** DRAFT FOR APPROVAL (Phase 1 Audit & Phase 2 Blueprint Complete)  
**Date:** 2026-10-15  

---

## 1. Phase 1: Workspace & Environment Audit Report

Prior to modifying or architecting any components, an audit of `c:\Users\hp\OneDrive\Documents\StayWise` was executed:

### 1.1 Development Environment & Tooling
* **Operating System:** Windows 10/11 (64-bit PowerShell)
* **Python Runtime:** Python 3.11.9 in `C:\StayWise_env\Scripts\python.exe`
* **Python ML Stack Installed:** `pandas 2.3.3`, `numpy 2.4.6`, `scikit-learn 1.9.1`, `xgboost 2.1.4`, `shap 0.51.0`, `joblib 1.6.0`, `pytest 8.4.2`, `uvicorn 0.54.0`.
* **Missing Python Packages for Backend:** `fastapi`, `python-multipart` (easily installed via `pip install fastapi python-multipart`).
* **Node.js / npm Status:** **Not installed** on current system PATH (`winget` is available: `v1.29.380` for clean automated installation of Node.js LTS).
* **Dataset Status:** Present and verified at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns, 26.54% churn rate).
* **Existing ML Codebase:** Modular modules exist in `src/` (`data.py`, `features.py`, `train.py`, `evaluate.py`, `predict.py`, `explain.py`, `roi.py`, `actions.py`, `survival.py`). All 85 unit and integration tests pass.
* **Preservation Policy:** All existing verified ML algorithms, zero-leakage preprocessing, SHAP explainers, and test suites will be preserved and wrapped seamlessly by the FastAPI service.

---

## 2. Section A: Project Specification

### 2.1 Problem Statement
Customer defection in telecommunications causes severe recurring revenue contraction. Generic churn systems provide uncalibrated probabilities with no explanation or actionable retention steps. StayWise bridges machine learning and business operations by delivering:
1. Real-time calibrated churn probability calculation.
2. Local and global game-theoretic explanations (SHAP attributions).
3. Value-based retention recommendations calculated through customer lifetime value (CLV) and net offer ROI.

### 2.2 Target Users
* **Retention Marketing Managers:** Prioritize outbound contact campaigns, evaluate cohort vulnerability, and download enriched CSV action lists.
* **Customer Success Representatives:** Instant 1-to-1 customer risk diagnosis during support calls with live countermeasure playbooks.
* **Executive Leadership & Data Scientists:** Model governance, algorithm benchmarking (Logistic Regression vs. Random Forest vs. XGBoost), and overall revenue at risk.

### 2.3 Scope & Acceptance Criteria
* **Mandatory MVP Scope:**
  - Independent **FastAPI** backend with typed Pydantic v2 schemas, structured logging, and CORS middleware.
  - Dedicated REST endpoints for data summary, model comparison, active model info, single inference, batch CSV scoring, and SHAP explainability.
  - Modern, responsive **React + TypeScript + Vite + Tailwind CSS** frontend dashboard.
  - 5 Interactive Views: (1) Executive Overview, (2) Single Customer Diagnosis & Presets, (3) Model Lab Benchmarking, (4) SHAP Explainability Dashboard, (5) Bulk Customer Risk Explorer.
  - 100% genuine model execution: zero hardcoded mock predictions or fabricated metrics.
  - Automated test coverage across ML pipelines, FastAPI endpoints, and frontend components.

---

## 3. Section B: System Architecture & Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                   StayWise Modern React Frontend                       │
│    (Vite + TypeScript + Tailwind CSS + Lucide Icons + Recharts)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / JSON (REST)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Application Backend                        │
│             (Uvicorn + Pydantic v2 + CORS Middleware)                  │
├────────────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                            │
│  • GET  /api/health            • POST /api/predict/single              │
│  • GET  /api/data/summary      • POST /api/predict/batch (CSV Upload)  │
│  • GET  /api/models/comparison • POST /api/explain/customer            │
│  • GET  /api/models/active     • GET  /api/explain/global              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python Object Calls
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      StayWise ML Pipeline Engine                       │
│   (Zero-Leakage ColumnTransformer -> Fitted Model -> Calibrator)       │
│                                                                        │
│   • Preprocessor: Median Imputer + StandardScaler + OneHotEncoder      │
│   • Estimator: Best Model (Logistic Regression / XGBoost / RF)         │
│   • Calibrator: CalibratedClassifierCV (Isotonic)                      │
│   • Explainability: TreeExplainer / LinearExplainer (SHAP)             │
│   • Business Logic: Financial Utility & Action Recommendation Engine   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Section C & D: Dataset & Machine Learning Pipeline

### 4.1 Dataset Properties
* **Source:** IBM Telco Customer Churn (7,043 rows, 21 columns)
* **Target:** `Churn` (encoded binary: 0 = No, 1 = Yes).
* **ID Column:** `customerID` (stripped from predictive features; preserved solely for batch tracking).
* **Cleaning:** `TotalCharges` coerced to numeric; blank spaces imputed via median learned strictly from the training split.

### 4.2 Machine Learning Architecture
* **Split:** Stratified 70% Train, 15% Validation, 15% Held-out Test.
* **Leakage Prevention:** Transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) fitted strictly on `X_train`.
* **Algorithms Evaluated:**
  1. **Logistic Regression (L2 regularized, balanced weights)**
  2. **Random Forest Classifier (300 estimators, balanced weights)**
  3. **XGBoost Classifier (300 estimators, scale_pos_weight)**
* **Model Selection:** Ranked by **Validation PR-AUC** (Average Precision) to prioritize positive class ranking under 26.5% class imbalance.
* **Calibration & Thresholding:** Isotonic calibration applied to align output probabilities; threshold selected by maximizing financial retention utility on validation data.

---

## 5. Section E & F: Model Evaluation & Explainable AI (SHAP)

### 5.1 Measured Benchmark Metrics (Current Fitted Models)
| Model | 5-Fold CV PR-AUC | Val PR-AUC | Val ROC-AUC | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|---|---|
| **Logistic Regression (Best)** | **0.6695** | **0.6470** | **0.8366** | 0.6224 | 0.5139 | 0.7888 |
| **XGBoost** | 0.6492 | 0.6306 | 0.8286 | 0.6216 | 0.5442 | 0.7246 |
| **Random Forest** | 0.6243 | 0.6038 | 0.8179 | 0.5912 | 0.5582 | 0.6283 |

* **Held-out Test Performance:** PR-AUC: `0.6261` | ROC-AUC: `0.8419` | Recall: `95.72%` | F1: `0.5568`.

### 5.2 Explainability Framework
* **Local Explanations:** SHAP attributions generated for each individual prediction, returned as ranked positive (churn-increasing) and negative (retention-increasing) drivers.
* **Rule-based Playbook:** Top positive SHAP drivers dynamically trigger targeted business actions (e.g., contract discount, security add-on trial, tech support concierge).
* **Disclaimers:** Model explanations clearly state correlational feature attribution, not causal proof.

---

## 6. Phase 4: UI/UX Design System Specification

### 6.1 Theme & Design Tokens
* **Design Aesthetic:** Modern Cyber-SaaS Glassmorphism with deep navy/slate backgrounds and neon cyan highlights.
* **Palette:**
  - Background Base: `#080C14` / `#0E1626`
  - Elevated Glass Surface: `rgba(18, 26, 43, 0.75)` with `1px solid rgba(0, 240, 255, 0.15)`
  - Accent Primary: `#00F0FF` (Cyan)
  - Accent Secondary: `#8B5CF6` (Violet)
  - Risk Low: `#10B981` (Emerald) | Medium: `#F59E0B` (Amber) | High/Critical: `#F43F5E` (Rose)
* **Typography:** `Plus Jakarta Sans` or `Inter`, high-contrast legible hierarchy.
* **Micro-interactions:** Glowing hover states, animated risk badges, skeleton loading states during API inference.

### 6.2 Application Views
1. **Executive Overview Dashboard:** KPI tiles (Total Customers, Churn %, Revenue at Risk, Avg Bill), interactive donut chart, churn by contract/tenure charts.
2. **Customer Prediction & Diagnosis:** 1-Click Customer Archetypes (High-Risk New User, Loyal Family, At-Risk Streamer), interactive input sliders, animated risk gauge, SHAP driver waterfall, and retention ROI cards.
3. **Model Laboratory:** Interactive model comparison table, ROC & PR curve visualizer, interactive confusion matrix with dynamic threshold slider.
4. **Explainability Hub:** Global feature importance ranking, beeswarm attributions, and plain-English factor breakdowns.
5. **Bulk Customer Risk Explorer:** CSV drag-and-drop zone, schema validation feedback, sortable risk data grid, filter controls, and enriched CSV download.

---

## 7. Phase 5: Backend REST API Specification

### 7.1 Endpoints
1. `GET /api/health`
   - Returns: `{"status": "healthy", "model_loaded": bool, "version": "1.0"}`
2. `GET /api/data/summary`
   - Returns: Total rows, historical churn rate, revenue at risk, class distribution, contract/tenure breakdowns.
3. `GET /api/models/comparison`
   - Returns: Comparison metrics for Logistic Regression, Random Forest, Decision Tree, and XGBoost.
4. `GET /api/models/active`
   - Returns: Active model metadata, calibration method, selected threshold, test set metrics.
5. `POST /api/predict/single`
   - Request: Single customer demographic and service features (JSON).
   - Response: `churn_probability`, `churn_prediction`, `risk_tier`, `clv`, `expected_saving`, `recommend_action`.
6. `POST /api/predict/batch`
   - Request: Multi-part form data with CSV file upload.
   - Response: JSON summary + downloadable scored records with risk tier and suggested retention offer.
7. `POST /api/explain/customer`
   - Request: Customer feature payload.
   - Response: Top positive/negative SHAP feature drivers + recommended actions.

---

## 8. Phase 6: Project Directory Structure

```
StayWise/
├── IMPLEMENTATION_PLAN.md             # Master Source of Truth
├── README.md                          # Production Documentation
├── .gitignore                         # Comprehensive ignore rules
├── data/
│   └── raw/                           # WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_pipeline.joblib           # Trained serialized pipeline
│   └── training_metadata.json         # Real measured metrics
├── reports/
│   └── model_comparison.csv          # Algorithm benchmark matrix
├── src/                               # Core Python ML Engine
│   ├── data.py                        # Ingestion, cleaning, splitting
│   ├── features.py                    # Feature engineering
│   ├── train.py                       # Cross-validation, training, calibration
│   ├── evaluate.py                    # PR-AUC, ROC-AUC, F1 metrics
│   ├── predict.py                     # Single & batch inference
│   ├── explain.py                     # SHAP explanations
│   ├── roi.py                         # Customer value & financial utility
│   └── actions.py                     # Rule-based retention interventions
├── backend/                           # FastAPI Application Service
│   ├── main.py                        # FastAPI entry point & CORS
│   ├── schemas.py                     # Pydantic v2 request/response models
│   ├── routes.py                      # REST route definitions
│   └── tests/                         # API endpoint pytest suite
├── frontend/                          # Modern React Application
│   ├── src/
│   │   ├── components/                # Glass cards, gauges, charts, badges
│   │   ├── pages/                     # Overview, Lookup, Lab, Explain, Batch
│   │   ├── services/                  # API client (Axios/Fetch)
│   │   ├── types/                     # TypeScript definitions
│   │   └── App.tsx                    # Shell & navigation
│   ├── package.json                   # React + Vite dependencies
│   ├── tailwind.config.js             # Cyber-SaaS design tokens
│   └── vite.config.ts                 # Build configuration & proxy
└── tests/                             # Full ML test suite (85 tests passing)
```

---

## 9. Phase 8: Strict Line-by-Line Execution Checklist

| Task ID | Phase / Component | Description | Prerequisites | Target Files | Verification Method | Status |
|---|---|---|---|---|---|---|
| **TASK-101** | Audit | Workspace & toolchain inspection | None | Environment | Node.js & Python check | **COMPLETE** |
| **TASK-102** | Blueprint | Master Full-Stack Plan creation | TASK-101 | `IMPLEMENTATION_PLAN.md` | Verification of sections A-F | **COMPLETE** |
| **TASK-201** | Backend Deps | Install FastAPI & Pydantic | TASK-101 | `backend/requirements.txt` | `import fastapi` verification | **NOT STARTED** |
| **TASK-202** | Backend API | Implement schemas & FastAPI app | TASK-201 | `backend/main.py`, `backend/routes.py` | Pytest backend test suite | **NOT STARTED** |
| **TASK-301** | Tooling | Setup Node.js runtime for React | TASK-101 | System / Node.js LTS | `node -v` & `npm -v` | **NOT STARTED** |
| **TASK-302** | Frontend Scaffold | Create React + Vite + Tailwind app | TASK-301 | `frontend/` | `npm run build` zero errors | **NOT STARTED** |
| **TASK-303** | UI Components | Build Glass Cards, Gauges & Nav | TASK-302 | `frontend/src/components/` | Visual rendering check | **NOT STARTED** |
| **TASK-304** | Views Assembly | Implement 5 dashboard views | TASK-303 | `frontend/src/pages/` | All pages active & connected | **NOT STARTED** |
| **TASK-401** | Integration | Connect React to FastAPI backend | TASK-202, 304 | `frontend/src/services/` | End-to-end inference verification | **NOT STARTED** |
| **TASK-501** | Verification | Automated tests (ML + API + UI) | All | `tests/`, `backend/tests/` | 100% tests passing | **NOT STARTED** |
| **TASK-601** | Git Push | Commit milestone & push to GitHub | All | Git remote | Clean git status on `main` | **NOT STARTED** |

---

## 10. Phase 9: Approval Gate

In accordance with strict execution protocol:
1. Phase 1 (Workspace inspection & environment constraints recorded) and Phase 2 (Creation of comprehensive blueprint) are complete.
2. The plan describes the full-stack architecture, exact data flow, Pydantic endpoints, UI design tokens, and ordered task checklist.
3. **Awaiting User Review and Approval** before proceeding to execution!
