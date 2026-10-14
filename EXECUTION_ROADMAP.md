# StayWise — Line-by-Line Execution Roadmap & Tracking Ledger
**Project:** StayWise (Full-Stack Churn Intelligence Platform)  
**Execution Type:** Strict Line-by-Line Phased Execution  
**Document Version:** 1.0.0  
**Current Status:** IN EXECUTION (Phases 1-3 Verified, Backend Underway)  

---

## 🗺️ Master Execution Flowchart

```
[ Phase 1: Environment & Tooling Audit ]
  ├── 1.1 Python 3.11 Runtime Check ────────────── [DONE]
  ├── 1.2 ML Dependencies Verification ─────────── [DONE]
  ├── 1.3 Dataset Verification (7,043 rows) ────── [DONE]
  └── 1.4 Node.js / Tooling Discovery ─────────── [IN PROGRESS]
           │
[ Phase 2: Core ML Engine & Unit Tests ]
  ├── 2.1 Zero-Leakage Pipeline Fitting ───────── [DONE]
  ├── 2.2 Isotonic Probability Calibration ────── [DONE]
  ├── 2.3 Financial Utility Thresholding ───────── [DONE]
  └── 2.4 Pytest Suite Verification (85/85) ────── [DONE]
           │
[ Phase 3: High-Performance FastAPI Backend ]
  ├── 3.1 Install FastAPI & Pydantic v2 ───────── [DONE]
  ├── 3.2 Define Pydantic Input/Output Schemas ── [TASK-BE-01]
  ├── 3.3 Implement Modular REST Routes ───────── [TASK-BE-02]
  ├── 3.4 Build Application Server & CORS ─────── [TASK-BE-03]
  └── 3.5 Automated Backend API Pytests ───────── [TASK-BE-04]
           │
[ Phase 4: Modern React + Vite Frontend Engine ]
  ├── 4.1 Scaffold React 18 + Vite + TS App ───── [TASK-FE-01]
  ├── 4.2 Configure Tailwind CSS & Design Tokens ─ [TASK-FE-02]
  ├── 4.3 Build Reusable UI Glass Components ───── [TASK-FE-03]
  │     ├── GlassCard & MetricHero
  │     ├── AnimatedRiskGauge
  │     ├── ArchetypePresetSelector
  │     └── SHAPWaterfallViewer
  └── 4.4 Build 5 Interactive App Views ────────── [TASK-FE-04]
        ├── View 1: Executive Overview & Cohorts
        ├── View 2: Customer Diagnosis & SHAP
        ├── View 3: What-If Counterfactual Simulator
        ├── View 4: Bulk CSV Uploader & Risk Grid
        └── View 5: Model Lab & Benchmark Matrix
           │
[ Phase 5: Full-Stack Integration & QA ]
  ├── 5.1 Connect Frontend API Client to FastAPI ─ [TASK-INT-01]
  ├── 5.2 End-to-End Single & Batch Prediction ── [TASK-INT-02]
  ├── 5.3 Error Boundary & Empty State Validation ─ [TASK-INT-03]
  └── 5.4 Full Test Suite Execution ───────────── [TASK-INT-04]
           │
[ Phase 6: Final Deliverables & Git Push ]
  ├── 6.1 Update README with Windows CLI Commands [TASK-DOC-01]
  ├── 6.2 Synchronize IMPLEMENTATION_PLAN.md ──── [TASK-DOC-02]
  └── 6.3 Milestone Commit & Push to GitHub ───── [TASK-GIT-01]
```

---

## 📋 Granular Line-by-Line Task Checklist

### Phase 1: Environment & Tooling Audit
* [x] **LINE-01**: Check Python version in `C:\StayWise_env` (Confirmed: Python 3.11.9 64-bit).
* [x] **LINE-02**: Audit installed packages: `scikit-learn 1.9.1`, `xgboost 2.1.4`, `shap 0.51.0`, `pandas 2.3.3`, `numpy 2.4.6`.
* [x] **LINE-03**: Verify IBM Telco dataset at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns).
* [x] **LINE-04**: Install backend dependencies: `fastapi` (v0.143.1) and `pydantic` (v2.14.1).
* [ ] **LINE-05**: Verify or provision Node.js LTS runtime for frontend compilation.

### Phase 2: Core Machine Learning Engine
* [x] **LINE-06**: Verify `src/data.py` data cleaning, schema validation, and 3-way stratified splitting.
* [x] **LINE-07**: Verify `src/features.py` feature engineering (`tenure_group`, `charges_per_month`, `num_addons`).
* [x] **LINE-08**: Verify `src/train.py` model training, cross-validation, and isotonic calibration.
* [x] **LINE-09**: Verify `src/evaluate.py` PR-AUC, ROC-AUC, Brier score, and confusion matrix calculation.
* [x] **LINE-10**: Verify `src/roi.py` customer lifetime value and financial profit utility curves.
* [x] **LINE-11**: Verify `src/actions.py` rule-based prescriptive retention countermeasures.
* [x] **LINE-12**: Run 85 pytest tests across all ML modules (`85 passed in 4.78s`).

### Phase 3: FastAPI Backend Implementation
* [x] **LINE-13**: Create `backend/requirements.txt` with locked dependencies.
* [x] **LINE-14**: Create `backend/schemas.py` containing Pydantic models for single prediction, batch prediction, metadata, and SHAP.
* [x] **LINE-15**: Create `backend/routes.py` with endpoints:
  - `GET /api/health`
  - `GET /api/data/summary`
  - `GET /api/models/comparison`
  - `GET /api/models/active`
  - `POST /api/predict/single`
  - `POST /api/predict/batch`
  - `POST /api/explain/customer`
* [x] **LINE-16**: Create `backend/main.py` configuring CORS middleware, static asset routing, and lifespan model loading.
* [x] **LINE-17**: Create `backend/tests/test_api.py` testing every endpoint with valid/invalid payloads using FastAPI `TestClient`.
* [x] **LINE-18**: Execute and verify backend test suite via `pytest backend/tests/` (8 / 8 tests passed in 3.30s).

### Phase 4: React + TypeScript Frontend
* [ ] **LINE-19**: Initialize Vite project in `frontend/` (`npm create vite@latest frontend -- --template react-ts`).
* [ ] **LINE-20**: Install UI dependencies: `tailwindcss`, `lucide-react`, `recharts`, `clsx`, `tailwind-merge`.
* [ ] **LINE-21**: Configure `tailwind.config.js` with Cyber-SaaS dark theme tokens (`#080C14`, `#00F0FF`, `#8B5CF6`).
* [ ] **LINE-22**: Build core UI components:
  - `Navbar.tsx` (Status pill, brand badge, module tabs)
  - `GlassCard.tsx` (Glassmorphic containers with glowing hover states)
  - `RiskGauge.tsx` (Dynamic circular/radial risk meter)
  - `PresetSelector.tsx` (1-Click Customer Archetypes)
  - `ShapWaterfall.tsx` (Color-coded feature attribution bars)
* [ ] **LINE-23**: Build page views:
  - `OverviewPage.tsx` (Executive KPI cards, Cohort charts)
  - `CustomerLookupPage.tsx` (Interactive customer scoring & ROI countermeasure card)
  - `WhatIfSimulatorPage.tsx` (Before vs. After scenario testing)
  - `TopAtRiskPage.tsx` (CSV drag-and-drop file upload & exportable data grid)
  - `ModelLabPage.tsx` (Interactive model comparisons and ROC/PR curves)

### Phase 5: End-to-End Integration & Verification
* [ ] **LINE-24**: Connect frontend services to backend REST endpoints (`http://127.0.0.1:8000/api/`).
* [ ] **LINE-25**: Verify single prediction inference roundtrip (<50ms).
* [ ] **LINE-26**: Verify CSV batch upload and downloadable scored predictions.
* [ ] **LINE-27**: Verify handling of invalid inputs and edge-case errors.
* [ ] **LINE-28**: Run production build: `npm run build` with zero TypeScript or styling warnings.

### Phase 6: Documentation & Git Versioning
* [ ] **LINE-29**: Update `README.md` with complete Windows CLI instructions for running backend and frontend.
* [ ] **LINE-30**: Update `IMPLEMENTATION_PLAN.md` with final measured performance.
* [ ] **LINE-31**: Stage, commit, and push milestone to GitHub repository `bhoomibatra/StayWise`.

---

## 📊 Live Verification Ledger

| Line Ref | Action Item | Target File / Module | Output / Verification | Status |
|---|---|---|---|---|
| LINE-01 | Python Audit | Environment | Python 3.11.9 (64-bit) | ✅ **PASS** |
| LINE-02 | ML Stack | `src/` | Scikit-learn 1.9.1, XGBoost 2.1.4, SHAP 0.51.0 | ✅ **PASS** |
| LINE-03 | Dataset Check | `data/raw/` | 7,043 rows, 21 columns, 26.54% churn | ✅ **PASS** |
| LINE-04 | FastAPI Install | `C:\StayWise_env` | FastAPI 0.143.1, Pydantic 2.14.1 | ✅ **PASS** |
| LINE-12 | Pytest Run | `tests/` | 85 / 85 tests passing in 4.78s | ✅ **PASS** |
| LINE-14 | Pydantic Schemas | `backend/schemas.py` | Schema validation | ⏳ **PENDING** |
| LINE-15 | REST Routes | `backend/routes.py` | 7 REST endpoints | ⏳ **PENDING** |
| LINE-16 | App Server | `backend/main.py` | Uvicorn ASGI server | ⏳ **PENDING** |
| LINE-18 | API Tests | `backend/tests/` | Endpoint test coverage | ⏳ **PENDING** |
| LINE-19 | Frontend Init | `frontend/` | React 18 + TypeScript + Vite | ⏳ **PENDING** |
| LINE-28 | Frontend Build | `frontend/dist/` | Production bundle compilation | ⏳ **PENDING** |
| LINE-31 | Git Milestone | GitHub Remote | Main branch synchronized | ⏳ **PENDING** |
