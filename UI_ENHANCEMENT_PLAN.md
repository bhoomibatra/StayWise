# StayWise — UI/UX Redesign & Workflow Execution Document
**Document:** UI & Workflow Modernization Master Plan  
**Target Application:** StayWise Churn Intelligence Platform  
**Design Standard:** Modern AI-SaaS Glassmorphism, Neon/Slate Dark Theme, High-Utility Workflows  
**Author:** Lead UI/UX Architect & Senior Software Engineer  

---

## 1. Executive Vision & Design Philosophy

The objective of this UI/UX overhaul is to elevate **StayWise** from a standard utility script to an **enterprise-grade, visually stunning AI decision-support platform**. The application will deliver an immediate "WOW" factor while dramatically improving usability and executive workflow speed.

### Core Design Principles:
1. **Sleek Cyber-SaaS Aesthetic:** Deep dark canvas (`#080C14` to `#0F172A`), multi-layered glassmorphism (`backdrop-filter: blur(16px)`), subtle neon cyan (`#00F0FF`) and purple/indigo accents.
2. **Instant Value Workflows (Zero Friction):**
   - Provide **1-Click Customer Archetype Presets** on the Customer Lookup page so users don't have to manually fill 19 form fields just to test the model.
   - Real-time side-by-side **Before vs. After** delta indicators for the What-If retention simulator.
   - High-contrast visual risk badges (Low, Medium, High, Critical) with micro-pulsing animations.
3. **Executive Data Density:** Clean typography (Inter / Plus Jakarta Sans), KPI metrics with contextual sparkline badges, interactive Plotly visualizations custom-themed with glowing traces and crisp dark tooltips.

---

## 2. Design System Tokens & Style Guide

### 2.1 Color Palette
| Token Name | Hex Code | Visual Role |
|---|---|---|
| **Canvas Deep** | `#080C14` | Main page gradient start |
| **Canvas Base** | `#0E1626` | Deep slate base background |
| **Card Glass Surface** | `rgba(18, 26, 43, 0.75)` | Elevated glassmorphic card containers |
| **Glass Border** | `rgba(0, 240, 255, 0.18)` | Subtle glowing perimeter for cards and widgets |
| **Brand Primary (Neon Cyan)** | `#00F0FF` | Primary action buttons, active tab borders, key KPIs |
| **Brand Secondary (Electric Violet)** | `#8B5CF6` | Secondary accents, gradient fills, SHAP positive drivers |
| **Status Emerald (Safe/Retained)** | `#10B981` | Low risk tier (<25%), positive savings, retained cohorts |
| **Status Amber (Warning)** | `#F59E0B` | Medium risk tier (25-50%), moderate churn risk |
| **Status Rose (Alert/Churn)** | `#F43F5E` | High risk tier (>50%), churn indicators, lost revenue |

### 2.2 Typography Hierarchy
* **Headings:** Bold / Extra-Bold (700/800) with subtle gradient text clipping (`background: linear-gradient(135deg, #00F0FF, #38BDF8)`).
* **Body Text:** Inter 400/500 `#E2E8F0` for maximum legibility on dark surfaces.
* **Captions & Muted Labels:** `#94A3B8` uppercase, letter-spacing `0.06em`.

---

## 3. Stepwise Execution Plan

```
[ Step 1: Core Design Engine ] ───> Implement central CSS & Plotly dark theme
            │
[ Step 2: Home Hub Enhancement ] ─> Interactive Hero + System Health Telemetry
            │
[ Step 3: Executive Overview ] ────> KPI Glow Tiles + Enhanced Cohort Visuals
            │
[ Step 4: Customer Lookup & Presets ] ──> 1-Click Archetypes + Dual Gauge & SHAP
            │
[ Step 5: What-If Simulator ] ────> Before/After Delta Cards + Financial ROI
            │
[ Step 6: Top At-Risk Explorer ] ─> Batch Telemetry + Filterable Data Grid + CSV
            │
[ Step 7: Model Lab Experience ] ──> Interactive Benchmark Matrix + Curves
            │
[ Step 8: Final Polish & Git Push ] ──> 85/85 Test Validation & GitHub Sync
```

---

## 4. Detailed Step-by-Step Implementation Tasks

### Step 1: Centralized UI Design Engine (`src/ui_theme.py`)
* **Objective:** Eliminate redundant inline CSS and ensure consistent visual language across all pages.
* **Components:**
  1. `apply_custom_css()`: Injects universal glassmorphic styles, custom scrollbars, animated badges, and responsive containers.
  2. `get_plotly_dark_template()`: Standardizes all Plotly charts with transparent backgrounds, glowing lines, clean gridlines (`rgba(255,255,255,0.06)`), and unified font styling.
  3. `render_kpi_card(title, value, subtitle, delta, status)`: Reusable HTML metric card with gradient border.

### Step 2: Main Application Hub (`app.py`)
* **Objective:** Create a commanding entrance to the platform with immediate operational status.
* **Enhancements:**
  - Modern hero banner with glowing brand badge and system status indicator (e.g., `● Model Engine Active | 85 Tests Passed`).
  - Interactive "Feature Cards" with hover micro-transitions directing users to the 5 specialized workflows.
  - Quick-start guide highlighting business use cases for executives and data scientists.

### Step 3: Overview Dashboard (`pages/01_📊_Overview.py`)
* **Objective:** Deliver executive-level clarity on dataset health, churn demographics, and revenue at risk.
* **Enhancements:**
  - 4 Glowing KPI Cards: Total Customer Base, Churn Rate %, Revenue at Monthly Risk ($), and Avg Monthly Charges.
  - Interactive Donut & Cohort Charts: Churn by Contract Type, Internet Service Type, and Tenure Bins with custom Plotly hover tooltips.
  - Data Quality Drawer with missing values breakdown and duplicate inspection.

### Step 4: Customer Lookup & 1-Click Archetypes (`pages/02_🔍_Customer_Lookup.py`)
* **Objective:** Streamline individual customer risk analysis with fast presets and actionable explanations.
* **Enhancements:**
  - **1-Click Customer Presets:**
    - ⚡ *High-Risk New Customer* (Month-to-month, Fiber, No tech support, high spend)
    - 🛡️ *Stable Long-Term Family* (Two-year, Partner, Dependents, DSL, low spend)
    - ⚠️ *At-Risk Senior Citizen* (Month-to-month, Electronic check, Streaming services)
  - Animated Risk Probability Meter with color transition (Emerald → Amber → Rose).
  - Top 5 Local SHAP Drivers with plain-English business interpretation.
  - Personalized Retention Playbook showing specific countermeasure offers and calculated ROI.

### Step 5: What-If Counterfactual Simulator (`pages/03_🎛️_What_If.py`)
* **Objective:** Allow retention managers to simulate contract shifts and service bundling in real time.
* **Enhancements:**
  - Dynamic "Before vs. After" comparison cards displaying probability drop ($\Delta\%$).
  - Estimated net revenue preservation and lifetime value uplift.
  - Recommended intervention packages based on simulated risk reduction.

### Step 6: Top At-Risk Customer Explorer (`pages/04_🚨_Top_At_Risk.py`)
* **Objective:** Enable bulk scoring and export for targeted outbound marketing campaigns.
* **Enhancements:**
  - CSV file drop-zone with instant schema validation and error feedback.
  - Interactive data grid with risk badges, sortable by churn probability and monthly charges.
  - Filter sliders (e.g., "Show only High Risk with Monthly Charges > $70").
  - Export enriched CSV with predicted churn probability, risk tier, and assigned retention action.

### Step 7: Model Lab & Explainability (`pages/05_🧪_Model_Lab.py`)
* **Objective:** Provide transparent ML governance, algorithm benchmarks, and global explainability.
* **Enhancements:**
  - Visual benchmark comparison table highlighting the winning algorithm (Logistic Regression vs. XGBoost vs. RF).
  - Interactive toggle for ROC Curve and Precision-Recall Curve.
  - Interactive Confusion Matrix with threshold slider demonstrating the precision/recall trade-off.
  - Global SHAP beeswarm / feature importance plot.

---

## 5. Verification & Testing Protocol

Before marking each step complete and pushing to GitHub:
1. **Syntax & Compilation:** Run `py_compile` across all modified page scripts.
2. **Automated Regression Suite:** Run `pytest tests/ -v` to ensure 85/85 tests continue to pass.
3. **Live UI Verification:** Validate the running Streamlit server at `http://localhost:8501` to confirm smooth rendering, responsive layouts, and zero runtime warnings.
4. **Git Versioning:** Stage, commit with descriptive message, and push cleanly to `main`.
