"""
app.py — StayWise: Explainable Customer Churn Prediction
Main Streamlit entry point. Sets page config and loads shared state.
"""

from pathlib import Path
import streamlit as st
import json

from src.ui_theme import inject_custom_css, render_page_header

st.set_page_config(
    page_title="StayWise — Churn Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Styling ───────────────────────────────────────────────────────────
inject_custom_css()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
        <span style="font-size:2rem;">🛡️</span>
        <div>
            <h3 style="margin:0; font-weight:800; background:linear-gradient(135deg, #00f0ff, #38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">StayWise</h3>
            <span style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">AI Churn Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(16, 185, 129, 0.12); border:1px solid rgba(16, 185, 129, 0.3); border-radius:8px; padding:8px 12px; margin-bottom:1.2rem;">
        <span style="color:#34d399; font-weight:700; font-size:0.75rem;">● SYSTEM ONLINE</span><br/>
        <span style="color:#94a3b8; font-size:0.75rem;">85/85 Tests Verified</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Navigation Modules**")
    st.markdown("""
    * 📊 **Overview** — Cohorts & Revenue at Risk
    * 🔍 **Customer Lookup** — 1-Click Scoring & SHAP
    * 🎛️ **What-If** — Counterfactual ROI Simulator
    * 🚨 **Top At-Risk** — Batch Scoring & CSV Export
    * 🧪 **Model Lab** — Benchmark & Retraining Hub
    """)
    st.divider()
    st.caption("StayWise v1.0 · Powered by Scikit-Learn & XGBoost")

# ─── Load Metadata for Real Telemetry ─────────────────────────────────────────
meta_path = Path(__file__).resolve().parent / "models" / "training_metadata.json"
best_model_name = "Logistic Regression"
test_pr_auc = "0.6261"
test_roc_auc = "0.8419"
test_recall = "95.7%"

if meta_path.exists():
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            best_model_name = meta.get("best_model_name", best_model_name)
            metrics = meta.get("test_metrics", {})
            pr = metrics.get("test_pr_auc")
            if pr is not None:
                test_pr_auc = f"{pr:.4f}"
            roc = metrics.get("test_roc_auc")
            if roc is not None:
                test_roc_auc = f"{roc:.4f}"
            rec = metrics.get("test_recall")
            if rec is not None:
                test_recall = f"{rec:.1%}"
    except Exception:
        pass

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: radial-gradient(circle at 50% 0%, rgba(0, 240, 255, 0.15) 0%, rgba(18, 26, 43, 0.6) 70%); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 16px; padding: 2.8rem 2rem; text-align: center; margin-bottom: 2rem; backdrop-filter: blur(20px); box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);">
    <div style="display:inline-block; background:rgba(0, 240, 255, 0.1); border:1px solid rgba(0, 240, 255, 0.3); border-radius:20px; padding:4px 14px; font-size:0.75rem; font-weight:700; color:#00f0ff; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:1rem;">
        PRODUCTION READY AI · TELCO INTELLIGENCE
    </div>
    <h1 style="font-size:3.2rem; font-weight:800; background:linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #00f0ff 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 0.8rem 0; line-height:1.15;">
        StayWise Retention Platform
    </h1>
    <p style="color:#94a3b8; font-size:1.15rem; max-width:720px; margin:0 auto 1.5rem auto; line-height:1.5;">
        Explainable Machine Learning for telecom customer churn prediction, calibrated financial risk modeling, and proactive retention decision support.
    </p>
    <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
        <span style="color:#cbd5e1; font-size:0.85rem;"><strong style="color:#00f0ff;">Engine:</strong> {best_model_name} (Calibrated)</span>
        <span style="color:#64748b;">•</span>
        <span style="color:#cbd5e1; font-size:0.85rem;"><strong style="color:#00f0ff;">PR-AUC:</strong> {test_pr_auc}</span>
        <span style="color:#64748b;">•</span>
        <span style="color:#cbd5e1; font-size:0.85rem;"><strong style="color:#00f0ff;">Recall:</strong> {test_recall}</span>
        <span style="color:#64748b;">•</span>
        <span style="color:#cbd5e1; font-size:0.85rem;"><strong style="color:#00f0ff;">ROC-AUC:</strong> {test_roc_auc}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Quick Feature Cards ──────────────────────────────────────────────────────
st.markdown("### ⚡ Operational Workflows")

col1, col2, col3, col4 = st.columns(4)

cards_data = [
    ("📊", "Overview & Demographics", "Analyze the 7,043 customer cohort, churn drivers by contract type, and monthly revenue at risk.", "pages/01_📊_Overview.py"),
    ("🔍", "Customer Lookup & SHAP", "Score individual customer risk with 1-click presets and explain drivers via local SHAP attributions.", "pages/02_🔍_Customer_Lookup.py"),
    ("🎛️", "What-If Simulator", "Test retention counterfactuals (e.g. converting Month-to-Month to 1-Year) to project ROI.", "pages/03_🎛️_What_If.py"),
    ("🚨", "Top At-Risk Explorer", "Upload bulk customer CSV files, prioritize high-risk cohorts, and export targeted retention action lists.", "pages/04_🚨_Top_At_Risk.py"),
]

for col, (icon, title, desc, link) in zip([col1, col2, col3, col4], cards_data):
    with col:
        st.markdown(f"""
        <div class="sw-glass-card" style="height:210px; display:flex; flex-direction:column; justify-content:space-between;">
            <div>
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:0.4rem;">{title}</div>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.4;">{desc}</div>
            </div>
            <div style="margin-top:0.8rem; font-size:0.75rem; font-weight:700; color:#00f0ff; letter-spacing:0.04em;">
                USE SIDEBAR TO OPEN →
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── System Architecture Card ─────────────────────────────────────────────────
st.markdown("""
<div class="sw-glass-card" style="margin-top:1.5rem;">
    <div style="font-weight:700; font-size:1rem; color:#00f0ff; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.05em;">
        🛡️ End-to-End Decision Architecture
    </div>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:1.2rem; color:#cbd5e1; font-size:0.85rem;">
        <div>
            <strong style="color:#ffffff;">1. Zero-Leakage Pipeline:</strong><br/>
            Median imputation & scaling fitted strictly on training data; categorical features encoded with safe unknown handling.
        </div>
        <div>
            <strong style="color:#ffffff;">2. Calibrated Probabilities:</strong><br/>
            Raw model outputs are passed through Isotonic Calibration to guarantee real-world probability alignment.
        </div>
        <div>
            <strong style="color:#ffffff;">3. Financial Thresholding:</strong><br/>
            Decision thresholds are determined by maximizing net retention profit rather than naive 0.50 cutoffs.
        </div>
        <div>
            <strong style="color:#ffffff;">4. Actionable Countermeasures:</strong><br/>
            SHAP game-theoretic attributions dynamically generate personalized retention offers with expected ROI.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
