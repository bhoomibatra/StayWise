"""
app.py — StayWise: Explainable Customer Churn Prediction
Main Streamlit entry point. Sets page config and loads shared state.
"""

import streamlit as st

st.set_page_config(
    page_title="StayWise — Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(14, 165, 233, 0.2);
}

/* Cards */
.sw-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(14, 165, 233, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* KPI tiles */
.sw-kpi {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(30, 41, 59, 0.8));
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.sw-kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0ea5e9;
}
.sw-kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Risk badges */
.badge-low      { background: #064e3b; color: #6ee7b7; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-medium   { background: #451a03; color: #fcd34d; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-high     { background: #450a0a; color: #fca5a5; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-critical { background: #4c0519; color: #f9a8d4; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; animation: pulse 1.5s infinite; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Section header */
.sw-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0ea5e9;
    border-bottom: 1px solid rgba(14, 165, 233, 0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Metric delta */
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #94a3b8;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0ea5e9;
    border-bottom-color: #0ea5e9;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ StayWise")
    st.markdown("*Explainable Churn Intelligence*")
    st.divider()
    st.markdown("""
    **Navigate using the pages below.**

    | Page | Purpose |
    |---|---|
    | 📊 Overview | Dataset stats & risk summary |
    | 🔍 Customer Lookup | Single prediction + advice |
    | 🎛️ What-If | Simulate scenario changes |
    | 🚨 Top At-Risk | Batch scoring & export |
    | 🧪 Model Lab | Train, compare, evaluate |
    """)
    st.divider()
    st.caption("IBM Telco Dataset · StayWise v1.0")

# ─── Home page ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 2rem;">
    <h1 style="font-size:3rem; font-weight:800; background: linear-gradient(135deg, #0ea5e9, #38bdf8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        🛡️ StayWise
    </h1>
    <p style="color:#94a3b8; font-size:1.2rem; margin-top:0.5rem;">
        Explainable Customer Churn Prediction & Decision-Support System
    </p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
cards = [
    ("📊", "Overview", "Dataset stats,\nchurn breakdown", "pages/01_📊_Overview.py"),
    ("🔍", "Customer Lookup", "Predict + explain\nindividual risk", "pages/02_🔍_Customer_Lookup.py"),
    ("🎛️", "What-If", "Simulate contract\nand service changes", "pages/03_🎛️_What_If.py"),
    ("🚨", "Top At-Risk", "Batch score\n& CSV export", "pages/04_🚨_Top_At_Risk.py"),
]

for col, (icon, name, desc, _) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div class="sw-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-weight:600; color:#e2e8f0; margin:0.4rem 0;">{name}</div>
            <div style="font-size:0.8rem; color:#94a3b8;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="sw-card" style="margin-top:1rem;">
    <p style="color:#94a3b8; font-size:0.9rem; margin:0;">
        ⬅️ Use the <strong>sidebar</strong> to navigate between pages.
        Start with <strong>🧪 Model Lab</strong> to train the models, then explore predictions.
    </p>
</div>
""", unsafe_allow_html=True)
