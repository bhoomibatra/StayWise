"""
pages/01_📊_Overview.py — StayWise Overview Dashboard
Displays historical dataset statistics and model training status.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import clean, get_quality_report, load_raw
from src.features import engineer
from src.predict import load_pipeline
from src.ui_theme import inject_custom_css, get_plotly_layout, render_page_header

st.set_page_config(page_title="Overview — StayWise", page_icon="📊", layout="wide")

# ─── Global Styles ────────────────────────────────────────────────────────────
inject_custom_css()
render_page_header("Executive Churn & Demographic Overview", "Exploratory statistics on the 7,043 customer cohort prior to predictive modeling", "Cohort Intelligence")

# ─── Load Data & Model ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def _load_data():
    raw = load_raw()
    df = clean(raw)
    df = engineer(df)
    return df

@st.cache_resource(show_spinner=False)
def _load_model():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None

try:
    df = _load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

payload = _load_model()
has_model = payload is not None

# ─── KPI Cards ────────────────────────────────────────────────────────────────
qr = get_quality_report(df)
churn_rate = qr["churn_rate"]
n_churners = int(df["Churn"].sum())
n_total = len(df)
revenue_at_risk = (df[df["Churn"] == 1]["MonthlyCharges"].sum())
avg_monthly = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)
kpis = [
    ("Total Customers", f"{n_total:,}", "Baseline active records"),
    ("Historical Churn Rate", f"{churn_rate:.1%}", f"{n_churners:,} defected customers"),
    ("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", "Monthly recurring revenue"),
    ("Average Monthly Bill", f"${avg_monthly:.2f}", "Across all active services"),
]

for col, (label, val, sub) in zip([col1, col2, col3, col4], kpis):
    with col:
        st.markdown(f"""
        <div class="sw-glass-card">
            <div class="sw-kpi-container">
                <div class="sw-kpi-num">{val}</div>
                <div class="sw-kpi-title">{label}</div>
                <div class="sw-kpi-subtitle">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Model Status Pill ────────────────────────────────────────────────────────
if has_model:
    meta = payload["metadata"]
    best = meta.get("best_model_name", "Logistic Regression")
    threshold = meta.get("threshold", 0.06)
    test_pr = meta.get("test_metrics", {}).get("test_pr_auc", 0.6261)
    st.markdown(f"""
    <div style="background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.25); border-radius: 10px; padding: 10px 16px; margin: 1rem 0; display: flex; align-items: center; justify-content: space-between;">
        <span style="color: #f1f5f9; font-size: 0.88rem;">
            🟢 <strong>Active Model:</strong> {best} | <strong>Threshold:</strong> {threshold} | <strong>Test PR-AUC:</strong> {test_pr:.4f}
        </span>
        <span style="color: #00f0ff; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">Calibrated & Live</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Cohort Visualizations ────────────────────────────────────────────────────
layout = get_plotly_layout()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Contract Breakdown", "⏳ Tenure Cohorts", "💳 Payment Methods", "🔬 Data Quality Audit"])

with tab1:
    c1, c2 = st.columns([3, 2])
    grp = df.groupby("Contract")["Churn"].agg(["mean", "count"]).reset_index()
    grp.columns = ["Contract", "Churn Rate", "Total"]
    
    with c1:
        fig1 = px.bar(
            grp, x="Contract", y="Churn Rate",
            text_auto=".1%",
            title="Churn Rate by Contract Term",
            color="Churn Rate",
            color_continuous_scale=[[0, "#0284c7"], [0.5, "#f59e0b"], [1, "#f43f5e"]],
        )
        fig1.update_layout(**layout, coloraxis_showscale=False)
        fig1.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.markdown("""
        <div class="sw-glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
            <h4 style="color:#00f0ff; margin-top:0;">Contract Insight</h4>
            <p style="color:#cbd5e1; font-size:0.9rem; line-height:1.5;">
                <strong>Month-to-Month contracts</strong> represent the single largest churn vulnerability, with churn rates exceeding <strong>42%</strong>.
            </p>
            <p style="color:#94a3b8; font-size:0.85rem; line-height:1.4;">
                In contrast, <strong>One-Year</strong> (~11%) and <strong>Two-Year</strong> (~3%) contracts exhibit dramatic customer stickiness.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    grp2 = df.groupby("tenure_group")["Churn"].mean().reset_index()
    grp2.columns = ["Tenure Group", "Churn Rate"]
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    grp2["Tenure Group"] = pd.Categorical(grp2["Tenure Group"], categories=order, ordered=True)
    grp2 = gr2 = grp2.sort_values("Tenure Group")
    
    fig2 = px.area(
        grp2, x="Tenure Group", y="Churn Rate",
        title="Churn Hazard by Customer Tenure (Months)",
        markers=True,
    )
    fig2.update_traces(line_color="#00f0ff", fillcolor="rgba(0, 240, 255, 0.15)")
    fig2.update_layout(**layout)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    grp3 = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
    grp3.columns = ["Payment Method", "Churn Rate"]
    fig3 = px.bar(
        grp3.sort_values("Churn Rate", ascending=True),
        x="Churn Rate", y="Payment Method",
        orientation="h",
        text_auto=".1%",
        title="Churn Prevalence Across Payment Methods",
        color="Churn Rate",
        color_continuous_scale="Reds",
    )
    fig3.update_layout(**layout, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    q1, q2 = st.columns(2)
    with q1:
        dist_df = pd.DataFrame({
            "Class": ["Retained (No)", "Churned (Yes)"],
            "Count": [int((df["Churn"] == 0).sum()), int((df["Churn"] == 1).sum())]
        })
        fig_pie = px.pie(
            dist_df, values="Count", names="Class",
            title="Class Imbalance (26.5% Defection)",
            color="Class",
            color_discrete_map={"Retained (No)": "#0ea5e9", "Churned (Yes)": "#f43f5e"},
            hole=0.55,
        )
        fig_pie.update_layout(**layout)
        st.plotly_chart(fig_pie, use_container_width=True)
    with q2:
        st.markdown(f"""
        <div class="sw-glass-card">
            <h4 style="color:#00f0ff; margin-top:0;">Dataset Integrity & Cleanliness</h4>
            <table style="width:100%; color:#cbd5e1; font-size:0.85rem; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <td style="padding:6px 0;">Total Rows</td>
                    <td style="text-align:right; font-weight:700; color:#ffffff;">{qr['n_rows']:,}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 0;">Total Features</td>
                    <td style="text-align:right; font-weight:700; color:#ffffff;">{qr['n_cols']}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <td style="padding:6px 0;">Duplicate Rows</td>
                    <td style="text-align:right; font-weight:700; color:#f59e0b;">{qr['n_duplicates']} (handled)</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;">Missing TotalCharges</td>
                    <td style="text-align:right; font-weight:700; color:#34d399;">11 (coerced & imputed)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
