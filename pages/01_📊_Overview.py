"""
pages/01_📊_Overview.py — StayWise Overview Dashboard
Displays historical dataset statistics and model training status.
Does NOT use model predictions for the historical stats section.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import clean, get_quality_report, load_raw
from src.features import engineer
from src.predict import load_pipeline, _risk_category

st.set_page_config(page_title="Overview — StayWise", page_icon="📊", layout="wide")

# ─── Load shared CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a); color:#e2e8f0; }
[data-testid="stSidebar"] { background: rgba(15,23,42,0.95); }
.sw-card { background:rgba(30,41,59,0.7); border:1px solid rgba(14,165,233,0.15); border-radius:12px; padding:1.5rem; margin-bottom:1rem; }
.sw-kpi-value { font-size:2rem; font-weight:700; color:#0ea5e9; }
.sw-kpi-label { font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
</style>""", unsafe_allow_html=True)

st.markdown("## 📊 Overview Dashboard")
st.caption("Historical dataset statistics — not model predictions")

# ─── Load data (cached) ───────────────────────────────────────────────────────
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
    has_data = True
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

payload = _load_model()
has_model = payload is not None

# ─── KPI tiles ────────────────────────────────────────────────────────────────
qr = get_quality_report(df)
churn_rate = qr["churn_rate"]
n_churners = int(df["Churn"].sum())
n_total = len(df)
revenue_at_risk = (df[df["Churn"] == 1]["MonthlyCharges"].sum())
avg_monthly = df["MonthlyCharges"].mean()

cols = st.columns(4)
kpis = [
    ("Total Customers", f"{n_total:,}", None),
    ("Churn Rate", f"{churn_rate:.1%}", "26.5% industry avg"),
    ("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo", f"{n_churners:,} churners"),
    ("Avg Monthly Bill", f"${avg_monthly:.2f}", None),
]
for col, (label, value, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""
        <div class="sw-card" style="text-align:center;">
            <div class="sw-kpi-value">{value}</div>
            <div class="sw-kpi-label">{label}</div>
            {"<div style='color:#64748b;font-size:0.75rem;margin-top:4px;'>" + sub + "</div>" if sub else ""}
        </div>""", unsafe_allow_html=True)

# ─── Model status badge ───────────────────────────────────────────────────────
if has_model:
    meta = payload["metadata"]
    best = meta.get("best_model_name", "Unknown")
    threshold = meta.get("threshold", "—")
    test_pr = meta.get("test_metrics", {}).get("test_pr_auc", "—")
    st.success(f"✅ Model trained — **{best}** | Threshold: {threshold} | Test PR-AUC: {test_pr}")
else:
    st.warning("⚠️ No trained model found. Go to **🧪 Model Lab** to train.")

st.divider()

# ─── Churn breakdown charts ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["By Contract", "By Tenure Group", "By Payment Method", "Data Quality"])

PLOT_STYLE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
)

with tab1:
    grp = df.groupby("Contract")["Churn"].mean().reset_index()
    grp.columns = ["Contract", "Churn Rate"]
    fig = px.bar(grp, x="Contract", y="Churn Rate", color="Churn Rate",
                 color_continuous_scale="Blues", text_auto=".1%",
                 title="Churn Rate by Contract Type")
    fig.update_layout(**PLOT_STYLE, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    grp2 = df.groupby("tenure_group")["Churn"].mean().reset_index()
    grp2.columns = ["Tenure Group", "Churn Rate"]
    # Keep order
    order = ["0-12", "13-24", "25-48", "49-60", "61+"]
    grp2["Tenure Group"] = pd.Categorical(grp2["Tenure Group"], categories=order, ordered=True)
    grp2 = grp2.sort_values("Tenure Group")
    fig2 = px.bar(grp2, x="Tenure Group", y="Churn Rate", color="Churn Rate",
                  color_continuous_scale="Oranges", text_auto=".1%",
                  title="Churn Rate by Tenure Group (months)")
    fig2.update_layout(**PLOT_STYLE, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    grp3 = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
    grp3.columns = ["Payment Method", "Churn Rate"]
    fig3 = px.bar(grp3, x="Payment Method", y="Churn Rate", color="Churn Rate",
                  color_continuous_scale="Reds", text_auto=".1%",
                  title="Churn Rate by Payment Method")
    fig3.update_layout(**PLOT_STYLE, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class Distribution**")
        dist_df = pd.DataFrame({
            "Class": ["No Churn (0)", "Churn (1)"],
            "Count": [int((df["Churn"] == 0).sum()), int((df["Churn"] == 1).sum())]
        })
        fig_pie = px.pie(dist_df, values="Count", names="Class",
                         color_discrete_sequence=["#0ea5e9", "#ef4444"],
                         title="Churn vs No-Churn")
        fig_pie.update_layout(**PLOT_STYLE)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("**Data Quality Summary**")
        st.markdown(f"- Rows: **{qr['n_rows']:,}**")
        st.markdown(f"- Columns: **{qr['n_cols']}**")
        st.markdown(f"- Duplicates: **{qr['n_duplicates']}**")
        st.markdown(f"- Total missing values: **{qr['n_missing_total']}**")
        if qr["missing_by_col"]:
            st.markdown("- Missing by column:")
            for col_name, cnt in qr["missing_by_col"].items():
                st.markdown(f"  - `{col_name}`: {cnt}")

# ─── High-risk summary (model-generated) ─────────────────────────────────────
if has_model:
    st.divider()
    st.markdown("### 🚨 High-Risk Customer Summary *(model predictions)*")
    from src.predict import predict_batch
    from src.features import engineer

    @st.cache_data(show_spinner="Scoring all customers …")
    def _score_all():
        return predict_batch(df.drop(columns=["Churn"]), payload=payload)

    scored = _score_all()
    n_high = int((scored["churn_probability"] >= payload["metadata"]["threshold"]).sum())
    n_critical = int((scored["churn_probability"] >= 0.80).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Churners", f"{n_high:,}", f"of {n_total:,} customers")
    c2.metric("Critical Risk (>80%)", f"{n_critical:,}")
    c3.metric("Predicted Churn Rate", f"{n_high/n_total:.1%}")

    top5 = scored.sort_values("churn_probability", ascending=False).head(5)
    st.markdown("**Top 5 Highest-Risk Customers (model prediction)**")
    st.dataframe(top5, use_container_width=True)
