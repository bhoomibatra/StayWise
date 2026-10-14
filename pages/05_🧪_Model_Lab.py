"""
pages/05_🧪_Model_Lab.py — StayWise Model Lab
Train models, compare metrics, view curves, calibration, and SHAP importance.
Avoids silent retraining on every rerun using session state.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import (
    plot_calibration,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_profit_vs_threshold,
    plot_roc_curves,
    compute_curves_for_model,
)
from src.explain import compute_global_importance, plot_global_importance
from src.predict import load_pipeline

st.set_page_config(page_title="Model Lab — StayWise", page_icon="🧪", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.95);}
.sw-card{background:rgba(30,41,59,0.7);border:1px solid rgba(14,165,233,0.15);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.metric-good{color:#10b981;font-weight:700;}
.metric-warn{color:#f59e0b;font-weight:700;}
</style>""", unsafe_allow_html=True)

st.markdown("## 🧪 Model Lab")
st.caption("Train, compare, and evaluate the four StayWise classification models")

# ─── Check if model exists ────────────────────────────────────────────────────
model_path = Path(__file__).resolve().parent.parent / "models" / "best_pipeline.joblib"
comparison_path = Path(__file__).resolve().parent.parent / "reports" / "model_comparison.csv"
meta_path = Path(__file__).resolve().parent.parent / "models" / "training_metadata.json"

has_model = model_path.exists()

# ─── Train button (guard against silent retraining) ───────────────────────────
if "training_done" not in st.session_state:
    st.session_state["training_done"] = has_model

col_btn, col_status = st.columns([1, 3])
with col_btn:
    train_btn = st.button(
        "🚀 Train All Models" if not st.session_state["training_done"] else "🔄 Retrain Models",
        type="primary",
        use_container_width=True,
    )
with col_status:
    if st.session_state["training_done"]:
        st.success("✅ Model trained and ready.")
    else:
        st.info("ℹ️ No trained model found. Click **Train All Models** to begin.")

if train_btn:
    with st.spinner("Training 4 models — this may take 1–3 minutes …"):
        progress_bar = st.progress(0, "Initialising …")
        try:
            from src.train import train as run_training
            progress_bar.progress(10, "Loading data …")
            results = run_training()
            progress_bar.progress(100, "Done!")
            st.session_state["training_done"] = True
            st.session_state["training_results"] = results
            st.success(f"✅ Training complete! Best model: **{results['best_model_name']}**")
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Training failed: {e}")
            raise

# ─── Display results if model exists ──────────────────────────────────────────
if not model_path.exists():
    st.warning("Train the model first to see results.")
    st.stop()

# Load artifacts
@st.cache_resource
def _get_payload():
    return load_pipeline()

@st.cache_data
def _load_comparison():
    if comparison_path.exists():
        return pd.read_csv(comparison_path)
    return None

@st.cache_data
def _load_meta():
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}

payload = _get_payload()
comp_df = _load_comparison()
meta = _load_meta()

best_name = meta.get("best_model_name", "Unknown")
threshold = meta.get("threshold", 0.5)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📊 Comparison", "📈 Curves", "🎯 Test Metrics", "🧠 Explainability"])

with t1:
    st.markdown(f"### Model Comparison — Selected: **{best_name}** (by validation PR-AUC)")
    if comp_df is not None:
        display_cols = ["model", "cv_pr_auc", "cv_roc_auc", "cv_f1",
                        "val_pr_auc", "val_roc_auc", "val_f1", "val_recall", "train_time_s"]
        available = [c for c in display_cols if c in comp_df.columns]
        styled = comp_df[available].style.highlight_max(
            subset=[c for c in ["cv_pr_auc", "val_pr_auc", "val_roc_auc", "val_f1"] if c in available],
            color="#064e3b",
        ).format({c: "{:.4f}" for c in available if c not in ["model", "train_time_s"]})
        st.dataframe(styled, use_container_width=True)
        st.caption("Green = best value per column. Selection criterion: **val_pr_auc**")

        # Bar chart
        if "val_pr_auc" in comp_df.columns:
            import plotly.express as px
            fig = px.bar(
                comp_df, x="model", y="val_pr_auc",
                color="model", text_auto=".3f",
                title="Validation PR-AUC — All Models",
                color_discrete_sequence=["#0ea5e9", "#f59e0b", "#10b981", "#ef4444"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)",
                font=dict(family="Inter", color="#e2e8f0"), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("model_comparison.csv not found.")

with t2:
    st.markdown("### Evaluation Curves — Best Model on Validation Set")

    @st.cache_data
    def _get_val_predictions():
        """Recompute val predictions using saved pipeline."""
        try:
            from src.data import clean, load_raw
            from src.features import engineer
            from src.data import split

            raw = load_raw()
            df = clean(raw)
            df = engineer(df)
            _, X_val, _, _, y_val, _ = split(df)
            pipeline = payload["pipeline"]
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            return y_val.values, y_pred, y_prob
        except Exception:
            return None, None, None

    y_true, y_pred, y_prob = _get_val_predictions()

    if y_true is not None:
        tab_roc, tab_pr, tab_cal, tab_cm, tab_profit = st.tabs(
            ["ROC", "PR", "Calibration", "Confusion Matrix", "Profit vs Threshold"]
        )
        curves = compute_curves_for_model(y_true, y_prob)

        with tab_roc:
            fig = plot_roc_curves({best_name: curves["roc"]})
            st.plotly_chart(fig, use_container_width=True)
        with tab_pr:
            fig = plot_pr_curves({best_name: curves["pr"]})
            st.plotly_chart(fig, use_container_width=True)
        with tab_cal:
            fig = plot_calibration(y_true, y_prob, best_name)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("A well-calibrated model's curve should follow the diagonal.")
        with tab_cm:
            fig = plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix — {best_name} (val set)")
            st.plotly_chart(fig, use_container_width=True)
        with tab_profit:
            fig = plot_profit_vs_threshold(y_true, y_prob, current_threshold=threshold)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Selected threshold: **{threshold}** (profit-maximising on validation set)")
    else:
        st.warning("Dataset required to recompute validation curves.")

with t3:
    st.markdown("### ✅ Final Test Set Metrics")
    st.markdown(
        "> These metrics were computed **once** on the held-out test set, "
        "after all model selection and threshold decisions were finalised on the validation set."
    )
    test_metrics = meta.get("test_metrics", {})
    if test_metrics:
        cols = st.columns(3)
        metric_items = [
            ("PR-AUC", "test_pr_auc"),
            ("ROC-AUC", "test_roc_auc"),
            ("F1-Score", "test_f1"),
            ("Precision", "test_precision"),
            ("Recall", "test_recall"),
            ("Accuracy", "test_accuracy"),
        ]
        for i, (label, key) in enumerate(metric_items):
            with cols[i % 3]:
                val = test_metrics.get(key, "—")
                st.metric(label, f"{val:.4f}" if isinstance(val, float) else val)
    else:
        st.info("Test metrics not available yet.")

with t4:
    st.markdown("### 🧠 Global Feature Importance (SHAP)")
    st.caption(
        "⚠️ SHAP values describe what the model uses — not causal proof of churn drivers."
    )

    @st.cache_data(show_spinner="Computing SHAP importance …")
    def _get_importance():
        try:
            from src.data import clean, load_raw
            from src.features import engineer
            from src.data import split
            raw = load_raw()
            df = clean(raw)
            df = engineer(df)
            X_train, _, _, _, _, _ = split(df)
            return compute_global_importance(payload["pipeline"], X_train, max_samples=150)
        except Exception as e:
            return None

    importance_df = _get_importance()
    if importance_df is not None:
        st.plotly_chart(plot_global_importance(importance_df), use_container_width=True)
        with st.expander("View raw importance table"):
            st.dataframe(importance_df.head(30), use_container_width=True)
    else:
        st.info("SHAP importance unavailable — dataset may be missing or SHAP incompatible with this model.")
