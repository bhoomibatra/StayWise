"""Generate all high-resolution figures for the StayWise Academic Project Report."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "report" / "report_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
EVAL_DETAILS_PATH = MODELS_DIR / "evaluation_details.json"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

# ------------------------------------------------------------------------------
# FIGURE 1: SYSTEM ARCHITECTURE DIAGRAM
# ------------------------------------------------------------------------------
def create_fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color, text_color="black"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=color, edgecolor="#334155", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=text_color)
        ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=8.5, color=text_color, style="italic")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f766e")
        )
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha="center", va="bottom", fontsize=8, color="#0f766e", fontweight="bold")

    # Layer 1: Data Source
    draw_box(0.5, 4.2, 2.5, 1.8, "Raw Data Source", "IBM Telco Churn CSV\n7,043 rows, 21 attributes", "#e0f2fe")
    
    # Layer 2: Preprocessing
    draw_box(3.6, 4.2, 2.6, 1.8, "ColumnTransformer", "Numerical: Median + Scaler\nCategorical: Freq + OneHot", "#fef3c7")
    
    # Layer 3: Pipeline & Models
    draw_box(6.8, 4.2, 2.6, 1.8, "Scikit-Learn Pipeline", "Logistic Regression (C=10.0)\nSerialized best_model.joblib", "#dcfce7")

    # Layer 4: Storage & DB
    draw_box(6.8, 1.0, 2.6, 1.8, "Local Persistence", "SQLite DB (retention.db)\nTasks, Notes, Milestones", "#ede9fe")

    # Layer 5: Web Dashboards
    draw_box(9.8, 4.2, 2.0, 1.8, "FastAPI Backend", "Uvicorn REST APIs\nAuto-Scoring Engine", "#cffafe")
    draw_box(9.8, 1.0, 2.0, 1.8, "React 18 SPA", "Enterprise SaaS UI\nRecharts & 360 Cockpit", "#ffe4e6")

    # Connections
    draw_arrow(3.0, 5.1, 3.6, 5.1, "Clean / Impute")
    draw_arrow(6.2, 5.1, 6.8, 5.1, "Fit on X_train")
    draw_arrow(9.4, 5.1, 9.8, 5.1, "Inference")
    draw_arrow(8.1, 4.2, 8.1, 2.8, "Serialize")
    draw_arrow(9.4, 1.9, 9.8, 1.9, "Track Tasks")
    draw_arrow(10.8, 4.2, 10.8, 2.8, "Serve API / UI")

    plt.title("Figure 3.1: StayWise End-to-End System Architecture", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig1_system_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_system_architecture.png")


# ------------------------------------------------------------------------------
# FIGURE 2: ML WORKFLOW DIAGRAM
# ------------------------------------------------------------------------------
def create_fig2_workflow():
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    steps = [
        ("1. Data Ingestion", "Read telco_churn.csv\nHandle 11 TotalCharges NaNs", "#f1f5f9"),
        ("2. Stratified Split", "80% Train (5,634 rows)\n20% Holdout Test (1,409)", "#fef08a"),
        ("3. Preprocessing", "ColumnTransformer\nStrictly fitted on X_train", "#bae6fd"),
        ("4. 5-Fold Stratified CV", "GridSearchCV Hyperparameter\nTuning (C, k, n_est, depth)", "#fed7aa"),
        ("5. Model Evaluation", "Evaluate LR, KNN, RF, SVM\non untouched X_test", "#bbf7d0"),
        ("6. Pipeline Export", "Select best model (LR)\njoblib.dump() to disk", "#e9d5ff"),
    ]

    for idx, (title, sub, col) in enumerate(steps):
        x = 0.4 + idx * 2.05
        y = 1.6
        rect = patches.FancyBboxPatch(
            (x, y), 1.85, 1.8, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=col, edgecolor="#475569", linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(x + 0.925, y + 1.25, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax.text(x + 0.925, y + 0.6, sub, ha="center", va="center", fontsize=7.8, color="#334155", style="italic")

        if idx < len(steps) - 1:
            ax.annotate(
                "", xy=(x + 2.05, y + 0.9), xytext=(x + 1.85, y + 0.9),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#0f766e")
            )

    plt.title("Figure 3.2: StayWise Leak-Free Machine Learning Workflow Pipeline", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig2_ml_workflow.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_ml_workflow.png")


# ------------------------------------------------------------------------------
# FIGURE 3: CLASS DISTRIBUTION (BAR + DONUT)
# ------------------------------------------------------------------------------
def create_fig3_target_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    categories = ["No Churn (Stay)", "Churned"]
    counts = [5174, 1869]
    pcts = [73.46, 26.54]
    colors = ["#0ea5e9", "#f43f5e"]

    # Bar Chart
    bars = ax1.bar(categories, counts, color=colors, width=0.55, edgecolor="#334155", linewidth=1.2)
    ax1.set_ylabel("Number of Customers", fontsize=10, fontweight="bold")
    ax1.set_title("Customer Churn Class Count", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 6000)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 120,
            f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts, labels=categories, autopct="%1.2f%%", startangle=140,
        colors=colors, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Class Imbalance Ratio (~2.77 : 1)", fontsize=11, fontweight="bold")

    plt.suptitle("Figure 4.1: Empirical Target Class Distribution in IBM Telco Dataset (N = 7,043)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig3_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_target_distribution.png")


# ------------------------------------------------------------------------------
# FIGURE 4: CONFUSION MATRICES (4-PANEL)
# ------------------------------------------------------------------------------
def create_fig4_confusion_matrices():
    if not EVAL_DETAILS_PATH.exists():
        print("Missing evaluation_details.json")
        return

    with open(EVAL_DETAILS_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    models = ["Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine"]
    cm_palettes = ["Blues", "Greens", "Oranges", "Purples"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    for idx, name in enumerate(models):
        ax = axes[idx]
        data = eval_data[name]["confusion_matrix"]
        cm = np.array([[data["tn"], data["fp"]], [data["fn"], data["tp"]]])

        im = ax.imshow(cm, cmap=cm_palettes[idx], interpolation="nearest")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Stay (0)", "Pred Churn (1)"], fontsize=8.5)
        ax.set_yticklabels(["True Stay (0)", "True Churn (1)"], fontsize=8.5)
        ax.set_xlabel("Predicted Label", fontsize=9)
        if idx == 0:
            ax.set_ylabel("True Label", fontsize=9)

        thresh = cm.max() / 2.0
        labels = [
            [f"TN\n{data['tn']}", f"FP\n{data['fp']}"],
            [f"FN\n{data['fn']}", f"TP\n{data['tp']}"],
        ]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, labels[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=10
                )

    plt.suptitle("Figure 6.1: Side-by-Side Holdout Test Confusion Matrices (1,409 Test Subscribers)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig4_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_confusion_matrices.png")


# ------------------------------------------------------------------------------
# FIGURE 5: OVERLAID ROC CURVES
# ------------------------------------------------------------------------------
def create_fig5_roc_curves():
    if not EVAL_DETAILS_PATH.exists():
        return

    with open(EVAL_DETAILS_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    palette = {
        "Logistic Regression": "#0d9488",
        "K-Nearest Neighbors": "#3b82f6",
        "Random Forest": "#f59e0b",
        "Support Vector Machine": "#8b5cf6",
    }

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, m_info in eval_data.items():
        curve = m_info["roc_curve"]
        fpr = [p["fpr"] for p in curve]
        tpr = [p["tpr"] for p in curve]
        auc = m_info["metrics"]["roc_auc"]
        ax.plot(
            fpr, tpr,
            label=f"{name} (ROC-AUC = {auc:.4f})",
            color=palette.get(name, "black"),
            linewidth=2.4
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess / No-Skill (AUC = 0.5000)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 6.2: Overlaid Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig5_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig5_roc_curves.png")


# ------------------------------------------------------------------------------
# FIGURE 6: METRICS COMPARISON BAR CHART
# ------------------------------------------------------------------------------
def create_fig6_metrics_comparison():
    if not METRICS_PATH.exists():
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    summary = metrics_data["metrics_summary"]
    models = list(summary.keys())
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    data_matrix = []
    for m in models:
        row = [
            summary[m]["accuracy"] * 100,
            summary[m]["precision"] * 100,
            summary[m]["recall"] * 100,
            summary[m]["f1"] * 100,
            summary[m]["roc_auc"] * 100,
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    x = np.arange(len(metrics_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    colors = ["#0d9488", "#3b82f6", "#f59e0b", "#8b5cf6"]

    for i, m in enumerate(models):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, data_matrix[i], width, label=m, color=colors[i], edgecolor="#334155", linewidth=0.8)
        # annotate recall specifically
        recall_val = data_matrix[i][2]

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Figure 6.3: Systematic Performance Benchmark Across 4 Machine Learning Models", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax.set_ylim(30, 95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig6_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig6_metric_comparison.png")


# ------------------------------------------------------------------------------
# FIGURE 7: RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
def create_fig7_feature_importance():
    if not METRICS_PATH.exists():
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    rf_features = metrics_data.get("random_forest_features", [])
    if not rf_features:
        return

    names = [f["feature"].replace("cat__", "").replace("num__", "").replace("_", " ") for f in rf_features]
    scores = [f["importance"] * 100 for f in rf_features]

    # Invert order for horizontal plot
    names = names[::-1]
    scores = scores[::-1]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.barh(names, scores, color="#0f766e", edgecolor="#134e4a", height=0.6)

    ax.set_xlabel("Relative Importance (% Mean Gini Impurity Decrease)", fontsize=9.5, fontweight="bold")
    ax.set_title("Figure 6.4: Top 10 Churn Drivers (Random Forest Feature Importance)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 20)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}%", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#0f172a"
        )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig7_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig7_feature_importance.png")


# ------------------------------------------------------------------------------
# FIGURE 8: ENTERPRISE UI DASHBOARD MOCKUP / RENDER
# ------------------------------------------------------------------------------
def create_fig8_dashboard_ui():
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Dark Sidebar
    sidebar = patches.Rectangle((0, 0), 2.4, 7, facecolor="#0f172a", edgecolor="none")
    ax.add_patch(sidebar)
    ax.text(1.2, 6.4, "📡 StayWise", color="white", fontsize=13, fontweight="bold", ha="center")
    ax.text(1.2, 6.05, "Retention SaaS Platform", color="#2dd4bf", fontsize=8, ha="center")

    navs = ["📊 Executive Overview", "👥 Customer Directory", "⚠️ At-Risk Accounts", "📋 Retention Actions", "📈 Analytics & Insights", "⚖️ Model Performance", "⚙️ Settings & Status"]
    for i, nav in enumerate(navs):
        y = 5.3 - i * 0.6
        bg_col = "#1e293b" if i == 0 else "#0f172a"
        text_col = "#2dd4bf" if i == 0 else "#94a3b8"
        r = patches.Rectangle((0.2, y - 0.2), 2.0, 0.45, facecolor=bg_col, edgecolor="none")
        ax.add_patch(r)
        ax.text(0.35, y, nav, color=text_col, fontsize=8.5, va="center", fontweight="bold" if i == 0 else "normal")

    # Header
    header = patches.Rectangle((2.4, 6.0), 9.6, 1.0, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.add_patch(header)
    ax.text(2.7, 6.55, "Executive Overview Dashboard", fontsize=12, fontweight="bold", color="#0f172a")
    ax.text(2.7, 6.25, "Real-time retention KPIs & cohort health across 7,043 subscriber records", fontsize=8, color="#64748b")
    ax.text(11.5, 6.4, "● Pipeline Active: Logistic Regression (C=10.0)", fontsize=8, color="#0f766e", ha="right", fontweight="bold")

    # KPI Cards
    kpis = [
        ("Total Subscribers", "7,043", "100% Scored via Pipeline", "#ffffff", "#0ea5e9"),
        ("High Risk Accounts", "1,039 (14.8%)", "Threshold prob ≥ 60%", "#fff1f2", "#e11d48"),
        ("Predicted Churn", "22.21%", "1,564 Churners vs 5,479 Stay", "#ffffff", "#0d9488"),
        ("Model Holdout ROC", "0.8411", "Accuracy: 80.6% | F1: 0.604", "#ffffff", "#6366f1"),
    ]
    for idx, (title, val, sub, bg, border) in enumerate(kpis):
        x = 2.7 + idx * 2.25
        card = patches.FancyBboxPatch((x, 4.6), 2.1, 1.15, boxstyle="round,pad=0.05,rounding_size=0.1", facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(card)
        ax.text(x + 0.1, 5.5, title.upper(), fontsize=7, color="#64748b", fontweight="bold")
        ax.text(x + 0.1, 5.1, val, fontsize=12, color="#0f172a", fontweight="bold")
        ax.text(x + 0.1, 4.8, sub, fontsize=6.5, color="#475569")

    # Chart 1: Donut (Risk Distribution)
    card1 = patches.FancyBboxPatch((2.7, 0.4), 4.3, 3.9, boxstyle="round,pad=0.05,rounding_size=0.1", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.0)
    ax.add_patch(card1)
    ax.text(2.9, 4.0, "Customer Risk Distribution", fontsize=10, fontweight="bold", color="#0f172a")
    ax.text(2.9, 3.75, "Operational Triage Tiers (High: 14.8%, Med: 18.7%, Low: 66.6%)", fontsize=7.5, color="#64748b")
    # Draw simulated donut
    donut = patches.Wedge((4.85, 2.2), 1.2, 0, 360, width=0.45, facecolor="#0ea5e9")
    ax.add_patch(donut)
    wedge_hi = patches.Wedge((4.85, 2.2), 1.2, 0, 53, width=0.45, facecolor="#f43f5e")
    ax.add_patch(wedge_hi)
    wedge_med = patches.Wedge((4.85, 2.2), 1.2, 53, 120, width=0.45, facecolor="#f59e0b")
    ax.add_patch(wedge_med)
    ax.text(4.85, 2.2, "7,043\nTotal", fontsize=8.5, fontweight="bold", ha="center", va="center", color="#0f172a")

    # Chart 2: Contract Breakdown
    card2 = patches.FancyBboxPatch((7.3, 0.4), 4.4, 3.9, boxstyle="round,pad=0.05,rounding_size=0.1", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.0)
    ax.add_patch(card2)
    ax.text(7.5, 4.0, "Churn Rate by Contract Structure", fontsize=10, fontweight="bold", color="#0f172a")
    ax.text(7.5, 3.75, "Month-to-month (42.7%) vs 1-Year (11.3%) vs 2-Year (2.8%)", fontsize=7.5, color="#64748b")
    # Draw mini bars
    contracts = [("Month-to-month", 0.427, "#f43f5e"), ("One year", 0.113, "#f59e0b"), ("Two year", 0.028, "#10b981")]
    for j, (cname, crate, ccol) in enumerate(contracts):
        cy = 3.0 - j * 0.9
        ax.text(7.5, cy + 0.35, cname, fontsize=8, color="#334155", fontweight="bold")
        ax.text(11.4, cy + 0.35, f"{crate*100:.1f}% Churn", fontsize=8, color=ccol, ha="right", fontweight="bold")
        bg_bar = patches.Rectangle((7.5, cy), 3.9, 0.22, facecolor="#f1f5f9", edgecolor="none")
        ax.add_patch(bg_bar)
        fg_bar = patches.Rectangle((7.5, cy), 3.9 * crate, 0.22, facecolor=ccol, edgecolor="none")
        ax.add_patch(fg_bar)

    plt.title("Figure 7.1: StayWise Enterprise Executive Dashboard Interface", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig8_dashboard_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig8_dashboard_overview.png")


# ------------------------------------------------------------------------------
# FIGURE 9: CUSTOMER 360 & RETENTION WORKFLOW
# ------------------------------------------------------------------------------
def create_fig9_customer_360():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Main Card
    main_box = patches.FancyBboxPatch((0.5, 0.4), 11.0, 5.7, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(main_box)

    # Top Header
    top_header = patches.Rectangle((0.5, 5.0), 11.0, 1.1, facecolor="#0f172a", edgecolor="none")
    ax.add_patch(top_header)
    ax.text(0.9, 5.65, "Customer 360° Profile & Retention Cockpit — Subscriber: 9237-HQITU", color="white", fontsize=12, fontweight="bold")
    ax.text(0.9, 5.25, "Model Churn Likelihood: 86.4% | Risk Tier: High Risk | Historical Label: Yes (Churned)", color="#2dd4bf", fontsize=9)
    ax.text(11.1, 5.5, "[Assign Retention Follow-Up]", color="white", fontsize=8.5, ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d9488", edgecolor="none"))

    # Section 1: Empirical Drivers
    d_box = patches.FancyBboxPatch((0.9, 2.7), 4.9, 2.0, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor="#fff1f2", edgecolor="#fda4af")
    ax.add_patch(d_box)
    ax.text(1.1, 4.4, "🚨 Empirical Risk Indicators (High Hazard)", fontsize=9.5, fontweight="bold", color="#9f1239")
    ax.text(1.1, 4.0, "• Month-to-Month Contract: Zero cancellation friction (~42.7% cohort churn).", fontsize=7.5, color="#4c0519")
    ax.text(1.1, 3.6, "• Early Tenure Hazard: Month 2 of subscription (peak drop-off window).", fontsize=7.5, color="#4c0519")
    ax.text(1.1, 3.2, "• Unbundled Fiber Optic: Premium speed without Tech Support / Online Security.", fontsize=7.5, color="#4c0519")
    ax.text(1.1, 2.85, "• Electronic Check Billing: Manual payment friction increases attrition.", fontsize=7.5, color="#4c0519")

    # Section 2: Account & Financials
    a_box = patches.FancyBboxPatch((6.2, 2.7), 4.9, 2.0, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor="#f8fafc", edgecolor="#e2e8f0")
    ax.add_patch(a_box)
    ax.text(6.4, 4.4, "💳 Financial & Service Footprint", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax.text(6.4, 4.0, "Monthly Rate: $70.70/mo  |  Tenure: 2 months  |  Total Charges: $151.65", fontsize=8, color="#334155")
    ax.text(6.4, 3.6, "Internet: Fiber Optic  |  Tech Support: No  |  Online Security: No", fontsize=8, color="#334155")
    ax.text(6.4, 3.2, "Phone Service: Yes (1 Line)  |  Streaming TV/Movies: No", fontsize=8, color="#334155")
    ax.text(6.4, 2.85, "Demographics: Female, Non-Senior, Single, Independent Household", fontsize=8, color="#334155")

    # Section 3: Recommended Retention Actions & SQLite Log
    r_box = patches.FancyBboxPatch((0.9, 0.7), 10.2, 1.7, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor="#f0fdf4", edgecolor="#86efac")
    ax.add_patch(r_box)
    ax.text(1.1, 2.1, "📋 Recommended Action & Persistent SQLite Log", fontsize=9.5, fontweight="bold", color="#166534")
    ax.text(1.1, 1.7, "1. Contract Upgrade Offer: Propose 15% discount for 1-year contract extension. (Priority: High | Assigned: Sarah Chen [CSM])", fontsize=7.8, color="#14532d")
    ax.text(1.1, 1.35, "2. Value Add-On Pilot: Enroll customer in complimentary 6-month Tech Support trial. (Priority: Medium)", fontsize=7.8, color="#14532d")
    ax.text(1.1, 1.0, "3. Auto-Pay Migration: Offer $5 monthly recurring statement credit to switch from Electronic Check to ACH transfer.", fontsize=7.8, color="#14532d")

    plt.title("Figure 7.2: StayWise Customer 360° Account Profile and Retention Action Drawer", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "fig9_customer_360_ui.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig9_customer_360_ui.png")


if __name__ == "__main__":
    print("Generating report assets...")
    create_fig1_architecture()
    create_fig2_workflow()
    create_fig3_target_distribution()
    create_fig4_confusion_matrices()
    create_fig5_roc_curves()
    create_fig6_metrics_comparison()
    create_fig7_feature_importance()
    create_fig8_dashboard_ui()
    create_fig9_customer_360()
    print("All report assets successfully generated in report/report_assets!")
