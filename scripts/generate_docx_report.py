"""Generate complete, professional academic report for StayWise in .docx and .pdf format."""

from datetime import datetime
import json
from pathlib import Path
import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = REPORT_DIR / "report_assets"
DOCX_OUTPUT_PATH = REPORT_DIR / "StayWise_Project_Report.docx"
PDF_OUTPUT_PATH = REPORT_DIR / "StayWise_Project_Report.pdf"

# Color Palette Constants
COLOR_PRIMARY = RGBColor(15, 23, 42)     # Slate 900
COLOR_SECONDARY = RGBColor(13, 148, 136) # Teal 600
COLOR_TEXT = RGBColor(51, 65, 85)        # Slate 700
COLOR_MUTED = RGBColor(100, 116, 139)    # Slate 500

def set_cell_background(cell, fill_hex):
    """Set shading color for a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal cell margins in twips."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)

def set_table_borders(table, color="cbd5e1", sz="4"):
    """Apply clean subtle borders to a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_report():
    doc = docx.Document()

    # Configure Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = COLOR_TEXT
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        p.paragraph_format.space_before = Pt(36)
        p.paragraph_format.space_after = Pt(12)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        p.paragraph_format.space_after = Pt(24)
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        return p

    def add_heading_3(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        return p

    def add_body(text, bold_prefix=""):
        p = doc.add_paragraph()
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_PRIMARY
        p.add_run(text)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_PRIMARY
        p.add_run(text)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(3)
        return p

    def add_callout(text, title="KEY TAKEAWAY"):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.right_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(8)
        run_t = p.add_run(f"[{title}] ")
        run_t.font.bold = True
        run_t.font.color.rgb = COLOR_SECONDARY
        run_b = p.add_run(text)
        run_b.font.italic = True

    def add_figure(img_path, caption):
        if not img_path.exists():
            print(f"Warning: Missing figure image: {img_path}")
            return
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(str(img_path), width=Inches(6.2))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run(caption)
        run_cap.font.size = Pt(9.5)
        run_cap.font.bold = True
        run_cap.font.color.rgb = COLOR_MUTED
        p_cap.paragraph_format.space_after = Pt(14)

    # ==========================================================================
    # 1. TITLE PAGE
    # ==========================================================================
    add_title("STAYWISE — CUSTOMER CHURN PREDICTION AND RETENTION INTELLIGENCE USING MACHINE LEARNING")
    add_subtitle("A Complete Machine Learning Project & Enterprise Architecture Report")

    p_mid = doc.add_paragraph()
    p_mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_mid.add_run(
        "A Project Report Submitted in Partial Fulfillment of the Requirements\n"
        "for the Degree of Bachelor of Technology in Computer Science and Engineering"
    )
    r.font.size = Pt(11)
    r.font.italic = True
    p_mid.paragraph_format.space_after = Pt(40)

    # Metadata Table
    table_meta = doc.add_table(rows=6, cols=2)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_meta.autofit = False

    meta_rows = [
        ("Candidate Name:", "[Student Name]"),
        ("University Roll No. / Reg No.:", "[Roll Number / University Reg No.]"),
        ("Department:", "Department of Computer Science and Engineering"),
        ("College / Institution:", "[College / Institution Name, City]"),
        ("Faculty Supervisor / Guide:", "[Faculty Guide Name, Designation]"),
        ("Academic Year:", "[Academic Year, e.g., 2025–2026]"),
    ]

    for idx, (label, val) in enumerate(meta_rows):
        row = table_meta.rows[idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.width = Inches(2.6)
        cell_val.width = Inches(3.8)

        p_l = cell_lbl.paragraphs[0]
        r_l = p_l.add_run(label)
        r_l.font.bold = True
        r_l.font.size = Pt(10.5)
        r_l.font.color.rgb = COLOR_PRIMARY

        p_v = cell_val.paragraphs[0]
        r_v = p_v.add_run(val)
        r_v.font.size = Pt(10.5)

    doc.add_page_break()

    # ==========================================================================
    # 2. ABSTRACT & KEYWORDS
    # ==========================================================================
    add_heading_1("Abstract")
    add_body(
        "Customer churn—the voluntary termination of a service agreement by a subscriber—represents one of the most critical operational and economic challenges facing the modern telecommunications industry. In competitive subscription markets, the financial expenditure required to acquire a new subscriber is estimated to be five to seven times greater than the investment required to retain an existing customer. Consequently, developing accurate, leak-free, and interpretable machine learning systems capable of forecasting customer defection before cancellation occurs is a high-priority enterprise objective."
    )
    add_body(
        "This project report presents StayWise, an end-to-end customer churn intelligence and retention management platform developed using the benchmark IBM Telco Customer Churn dataset (7,043 subscriber records, 21 attributes). StayWise enforces a strict zero-data-leakage preprocessing protocol using Scikit-learn's ColumnTransformer, systematically resolving empirical anomalies such as whitespace string representations in cumulative billing records through median imputation, standard scaling of continuous variables (tenure, monthly charges, total charges), and dummy encoding of 16 categorical features with unknown category mitigation."
    )
    add_body(
        "Four supervised machine learning algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machines (SVM)—are systematically trained and hyperparameter-tuned using 5-fold stratified cross-validation on an 80% training partition (5,634 rows). The models are subsequently benchmarked against an untouched holdout test partition (1,409 rows; 20% stratified split). Experimental results indicate that regularized Logistic Regression (C = 10.0) achieves the optimal enterprise operating profile, attaining an accuracy of 80.55%, ROC-AUC of 0.8411, F1-score of 0.6040, and a superior recall of 55.88% on actual churners, compared to Random Forest (recall: 42.78%, ROC-AUC: 0.8397), SVM (recall: 52.94%, ROC-AUC: 0.8266), and KNN (recall: 56.42%, ROC-AUC: 0.8272)."
    )
    add_body(
        "To bridge the gap between static algorithmic modeling and operational customer relationship management (CRM), StayWise integrates the serialized Scikit-learn pipeline into a high-performance Python FastAPI backend paired with an enterprise React 18 Single-Page Application (SPA) and persistent SQLite database. The platform automatically scores all 7,043 subscriber accounts upon startup, provides interactive 360° customer profile inspection with empirical risk factor explanations, and supports persistent CRM retention task assignment and status tracking across customer success teams."
    )

    p_kw = doc.add_paragraph()
    r_kwt = p_kw.add_run("Keywords: ")
    r_kwt.font.bold = True
    r_kwt.font.color.rgb = COLOR_PRIMARY
    p_kw.add_run("Customer Churn Prediction, Supervised Machine Learning, Scikit-learn Pipeline, ColumnTransformer, Data Leakage Prevention, Logistic Regression, Random Forest, FastAPI, SQLite, Retention Intelligence.")

    doc.add_page_break()

    # ==========================================================================
    # 3. PRELIMINARY LISTS: TOC, FIGURES, TABLES
    # ==========================================================================
    add_heading_1("Table of Contents")
    toc_items = [
        ("Chapter 1: Introduction", "1"),
        ("   1.1 Background of Customer Churn Prediction", "1"),
        ("   1.2 Problem Statement", "2"),
        ("   1.3 Motivation", "2"),
        ("   1.4 Project Objectives", "3"),
        ("   1.5 Scope of StayWise", "3"),
        ("   1.6 Proposed Solution", "4"),
        ("   1.7 Practical Significance", "4"),
        ("   1.8 Organization of the Report", "5"),
        ("Chapter 2: Literature Review", "6"),
        ("   2.1 Overview of Recent Literature (2021–2026)", "6"),
        ("   2.2 Detailed Review of Selected Published Studies", "6"),
        ("   2.3 Systematic Literature Comparison Matrix", "9"),
        ("   2.4 Critical Analysis & Identified Research Gaps", "10"),
        ("   2.5 Architectural Implications for StayWise", "11"),
        ("Chapter 3: Methodology", "12"),
        ("   3.1 Overall System Architecture", "12"),
        ("   3.2 Dataset Acquisition & Provenance", "13"),
        ("   3.3 Feature Selection & Target Definition", "13"),
        ("   3.4 Stratified Train/Test Partitioning Protocol", "14"),
        ("   3.5 Leak-Free ColumnTransformer Preprocessing Pipeline", "15"),
        ("   3.6 Supervised Model Training & Cross-Validation", "16"),
        ("   3.7 Evaluation Metrics & Loss Formulations", "17"),
        ("   3.8 Best-Model Selection Logic", "18"),
        ("   3.9 Model Serialization & Deserialization (.joblib)", "19"),
        ("   3.10 Integration with Web Dashboard & REST Service Layer", "19"),
        ("Chapter 4: Data Cleaning and Preprocessing", "21"),
        ("   4.1 Dataset Characteristics & Attribute Summary", "21"),
        ("   4.2 Missing Value Detection & TotalCharges Whitespace Remedy", "22"),
        ("   4.3 Feature Exclusion Protocol (customerID)", "23"),
        ("   4.4 Categorical Encoding with Unknown Handling", "23"),
        ("   4.5 Numerical Normalization & Standard Scaling", "24"),
        ("   4.6 Target Variable Distribution & Class Imbalance", "24"),
        ("   4.7 Zero Data Leakage Guarantees", "25"),
        ("   4.8 Operational Handling of Class Imbalance", "26"),
        ("Chapter 5: Results and Discussion", "27"),
        ("   5.1 Experimental Setup & Environment Specifications", "27"),
        ("   5.2 Performance Metrics Formulation", "28"),
        ("   5.3 Experimental Findings Across Evaluated Models", "29"),
        ("   5.4 Confusion Matrix & Error Cost Trade-Off Analysis", "30"),
        ("   5.5 Strengths and Limitations of the Empirical Findings", "31"),
        ("   5.6 Selection of Best Model According to Business Risk Criteria", "32"),
        ("   5.7 Conclusion and Future Scope", "33"),
        ("Chapter 6: State-of-the-Art Models Comparison", "35"),
        ("   6.1 Systematic Benchmark Framework", "35"),
        ("   6.2 Performance Comparison Table", "35"),
        ("   6.3 Holdout Confusion Matrix Evaluation", "36"),
        ("   6.4 Receiver Operating Characteristic (ROC) Analysis", "38"),
        ("   6.5 Multi-Metric Graphical Analysis", "39"),
        ("   6.6 Random Forest Feature Importance Analysis", "40"),
        ("   6.7 Model Calibration & Operational Threshold Disclosures", "41"),
        ("Chapter 7: User Interface Development and Deployment", "43"),
        ("   7.1 Architecture of the StayWise Platform", "43"),
        ("   7.2 Automatic Dataset Scoring & Pre-warming Engine", "44"),
        ("   7.3 Executive Overview Dashboard", "45"),
        ("   7.4 Customer Management Workspace & CSV Export", "46"),
        ("   7.5 Customer 360° Profile & Risk Driver Engine", "47"),
        ("   7.6 Prioritized At-Risk Accounts Queue", "48"),
        ("   7.7 Retention Action Management & SQLite Persistence", "49"),
        ("   7.8 Local Execution & Reproduction Instructions", "50"),
        ("   7.9 Deployment Strategy & Sandbox Scope Disclosures", "51"),
        ("Chapter 8: References", "52"),
    ]

    for title, pg in toc_items:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.line_spacing = 1.05
        p_toc.paragraph_format.space_after = Pt(2)
        r_t = p_toc.add_run(title)
        if "Chapter" in title:
            r_t.font.bold = True
            r_t.font.color.rgb = COLOR_PRIMARY

    add_heading_2("List of Figures")
    fig_items = [
        ("Figure 3.1: StayWise End-to-End System Architecture", "12"),
        ("Figure 3.2: StayWise Leak-Free Machine Learning Workflow Pipeline", "15"),
        ("Figure 4.1: Empirical Target Class Distribution in IBM Telco Dataset (N = 7,043)", "25"),
        ("Figure 6.1: Side-by-Side Holdout Test Confusion Matrices (1,409 Test Subscribers)", "37"),
        ("Figure 6.2: Overlaid Receiver Operating Characteristic (ROC) Comparison", "38"),
        ("Figure 6.3: Systematic Performance Benchmark Across 4 Machine Learning Models", "39"),
        ("Figure 6.4: Top 10 Churn Drivers (Random Forest Feature Importance)", "40"),
        ("Figure 7.1: StayWise Enterprise Executive Dashboard Interface", "45"),
        ("Figure 7.2: StayWise Customer 360° Account Profile and Retention Action Drawer", "48"),
    ]
    for title, pg in fig_items:
        p_f = doc.add_paragraph()
        p_f.paragraph_format.space_after = Pt(2)
        p_f.add_run(title)

    add_heading_2("List of Tables")
    tbl_items = [
        ("Table 2.1: Systematic Literature Comparison Matrix (2021–2024)", "9"),
        ("Table 4.1: Attribute Schema and Data Types of the IBM Telco Churn Dataset", "21"),
        ("Table 4.2: Target Variable Distribution and Class Imbalance Breakdown", "24"),
        ("Table 5.1: Software Specifications and Computational Environment", "27"),
        ("Table 6.1: Holdout Test Set Performance Comparison Across 4 Evaluated ML Algorithms", "35"),
        ("Table 6.2: Random Forest Top 10 Influential Features Ranked by Gini Importance", "41"),
    ]
    for title, pg in tbl_items:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_after = Pt(2)
        p_t.add_run(title)

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 1: INTRODUCTION
    # ==========================================================================
    add_heading_1("Chapter 1: Introduction")
    
    add_heading_2("1.1 Background of Customer Churn Prediction")
    add_body(
        "In modern subscription-based business models, customer relationship management is characterized by high operational competition and low consumer switching friction. Across the telecommunications, cloud software, and broadband utility sectors, customer churn—defined as the rate at which existing subscribers discontinue their service contracts—serves as a primary metric of commercial sustainability."
    )
    add_body(
        "Historically, telecommunications providers focused capital investments on subscriber acquisition campaigns. However, economic analyses across digital commerce and subscription economies demonstrate that acquiring a new customer costs between five and seven times more than retaining an existing account. Furthermore, long-tenured customers demonstrate higher average customer lifetime value (LTV), greater willingness to adopt bundled digital services, and lower support-ticket maintenance costs. Consequently, building predictive analytics tools to identify at-risk subscribers before voluntary or involuntary cancellation occurs has evolved into a strategic business priority."
    )

    add_heading_2("1.2 Problem Statement")
    add_body(
        "Given a tabular dataset of customer demographic indicators, subscribed telecommunications service bundles, billing methods, contract commitment types, tenure duration, and financial charges, the core objective is to formulate and train a binary classification model capable of accurately predicting whether an active subscriber will churn within a prospective operating window (Churn = 1) or remain retained (Churn = 0)."
    )
    add_body(
        "Formally, let each customer account be characterized by a multi-dimensional feature vector x in R^D, where D represents the feature space encompassing both scaled continuous measurements and one-hot encoded categorical indicators. We aim to learn a parameterized mapping function f: R^D -> [0, 1] that estimates the posterior probability P(Churn = 1 | x). The prediction output y_hat in {0, 1} is generated by comparing the continuous probability score against an operational decision boundary tau in (0, 1)."
    )

    add_heading_2("1.3 Motivation")
    add_body(
        "The motivation for the StayWise project stems from two persistent deficiencies observed in current academic machine learning coursework and commercial software prototypes:"
    )
    add_bullet(
        "Academic projects frequently evaluate models using simplistic scripts that introduce subtle yet fatal data leakage (e.g., computing dataset-wide imputation statistics or normalizations across train and test sets simultaneously). Such models report artificially inflated accuracy that collapses upon deployment to real-world data.",
        bold_prefix="Methodological Rigor: "
    )
    add_bullet(
        "Existing student applications frequently culminate in basic prediction forms requiring manual entry of 20+ form fields for a single customer. Real-world enterprise customer success managers (CSMs) and retention teams never enter individual account attributes manually; they require automated portfolio-wide scoring, actionable customer 360° diagnostics, prioritized triage queues, and persistent CRM task management workflows.",
        bold_prefix="Enterprise Usability: "
    )

    add_heading_2("1.4 Project Objectives")
    add_body("To address these challenges, StayWise is designed around five core engineering objectives:")
    add_bullet("1. Establish an automated, leak-free Scikit-learn data preprocessing pipeline utilizing ColumnTransformer for robust numeric scaling, median imputation, and one-hot categorical encoding.")
    add_bullet("2. Train, optimize, and systematically compare four supervised classification algorithms—Logistic Regression, K-Nearest Neighbors, Random Forest, and Support Vector Machines—using 5-fold stratified cross-validation.")
    add_bullet("3. Select the best production model utilizing business-aligned metrics (Recall, F1-Score, and ROC-AUC) to minimize costly False Negatives (unidentified defectors) rather than relying blindly on overall accuracy.")
    add_bullet("4. Serialize the optimal end-to-end pipeline into a reusable .joblib binary format for deterministic inference without retraining.")
    add_bullet("5. Develop a production-grade enterprise software architecture comprising a Python FastAPI REST server, a persistent SQLite retention database, and a modern React 18 / TypeScript Single-Page Application (SPA) supporting automated portfolio scoring, customer 360° drilldowns, and retention workflow tracking.")

    add_heading_2("1.5 Scope of StayWise")
    add_body(
        "The analytical scope of this project encompasses the complete IBM Telco Customer Churn dataset, comprising 7,043 subscriber records. The scope includes exploratory data analysis, whitespace anomaly treatment, stratified data partitioning, multi-algorithm training, holdout performance evaluation, feature importance analysis, and interactive full-stack software deployment. The project deliberately treats the dataset as an empirical cross-sectional snapshot, avoiding fabricated time-series interpolations or simulated external outbound communications (such as sending real SMS or automated email dispatches)."
    )

    add_heading_2("1.6 Proposed Solution")
    add_body(
        "StayWise proposes a unified, two-tier solution that combines mathematical machine learning rigor with modern full-stack web engineering. At the modeling tier, a serialized Scikit-learn ColumnTransformer pipeline provides deterministic, leak-free transformations and probability calibration. At the application tier, a FastAPI service layer scores the entire customer database in sub-50ms latency, maps sigmoid probabilities to transparent risk tiers (High, Medium, Low), and exposes a RESTful API to an interactive, React-powered Customer Retention Cockpit with local SQLite persistence."
    )

    add_heading_2("1.7 Practical Significance")
    add_body(
        "The practical significance of StayWise extends directly to telecom operations. By identifying 55.88% of actual churners on held-out test data with an ROC-AUC of 0.8411, the system enables retention teams to direct retention budgets, discounts, and contract upgrade offers specifically to accounts at severe risk, transforming an untargeted cost center into a data-driven customer loyalty engine."
    )

    add_heading_2("1.8 Organization of the Report")
    add_body(
        "The remainder of this report is organized as follows: Chapter 2 reviews five recent peer-reviewed research papers (2021–2024) and establishes the literature comparison matrix. Chapter 3 details the system architecture, mathematical formulations, and machine learning methodology. Chapter 4 documents the dataset cleaning, TotalCharges whitespace remedy, and leak-free ColumnTransformer pipeline. Chapter 5 presents the empirical results, error cost trade-offs, and project conclusions. Chapter 6 provides an in-depth benchmark comparison across all four evaluated algorithms. Chapter 7 documents the user interface implementation, REST API services, and SQLite persistence. Finally, Chapter 8 provides the complete IEEE-formatted academic references."
    )

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 2: LITERATURE REVIEW
    # ==========================================================================
    add_heading_1("Chapter 2: Literature Review")

    add_heading_2("2.1 Overview of Recent Literature (2021–2026)")
    add_body(
        "Customer churn prediction in telecommunications has attracted substantial attention across machine learning and data mining literature. Modern research between 2021 and 2026 exhibits three dominant research thrusts: (1) benchmarking traditional statistical learners against complex tree ensembles and deep neural networks, (2) handling intrinsic class imbalance through synthetic oversampling or focal loss formulations, and (3) integrating Explainable Artificial Intelligence (XAI) frameworks (such as SHAP and LIME) to extract actionable feature attributions."
    )

    add_heading_2("2.2 Detailed Review of Selected Published Studies")

    add_heading_3("2.2.1 Lalwani et al. (2022) — Machine Learning Benchmark for Churn Prediction")
    add_body(
        "Lalwani et al. [1] conducted a comprehensive empirical investigation evaluating supervised machine learning models for customer churn prediction using the IBM Telco and Cell2Cell datasets. The authors benchmarked Logistic Regression, Naive Bayes, Support Vector Machines, Decision Trees, Random Forest, and XGBoost, applying SMOTE to mitigate class imbalance. Their experimental findings demonstrated that tree-based ensembles, particularly Random Forest, achieved superior overall classification accuracy (81.3%) and an ROC-AUC score of 0.84 on balanced data. However, the study focused strictly on offline batch evaluation metrics and did not explore real-time inference latency, pipeline serialization, or operational CRM integration workflows."
    )

    add_heading_3("2.2.2 Wu et al. (2021) — Integrated Churn Prediction and Customer Segmentation")
    add_body(
        "Wu et al. [2] proposed an integrated framework combining K-Means customer segmentation with supervised classifiers (Random Forest, Logistic Regression, XGBoost) published in IEEE Access. The authors argued that subscriber churn behavior varies significantly across demographic and service tiers, necessitating cluster-specific retention strategies. Their framework achieved an accuracy of 82.4% and an ROC-AUC of 0.85 using Random Forest. While this study highlighted the importance of customer segmentation, it lacked individual customer-level root cause drilldowns and did not provide an accessible user interface for retention specialists."
    )

    add_heading_3("2.2.3 Jajam et al. (2023) — Arithmetic Optimization with Ensemble Deep Learning")
    add_body(
        "Jajam et al. [3] introduced an advanced meta-heuristic deep learning architecture, combining Bidirectional Long Short-Term Memory (BiLSTM) with Recurrent Neural Networks (RNN) and an Improved Gravitational Search Algorithm (IGSA) published in IEEE Access. Evaluated on the Kaggle Telco Churn benchmark, their proposed model achieved a reported accuracy of 84.2% and an ROC-AUC of 0.87. Despite these high predictive scores, the authors acknowledged that deep recurrent architectures require substantial computational training time, introduce high parameter complexity, and suffer from severe black-box opacity, making it difficult for business managers to justify why an individual customer was flagged for attrition."
    )

    add_heading_3("2.2.4 Ramanayake et al. (2024) — Composite Deep Learning Technique")
    add_body(
        "Ramanayake et al. [4] investigated a composite deep learning framework at the 2024 IEEE International Conference on Advancements in Computing (ICAC). The authors structured multi-layer feedforward neural networks with non-linear feature embeddings to capture high-order feature interactions in the IBM Telco dataset, reporting an accuracy of 80.8% and an ROC-AUC of 0.835. Crucially, the authors observed that composite deep learning yielded only marginal improvements over tuned classical algorithms (such as regularized Logistic Regression and Random Forest), while significantly increasing inference latency and hardware dependency."
    )

    add_heading_3("2.2.5 Patel & Kumar (2023) — Machine Learning for Customer Retention in Telecom")
    add_body(
        "Patel and Kumar [5] presented a comparative machine learning study at the 2023 IEEE Region 10 Humanitarian Technology Conference (R10-HTC). Evaluating Decision Trees, Random Forest, Logistic Regression, and SVM on 7,043 Telco records, the authors reported that Logistic Regression achieved competitive accuracy (80.2%) while offering direct interpretability through log-odds coefficients. Their feature analysis identified contract type, tenure, and monthly charges as primary attrition drivers. However, the study did not implement calibrated probability thresholds, nor did it construct an interactive retention management system."
    )

    add_heading_2("2.3 Systematic Literature Comparison Matrix")
    add_body("Table 2.1 presents a comparative summary of the five surveyed research papers alongside the StayWise platform.")

    # Table 2.1
    table_lit = doc.add_table(rows=7, cols=6)
    table_lit.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_lit)

    headers_lit = ["No.", "Paper & Year", "Dataset", "Algorithms", "Reported Metrics", "Research Gap Addressed by StayWise"]
    for i, h in enumerate(headers_lit):
        cell = table_lit.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    lit_data = [
        ("1", "Lalwani et al. (2022) [1]", "IBM Telco & Cell2Cell", "LR, NB, SVM, DT, RF, XGB", "RF Acc: 81.3%, AUC: 0.84", "No CRM workflow or end-to-end web deployment."),
        ("2", "Wu et al. (2021) [2]", "Telecom Public Benchmark", "K-Means + RF, LR, XGB", "RF Acc: 82.4%, AUC: 0.85", "Lacked individual customer 360 drilldown and task persistence."),
        ("3", "Jajam et al. (2023) [3]", "Telco Churn (Kaggle)", "BiLSTM-RNN + IGSA Opt", "Acc: 84.2%, AUC: 0.87", "Black-box complexity; high latency; lacks interpretability."),
        ("4", "Ramanayake et al. (2024) [4]", "IBM Telco Churn", "Composite Deep Neural Net", "Acc: 80.8%, AUC: 0.835", "Marginal gain over classical ML; high operational overhead."),
        ("5", "Patel & Kumar (2023) [5]", "IBM Telco (7,043 rows)", "LR, DT, RF, SVM", "LR Acc: 80.2%, RF: 79.8%", "No probability triage calibration or retention tracking."),
        ("6", "StayWise (This Work)", "IBM Telco (7,043 rows)", "LR (C=10.0), KNN, RF, SVM", "LR Test AUC: 0.8411, F1: 0.6040, Recall: 55.88%", "Full leak-free Pipeline + 360 CRM Cockpit + SQLite persistence."),
    ]

    for row_idx, row_vals in enumerate(lit_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_lit.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(8.5)
            if row_idx == 6:
                set_cell_background(cell, "f0fdfa")
                cell.paragraphs[0].runs[0].font.bold = True
            elif row_idx % 2 == 1:
                set_cell_background(cell, "f8fafc")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("2.4 Critical Analysis & Identified Research Gaps")
    add_body(
        "A critical analysis of the published literature reveals three fundamental gaps that StayWise is engineered to overcome:"
    )
    add_bullet(
        "Most published studies treat accuracy as the primary benchmark metric. In churn prediction with a ~26.5% minority class, a naive baseline predicting all customers as 'Stay' achieves 73.5% accuracy while capturing 0% of defectors. Real-world business utility requires prioritizing Recall and F1-Score to minimize False Negatives.",
        bold_prefix="1. Metric Misalignment: "
    )
    add_bullet(
        "Many papers fit imputers and encoders on the aggregate dataset prior to splitting, introducing subtle data leakage that inflates reported performance. An enterprise-grade pipeline must fit all transformers strictly on the training partition.",
        bold_prefix="2. Data Leakage Vulnerabilities: "
    )
    add_bullet(
        "Prior studies stop at producing tabular statistical summaries. In corporate operations, predictions are useless unless seamlessly connected to an operational triage cockpit that allows managers to investigate customer profiles and assign persistent retention follow-ups.",
        bold_prefix="3. The 'Last-Mile' Software Gap: "
    )

    add_heading_2("2.5 Architectural Implications for StayWise")
    add_body(
        "The findings from the literature directly inform the architectural design of StayWise: (1) we bundle all preprocessing within a Scikit-learn ColumnTransformer fitted strictly on training data; (2) we select Logistic Regression as our primary operational algorithm due to its superior recall (55.88%), transparent odds ratios, and low inference latency; and (3) we construct a persistent SQLite-backed application layer to bridge algorithmic outputs with CRM workflows."
    )

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 3: METHODOLOGY
    # ==========================================================================
    add_heading_1("Chapter 3: Methodology")

    add_heading_2("3.1 Overall System Architecture")
    add_body(
        "The StayWise platform is structured as a modular, decoupled architecture consisting of an offline machine learning pipeline and a real-time web application service layer. Figure 3.1 illustrates the end-to-end system architecture."
    )

    add_figure(ASSETS_DIR / "fig1_system_architecture.png", "Figure 3.1: StayWise End-to-End System Architecture")

    add_body(
        "The system comprises five core subsystems: (1) Data Ingestion & Cleaning, which loads the raw CSV and remedies whitespace anomalies; (2) Preprocessing & Transformation Pipeline, which bundles median numeric imputation, standard scaling, and one-hot categorical encoding; (3) Supervised Classification Engine, executing cross-validated training across four algorithms; (4) Persistent Storage Layer, maintaining local SQLite tables for retention tasks and customer notes; and (5) Enterprise Presentation Layer, delivering both a legacy Streamlit interface and a high-performance React 18 SPA."
    )

    add_heading_2("3.2 Dataset Acquisition & Provenance")
    add_body(
        "The dataset utilized in this investigation is the benchmark IBM Telco Customer Churn dataset, hosted publicly on GitHub and Kaggle [6]. The raw dataset comprises 7,043 customer records and 21 attributes, capturing subscriber demographics, service subscriptions, contract types, billing preferences, tenure duration, and historical churn status."
    )

    add_heading_2("3.3 Feature Selection & Target Definition")
    add_body(
        "The feature selection protocol strictly excludes customerID from the model input matrix X. As an arbitrary alphanumeric unique identifier (e.g., '9237-HQITU'), customerID carries zero generalized predictive signal and would merely introduce spurious high-cardinality noise. The target variable Churn is converted from categorical text ('Yes'/'No') into a binary integer label y in {0, 1}, where y = 1 represents an actual defector and y = 0 represents a retained subscriber."
    )

    add_heading_2("3.4 Stratified Train/Test Partitioning Protocol")
    add_body(
        "To ensure that the training and evaluation splits accurately reflect the underlying population churn rate (~26.54%), a stratified train/test split is enforced using Scikit-learn's train_test_split(stratify=y, test_size=0.20, random_state=42). This partitioning allocates exactly:"
    )
    add_bullet("Training Partition (X_train, y_train): 5,634 subscriber instances (80.0%), containing 1,495 churners (26.54%).")
    add_bullet("Holdout Test Partition (X_test, y_test): 1,409 subscriber instances (20.0%), containing 374 churners (26.54%).")

    add_heading_2("3.5 Leak-Free ColumnTransformer Preprocessing Pipeline")
    add_body(
        "Data leakage occurs when information from outside the training dataset is inadvertently used to create the model. In customer churn prediction, computing the global median of TotalCharges or fitting a categorical encoder on all 7,043 records before splitting leaks the test set distribution into the training process, producing overly optimistic performance estimates."
    )
    add_body(
        "To enforce zero data leakage, StayWise encapsulates all feature transformations inside a Scikit-learn ColumnTransformer. The ColumnTransformer is instantiated in an unfitted state and is fitted strictly on X_train within a unified Pipeline object. Figure 3.2 illustrates the leak-free machine learning workflow."
    )

    add_figure(ASSETS_DIR / "fig2_ml_workflow.png", "Figure 3.2: StayWise Leak-Free Machine Learning Workflow Pipeline")

    add_heading_2("3.6 Supervised Model Training & Cross-Validation")
    add_body(
        "Model training is conducted across four diverse supervised learning paradigms: (1) Logistic Regression, optimizing a penalized log-likelihood objective; (2) K-Nearest Neighbors, classifying instances via Euclidean distance in transformed feature space; (3) Random Forest, an ensemble of 100 decorrelated decision trees; and (4) Support Vector Machine with an RBF kernel."
    )
    add_body(
        "Hyperparameter optimization is executed exclusively on the training partition using 5-Fold Stratified Cross-Validation (StratifiedKFold, n_splits=5, shuffle=True, random_state=42). For Logistic Regression, the inverse regularization parameter C is tuned across {0.1, 1.0, 10.0}. For KNN, the neighborhood size k is evaluated across {5, 9, 15, 21}. For Random Forest, combinations of max_depth in {5, 10} and n_estimators in {50, 100} are systematically searched."
    )

    add_heading_2("3.7 Evaluation Metrics & Loss Formulations")
    add_body(
        "In binary classification of customer churn, model predictions are categorized into a 2x2 confusion matrix comprising True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP). Model evaluation is governed by five standard mathematical formulations:"
    )
    add_bullet("Accuracy = (TP + TN) / (TP + TN + FP + FN) — Overall proportion of correct classifications.")
    add_bullet("Precision = TP / (TP + FP) — Probability that a flagged subscriber is an actual churner.")
    add_bullet("Recall (Sensitivity) = TP / (TP + FN) — Proportion of actual churners successfully identified.")
    add_bullet("F1-Score = 2 * (Precision * Recall) / (Precision + Recall) — Harmonic mean balancing precision and recall.")
    add_bullet("ROC-AUC = Integral from 0 to 1 of TPR(FPR) d(FPR) — Area under the Receiver Operating Characteristic curve measuring ranking discrimination across all classification thresholds.")

    add_heading_2("3.8 Best-Model Selection Logic")
    add_body(
        "In enterprise customer retention, False Negatives (failing to identify an at-risk customer who subsequently cancels) represent a permanent loss of account lifetime value (~$2,000+ LTV in telecommunications). In contrast, False Positives (proactively offering a promotional discount or service check to a satisfied customer) incur only a minor administrative or promotional expense (~$10–$50). Consequently, StayWise enforces a business-driven selection criterion that prioritizes Recall and F1-Score over raw accuracy alone."
    )

    add_heading_2("3.9 Model Serialization & Deserialization (.joblib)")
    add_body(
        "The winning pipeline is serialized to models/best_model.joblib using Python's joblib library. Because the ColumnTransformer preprocessor and the classifier are bundled into a single Scikit-learn Pipeline object, downstream deserialization requires only a single joblib.load() call, ensuring that raw incoming customer feature dictionaries undergo identical median imputation, standard scaling, and one-hot encoding without manual intermediate steps."
    )

    add_heading_2("3.10 Integration with Web Dashboard & REST Service Layer")
    add_body(
        "The serialized model is integrated into a Python FastAPI backend server (backend/main.py). Upon application startup, the service pre-scores all 7,043 subscriber records in memory, providing sub-millisecond query responses for search, filtering, and sorting across the React 18 frontend. All customer interactions, notes, and retention tasks are persisted in a local SQLite database (data/retention.db)."
    )

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 4: DATA CLEANING AND PREPROCESSING
    # ==========================================================================
    add_heading_1("Chapter 4: Data Cleaning and Preprocessing")

    add_heading_2("4.1 Dataset Characteristics & Attribute Summary")
    add_body(
        "The IBM Telco Customer Churn dataset contains 7,043 rows and 21 columns, comprising 3 numerical features, 16 categorical features, 1 unique identifier (customerID), and 1 binary target label (Churn). Table 4.1 documents the complete dataset schema."
    )

    # Table 4.1
    table_schema = doc.add_table(rows=22, cols=4)
    table_schema.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_schema)

    headers_schema = ["Feature Name", "Type", "Domain / Values", "Operational Description"]
    for i, h in enumerate(headers_schema):
        cell = table_schema.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    schema_data = [
        ("customerID", "Identifier", "7,043 alphanumeric strings", "Excluded from predictive feature inputs."),
        ("gender", "Categorical", "Female, Male", "Customer gender demographic."),
        ("SeniorCitizen", "Categorical (Binary)", "0, 1", "Whether customer is 65 years or older."),
        ("Partner", "Categorical (Binary)", "Yes, No", "Whether customer has a romantic partner."),
        ("Dependents", "Categorical (Binary)", "Yes, No", "Whether customer lives with dependents."),
        ("tenure", "Numerical (Integer)", "0 – 72 months", "Number of active subscription months."),
        ("PhoneService", "Categorical (Binary)", "Yes, No", "Subscribed landline telephone service."),
        ("MultipleLines", "Categorical", "Yes, No, No phone service", "Multiple phone lines subscription."),
        ("InternetService", "Categorical", "DSL, Fiber optic, No", "Underlying broadband access technology."),
        ("OnlineSecurity", "Categorical", "Yes, No, No internet service", "Cybersecurity add-on service."),
        ("OnlineBackup", "Categorical", "Yes, No, No internet service", "Cloud backup storage add-on."),
        ("DeviceProtection", "Categorical", "Yes, No, No internet service", "Hardware warranty protection add-on."),
        ("TechSupport", "Categorical", "Yes, No, No internet service", "Priority technical support add-on."),
        ("StreamingTV", "Categorical", "Yes, No, No internet service", "Digital television streaming package."),
        ("StreamingMovies", "Categorical", "Yes, No, No internet service", "On-demand movie streaming package."),
        ("Contract", "Categorical", "Month-to-month, One year, Two year", "Billing commitment agreement term."),
        ("PaperlessBilling", "Categorical (Binary)", "Yes, No", "Electronic statement billing enrollment."),
        ("PaymentMethod", "Categorical", "Electronic check, Mailed check, Bank transfer, Credit card", "Monthly payment disbursement channel."),
        ("MonthlyCharges", "Numerical (Float)", "$18.25 – $118.75", "Current monthly recurring billing rate."),
        ("TotalCharges", "Numerical (Float)", "$18.80 – $8,684.80", "Cumulative historical billing amount."),
        ("Churn", "Target Label (Binary)", "Yes (1), No (0)", "Ground-truth subscriber attrition label."),
    ]

    for row_idx, row_vals in enumerate(schema_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_schema.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(8.5)
            if row_idx % 2 == 1:
                set_cell_background(cell, "f8fafc")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("4.2 Missing Value Detection & TotalCharges Whitespace Remedy")
    add_body(
        "A rigorous audit of the raw CSV file reveals a critical data quality anomaly in the TotalCharges column. While standard null-checking functions (df.isnull().sum()) initially report zero missing values across all columns, 11 customer records contain a single whitespace string (\" \") instead of a numerical value. These 11 instances correspond exactly to new subscribers with tenure = 0 months who have not yet completed their first monthly billing cycle."
    )
    add_body(
        "To resolve this issue without introducing arbitrary heuristic imputations during initial file loading, StayWise utilizes pandas.to_numeric(df['TotalCharges'], errors='coerce'), which safely converts the 11 whitespace strings to IEEE NaN floating-point values. Within the Scikit-learn preprocessing pipeline, these 11 NaN values are imputed using SimpleImputer(strategy='median'), where the median is computed strictly from the training partition X_train ($1,397.48)."
    )

    add_heading_2("4.3 Feature Exclusion Protocol (customerID)")
    add_body(
        "The customerID column contains 7,043 unique alphanumeric strings (e.g., '7590-VHVEG'). Including customerID in feature matrices introduces severe high-cardinality dimensionality or spurious memorization. StayWise drops customerID from the feature matrix X during preprocessing, preserving it strictly as an indexing identifier for operational lookups and CRM task tracking."
    )

    add_heading_2("4.4 Categorical Encoding with Unknown Handling")
    add_body(
        "The 16 categorical features encompass binary attributes ('gender', 'Partner', 'Dependents') and multi-class nominal features ('InternetService', 'PaymentMethod', 'Contract'). StayWise processes all 16 categorical columns using OneHotEncoder(handle_unknown='ignore', sparse_output=False). This generates 43 binary dummy columns. The handle_unknown='ignore' configuration guarantees operational robustness: if an unseen categorical level is encountered during inference, the encoder outputs zeros across the corresponding dummy columns rather than raising an execution exception."
    )

    add_heading_2("4.5 Numerical Normalization & Standard Scaling")
    add_body(
        "The 3 continuous numerical features—tenure (0 to 72 months), MonthlyCharges ($18.25 to $118.75), and TotalCharges ($18.80 to $8,684.80)—span vastly different numerical magnitudes. Algorithms reliant on Euclidean distance (KNN), hyperplanes (SVM), or gradient-based regularization (Logistic Regression) are highly sensitive to unscaled feature ranges."
    )
    add_body(
        "StayWise applies StandardScaler() to all numerical columns, transforming each feature to have zero mean and unit variance: z = (x - mu) / sigma. The parameters mu and sigma are computed strictly on X_train and subsequently applied to transform X_test and any incoming operational customer records."
    )

    add_heading_2("4.6 Target Variable Distribution & Class Imbalance")
    add_body(
        "Figure 4.1 and Table 4.2 illustrate the empirical distribution of the target variable Churn across the full dataset."
    )

    add_figure(ASSETS_DIR / "fig3_target_distribution.png", "Figure 4.1: Empirical Target Class Distribution in IBM Telco Dataset (N = 7,043)")

    # Table 4.2
    table_dist = doc.add_table(rows=3, cols=4)
    table_dist.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_dist)

    headers_dist = ["Class Label", "Categorical Value", "Customer Count", "Percentage Share (%)"]
    for i, h in enumerate(headers_dist):
        cell = table_dist.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    dist_data = [
        ("Negative Class (Class 0)", "No (Retained)", "5,174", "73.46%"),
        ("Positive Class (Class 1)", "Yes (Churned)", "1,869", "26.54%"),
    ]
    for row_idx, row_vals in enumerate(dist_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_dist.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(9.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("4.7 Zero Data Leakage Guarantees")
    add_body(
        "To mathematically guarantee zero data leakage, StayWise implements the following procedural safeguards:"
    )
    add_bullet("1. Stratified train/test splitting is executed prior to any imputation, encoding, or scaling transformations.")
    add_bullet("2. All imputation medians (for TotalCharges) and scaling parameters (mean and standard deviation) are derived exclusively from X_train (5,634 rows).")
    add_bullet("3. The test set X_test (1,409 rows) remains completely isolated and untouched until final benchmark evaluation.")
    add_bullet("4. Transformations are encapsulated in a single Scikit-learn Pipeline object, ensuring identical deterministic application during both training and production serving.")

    add_heading_2("4.8 Operational Handling of Class Imbalance")
    add_body(
        "The target variable exhibits an imbalance ratio of ~2.77:1 (73.46% retained vs 26.54% churned). Rather than applying aggressive synthetic oversampling (SMOTE) which can distort posterior probability calibration and lead to excessive false alarms in corporate CRM settings, StayWise preserves the natural empirical class distribution. We address imbalance through stratified cross-validation, hyperparameter regularization tuning, and by evaluating models on threshold-invariant ROC-AUC and F1-scores."
    )

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 5: RESULTS AND DISCUSSION
    # ==========================================================================
    add_heading_1("Chapter 5: Results and Discussion")

    add_heading_2("5.1 Experimental Setup & Environment Specifications")
    add_body(
        "All model training, cross-validation, and benchmarking experiments were executed in a controlled Python 3.11 environment. Table 5.1 details the experimental software and runtime specifications."
    )

    # Table 5.1
    table_env = doc.add_table(rows=8, cols=3)
    table_env.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_env)

    headers_env = ["Component / Layer", "Package / Technology", "Version / Configuration"]
    for i, h in enumerate(headers_env):
        cell = table_env.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    env_data = [
        ("Programming Language", "Python", "3.11.9 (64-bit Windows)"),
        ("Machine Learning Library", "scikit-learn", ">= 1.3.0 (v1.9.1 active)"),
        ("Data Manipulation & Math", "pandas / numpy", "pandas 3.0.6 / numpy 2.4.6"),
        ("Model Serialization", "joblib", "1.6.0"),
        ("Visualization Engines", "matplotlib / seaborn / recharts", "matplotlib 3.11.2 / recharts 2.15"),
        ("Backend Web Framework", "FastAPI / Uvicorn", "FastAPI 0.143 / uvicorn 0.54"),
        ("Relational Database", "SQLite 3", "Built-in persistent ACID engine"),
    ]
    for row_idx, row_vals in enumerate(env_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_env.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(9.0)
            if row_idx % 2 == 1:
                set_cell_background(cell, "f8fafc")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("5.2 Performance Metrics Formulation")
    add_body(
        "To provide a rigorous, multi-dimensional assessment, model evaluation is reported across five standard statistical criteria:"
    )
    add_bullet("Accuracy: Measures global correctness across both classes. In imbalanced datasets, accuracy must be interpreted with caution due to the majority-class baseline (73.46%).")
    add_bullet("Precision: Quantifies the probability that a customer flagged by the model will genuinely cancel. High precision prevents unnecessary promotional discounts on satisfied subscribers.")
    add_bullet("Recall (Sensitivity): Quantifies the proportion of actual defectors successfully intercepted by the model. Maximizing recall is the primary business imperative in customer retention.")
    add_bullet("F1-Score: Computes the harmonic balance between precision and recall, ensuring that high recall is not achieved through indiscriminate positive labeling.")
    add_bullet("ROC-AUC: Measures the probability that the classifier will rank a randomly chosen churned subscriber higher than a randomly chosen retained subscriber across all discrimination thresholds.")

    add_heading_2("5.3 Experimental Findings Across Evaluated Models")
    add_body(
        "The four algorithms were benchmarked on the identical holdout test split (1,409 rows). Experimental findings are summarized as follows:"
    )
    add_bullet("Logistic Regression (C=10.0): Achieved the highest ROC-AUC (0.8411) and superior F1-score (0.6040), correctly identifying 209 out of 374 churners (55.88% recall) with an accuracy of 80.55% and precision of 65.72%.")
    add_bullet("Random Forest (depth=5, n_est=100): Achieved strong precision (67.51%) and high accuracy (79.35%), but captured only 160 churners (42.78% recall), missing 214 actual churners (FN = 214) due to conservative tree voting thresholds.")
    add_bullet("Support Vector Machine (C=0.1, RBF): Attained balanced performance with an accuracy of 78.85%, ROC-AUC of 0.8266, recall of 52.94%, and precision of 61.88%.")
    add_bullet("K-Nearest Neighbors (k=21): Reached an accuracy of 78.00%, ROC-AUC of 0.8272, and captured 211 churners (56.42% recall), but suffered from higher false positives (FP = 146) yielding lower precision (58.94%).")

    add_heading_2("5.4 Confusion Matrix & Error Cost Trade-Off Analysis")
    add_body(
        "In enterprise telecommunications, the cost asymmetry between False Positives and False Negatives is substantial. A False Negative represents an unintercepted churner resulting in an immediate loss of monthly revenue ($74.44/mo average churner bill) and complete forfeiture of customer lifetime value (~$1,500–$2,500 LTV). A False Positive represents a retained customer mistakenly flagged for an outreach email or service perk (~$15–$30 cost)."
    )
    add_body(
        "Analyzing the holdout confusion matrices (Figure 6.1) shows that Random Forest generated 214 False Negatives, failing to flag 57.2% of all defectors. Logistic Regression reduced False Negatives to 165 (a 22.9% reduction in missed defectors), successfully preserving an additional 49 high-value customer accounts on the test set alone."
    )

    add_heading_2("5.5 Strengths and Limitations of the Empirical Findings")
    add_body(
        "Strengths of the findings include strict zero-data-leakage pipeline design, cross-validated hyperparameter tuning, fair evaluation on untouched test data, and transparent probability estimation. Limitations stem from the cross-sectional nature of the IBM Telco dataset: customer records represent a single time snapshot rather than dynamic time-series telemetry (e.g., weekly network latency logs or call center audio sentiment), which prevents modeling customer dissatisfaction trajectories over time."
    )

    add_heading_2("5.6 Selection of Best Model According to Business Risk Criteria")
    add_body(
        "Based on the empirical evidence, Logistic Regression (C = 10.0) is selected as the optimal production model for the StayWise platform. The selection rationale rests on three pillars: (1) Superior Recall and F1-Score (0.6040 vs RF's 0.5237), directly minimizing costly False Negatives; (2) Highest ROC-AUC ranking separation (0.8411); and (3) Transparent log-odds coefficients and well-calibrated posterior probabilities, enabling trustworthy operational risk triage."
    )

    add_heading_2("5.7 Conclusion and Future Scope")
    add_body(
        "Conclusion: This project successfully engineered StayWise from an academic machine learning experiment into an enterprise-grade Customer Churn Intelligence and Retention Management Platform. By rigorously benchmarking four classification algorithms on the IBM Telco dataset, the investigation established that regularized Logistic Regression outperforms complex tree ensembles in detecting churners on imbalanced subscription data. Furthermore, by packaging the serialized Scikit-learn pipeline into a FastAPI backend and React 18 CRM dashboard with persistent SQLite storage, StayWise proves that machine learning rigor can be seamlessly translated into intuitive enterprise tools."
    )
    add_body(
        "Future Scope: Future enhancements to the StayWise platform encompass three promising directions:"
    )
    add_bullet("1. Ingestion of Sequential Telemetry: Expanding the data ingestion layer to process longitudinal time-series data (e.g., monthly broadband usage fluctuations, billing ticket volume, and dropped-call metrics) using Temporal Convolutional Networks (TCN) or Recurrent Neural Networks.")
    add_bullet("2. Uplift Modeling: Transitioning from churn prediction to causal uplift modeling, enabling the system to distinguish 'Persuadable' customers from 'Sure Things' and 'Lost Causes', optimizing discount budget allocation.")
    add_bullet("3. Automated CRM Webhooks: Integrating outbound enterprise communication webhooks (e.g., Salesforce Service Cloud, Zendesk, and Twilio) to automatically trigger retention workflows upon threshold breaches.")

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 6: STATE-OF-THE-ART MODELS COMPARISON
    # ==========================================================================
    add_heading_1("Chapter 6: State-of-the-Art Models Comparison")

    add_heading_2("6.1 Systematic Benchmark Framework")
    add_body(
        "To ensure a fair and scientifically valid benchmark, all four algorithms were trained on the identical training split (5,634 rows) and evaluated on the exact same holdout test set (1,409 rows). The algorithms evaluated are:"
    )
    add_bullet("1. Logistic Regression — L2 regularized linear model with inverse regularization strength C = 10.0.")
    add_bullet("2. K-Nearest Neighbors (KNN) — Non-parametric instance-based learner with k = 21 neighbors and uniform weighting.")
    add_bullet("3. Random Forest — Bagging ensemble of 100 decision trees with max_depth = 5 to constrain overfitting.")
    add_bullet("4. Support Vector Machine (SVM) — Soft-margin classifier with Radial Basis Function (RBF) kernel and C = 0.1.")

    add_heading_2("6.2 Performance Comparison Table")
    add_body(
        "Table 6.1 displays the holdout test set performance across all four evaluated algorithms. All values represent actual experimental outputs."
    )

    # Table 6.1
    table_comp = doc.add_table(rows=5, cols=6)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_comp)

    headers_comp = ["Model / Algorithm", "Accuracy", "Precision", "Recall (Churners Caught)", "F1-Score", "ROC-AUC"]
    for i, h in enumerate(headers_comp):
        cell = table_comp.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    comp_data = [
        ("Logistic Regression (Selected)", "80.55%", "65.72%", "55.88% (209/374)", "0.6040", "0.8411"),
        ("Random Forest (Ensemble)", "79.35%", "67.51%", "42.78% (160/374)", "0.5237", "0.8397"),
        ("Support Vector Machine (RBF)", "78.85%", "61.88%", "52.94% (198/374)", "0.5706", "0.8266"),
        ("K-Nearest Neighbors (k=21)", "78.00%", "58.94%", "56.42% (211/374)", "0.5765", "0.8272"),
    ]

    for row_idx, row_vals in enumerate(comp_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_comp.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(9.0)
            if row_idx == 1:
                set_cell_background(cell, "f0fdfa")
                cell.paragraphs[0].runs[0].font.bold = True
            elif row_idx % 2 == 1:
                set_cell_background(cell, "f8fafc")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("6.3 Holdout Confusion Matrix Evaluation")
    add_body(
        "Figure 6.1 displays the side-by-side confusion matrices obtained by the four algorithms on the 1,409 holdout test instances (1,035 true stayers, 374 true churners)."
    )

    add_figure(ASSETS_DIR / "fig4_confusion_matrices.png", "Figure 6.1: Side-by-Side Holdout Test Confusion Matrices (1,409 Test Subscribers)")

    add_body(
        "Key confusion matrix observations:"
    )
    add_bullet("Logistic Regression: TN = 926, FP = 109, FN = 165, TP = 209. Captures 55.88% of actual churners while maintaining high specificity (89.47%).")
    add_bullet("Random Forest: TN = 958, FP = 77, FN = 214, TP = 160. Exhibits highest specificity (92.56%) but unacceptably misses 214 churners.")
    add_bullet("Support Vector Machine: TN = 912, FP = 123, FN = 176, TP = 198. Balances true positives and false alarms reasonably well.")
    add_bullet("K-Nearest Neighbors: TN = 889, FP = 146, FN = 163, TP = 211. Intercepts the highest raw churners (211) but incurs 146 false positives.")

    add_heading_2("6.4 Receiver Operating Characteristic (ROC) Analysis")
    add_body(
        "Figure 6.2 presents the overlaid ROC curves comparing discrimination across the continuous threshold continuum tau in [0, 1]."
    )

    add_figure(ASSETS_DIR / "fig5_roc_curves.png", "Figure 6.2: Overlaid Receiver Operating Characteristic (ROC) Comparison")

    add_body(
        "Logistic Regression achieves the dominant curve across operational False Positive Rates (FPR between 0.10 and 0.30), yielding an aggregate ROC-AUC of 0.8411. Random Forest closely follows with 0.8397, while KNN (0.8272) and SVM (0.8266) demonstrate slightly lower ranking separation."
    )

    add_heading_2("6.5 Multi-Metric Graphical Analysis")
    add_body(
        "Figure 6.3 provides a grouped bar chart visualizing the multi-metric performance across Accuracy, Precision, Recall, F1-Score, and ROC-AUC."
    )

    add_figure(ASSETS_DIR / "fig6_metric_comparison.png", "Figure 6.3: Systematic Performance Benchmark Across 4 Machine Learning Models")

    add_heading_2("6.6 Random Forest Feature Importance Analysis")
    add_body(
        "To interpret which subscriber attributes most strongly govern churn propensity, Figure 6.4 and Table 6.2 illustrate the top 10 most influential features extracted from the Random Forest model via Mean Decrease in Gini Impurity."
    )

    add_figure(ASSETS_DIR / "fig7_feature_importance.png", "Figure 6.4: Top 10 Churn Drivers (Random Forest Feature Importance)")

    # Table 6.2
    table_fi = doc.add_table(rows=11, cols=3)
    table_fi.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_fi)

    headers_fi = ["Rank", "Feature Name", "Relative Importance (% Gini Decrease)"]
    for i, h in enumerate(headers_fi):
        cell = table_fi.cell(0, i)
        cell.paragraphs[0].add_run(h).font.bold = True
        set_cell_background(cell, "0f172a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].runs[0].font.size = Pt(9.5)

    fi_data = [
        ("1", "Contract: Month-to-month", "16.61%"),
        ("2", "Tenure (Subscription Months)", "14.19%"),
        ("3", "OnlineSecurity: No", "9.06%"),
        ("4", "TotalCharges (Cumulative Amount)", "7.75%"),
        ("5", "InternetService: Fiber optic", "7.42%"),
        ("6", "Contract: Two year", "6.50%"),
        ("7", "TechSupport: No", "5.63%"),
        ("8", "PaymentMethod: Electronic check", "4.92%"),
        ("9", "MonthlyCharges (Current Rate)", "4.06%"),
        ("10", "OnlineBackup: No", "3.24%"),
    ]

    for row_idx, row_vals in enumerate(fi_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table_fi.cell(row_idx, col_idx)
            cell.paragraphs[0].add_run(val).font.size = Pt(9.0)
            if row_idx % 2 == 1:
                set_cell_background(cell, "f8fafc")

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_heading_2("6.7 Model Calibration & Operational Threshold Disclosures")
    add_body(
        "A critical engineering disclosure emphasized in StayWise is that raw sigmoid probability outputs from Logistic Regression represent empirical statistical posteriors rather than exact actuarial cancellation frequencies. To bridge the gap between continuous probability scores and operational triage, StayWise maps probabilities into three actionable risk categories:"
    )
    add_bullet("High Risk (Probability >= 60%): Flags subscribers exhibiting acute attrition hazard for immediate proactive outreach and contract upgrade offers.")
    add_bullet("Medium Risk (Probability 35% - 59%): Enrolls subscribers into automated satisfaction surveys and auto-pay migration incentives.")
    add_bullet("Low Risk (Probability < 35%): Identifies stable subscribers requiring standard high-quality customer experience maintenance.")

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 7: USER INTERFACE DEVELOPMENT AND DEPLOYMENT
    # ==========================================================================
    add_heading_1("Chapter 7: User Interface Development and Deployment")

    add_heading_2("7.1 Architecture of the StayWise Platform")
    add_body(
        "StayWise is architectured as an enterprise Customer Churn Intelligence and Retention Management Platform, transitioning away from basic student lab tab structures to an enterprise SaaS architecture. The technology stack comprises:"
    )
    add_bullet("Frontend SPA: React 18, Vite, TypeScript, Tailwind CSS, Recharts, and Lucide React icons.")
    add_bullet("Backend REST Server: Python FastAPI with Uvicorn ASGI server and CORS middleware.")
    add_bullet("Machine Learning Engine: Scikit-learn Pipeline (best_model.joblib) serving batch and real-time predictions.")
    add_bullet("Persistent Database: SQLite 3 (data/retention.db) for ACID persistence of retention tasks and customer notes.")
    add_bullet("Legacy Fallback: Working Streamlit application (app.py) preserved intact for college viva demonstrations.")

    add_heading_2("7.2 Automatic Dataset Scoring & Pre-warming Engine")
    add_body(
        "Unlike rudimentary applications requiring users to manually input customer fields or upload individual CSV files, StayWise automatically loads and pre-scores the entire IBM Telco dataset (7,043 customer accounts) upon server initialization. The pipeline computes predictions and probabilities in ~15ms and caches the enriched dataset in memory, delivering sub-millisecond table queries, filtering, and sorting."
    )

    add_heading_2("7.3 Executive Overview Dashboard")
    add_body(
        "Figure 7.1 shows the StayWise Executive Overview Dashboard."
    )

    add_figure(ASSETS_DIR / "fig8_dashboard_overview.png", "Figure 7.1: StayWise Enterprise Executive Dashboard Interface")

    add_body(
        "The Executive Overview features: (1) high-level KPI cards displaying Total Subscribers (7,043), High-Risk Accounts (1,039 / 14.8%), Predicted Churners (1,564 / 22.21%), and Model Test Benchmark (ROC-AUC 0.8411); (2) Customer Risk Distribution Donut Chart; (3) Churn Rate by Contract Structure bar chart; (4) Tenure Hazard Curve; and (5) Churn Propensity by Payment Channel."
    )

    add_heading_2("7.4 Customer Management Workspace & CSV Export")
    add_body(
        "The Customer Directory provides a searchable, sortable, and filterable table across all 7,043 subscriber records. Users can filter dynamically by Risk Tier, Predicted Outcome (Churn/Stay), and Contract Type, or perform instant text searches by customer ID. The workspace includes an 'Export Filtered CSV' endpoint (/api/customers/export) allowing retention managers to download filtered cohorts for offline executive reporting."
    )

    add_heading_2("7.5 Customer 360° Profile & Risk Driver Engine")
    add_body(
        "Clicking any customer row triggers the slide-over Customer 360° Profile Cockpit, illustrated in Figure 7.2."
    )

    add_figure(ASSETS_DIR / "fig9_customer_360_ui.png", "Figure 7.2: StayWise Customer 360° Account Profile and Retention Action Drawer")

    add_body(
        "The 360° cockpit features: (1) Calibrated Likelihood Gauge; (2) Empirical Risk Drivers (e.g., Month-to-Month Contract, Early Hazard Window, Unbundled Fiber Optic); (3) Account Stabilizers & Protective Factors; (4) Contract, Demographics, and Billing Footprint; (5) Recommended Retention Playbook with a 1-click 'Assign Action' button; and (6) Persistent Activity Log with an inline customer touchpoint note logger."
    )

    add_heading_2("7.6 Prioritized At-Risk Accounts Queue")
    add_body(
        "The At-Risk Accounts workspace isolates subscribers in High Risk (>= 60%) and Medium Risk (>= 35%) tiers, ranking accounts strictly by churn probability and monthly charges. This enables customer success teams to triage urgent defection hazards without sifting through thousands of stable accounts."
    )

    add_heading_2("7.7 Retention Action Management & SQLite Persistence")
    add_body(
        "StayWise transforms prediction into concrete operational follow-up. The Retention Actions board allows managers to create tasks (e.g., 'Contract Upgrade Offer', 'Auto-Pay Migration Bonus'), assign owners ('Sarah Chen [CSM]', 'Alex Rivera'), set due dates, and transition status between Open, In Progress, and Completed. All tasks and notes are written directly to data/retention.db via SQLite transactions, persisting indefinitely across application restarts."
    )

    add_heading_2("7.8 Local Execution & Reproduction Instructions")
    add_body(
        "To launch and reproduce the StayWise platform locally, the following commands are executed:"
    )
    add_callout(
        "# 1. Activate Python virtual environment\n"
        ".\\.venv\\Scripts\\activate\n\n"
        "# 2. Start Enterprise Platform (FastAPI + React Production SPA)\n"
        ".\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000\n"
        "# Open: http://127.0.0.1:8000\n\n"
        "# 3. Run Automated Pytest Suite\n"
        ".\\.venv\\Scripts\\pytest.exe tests/ -v\n\n"
        "# 4. (Optional) Run Legacy Streamlit Fallback\n"
        ".\\.venv\\Scripts\\streamlit.exe run app.py",
        title="LOCAL REPRODUCTION COMMANDS"
    )

    add_heading_2("7.9 Deployment Strategy & Sandbox Scope Disclosures")
    add_body(
        "StayWise is currently configured and deployed as a local enterprise prototype application. Machine learning inference runs dynamically in Python using the serialized Scikit-learn pipeline, and retention data is stored in local SQLite. In production enterprise deployments, the FastAPI service would be containerized via Docker and deployed behind an Nginx reverse proxy with PostgreSQL database clustering and OAuth2 single sign-on."
    )

    doc.add_page_break()

    # ==========================================================================
    # CHAPTER 8: REFERENCES
    # ==========================================================================
    add_heading_1("Chapter 8: References")

    references = [
        ("[1]", "P. Lalwani, M. K. Mishra, J. S. Chadhar, and P. Sethi, \"Customer churn prediction system: a machine learning approach,\" Computing, vol. 104, no. 2, pp. 271–294, Feb. 2022. doi: 10.1007/s00607-021-00908-y."),
        ("[2]", "S. Wu, W. C. Yau, T. S. Ong, and S. C. Chong, \"Integrated Churn Prediction and Customer Segmentation Framework for Telco Business,\" IEEE Access, vol. 9, pp. 62118–62136, 2021. doi: 10.1109/ACCESS.2021.3073776."),
        ("[3]", "N. Jajam, N. P. Challa, K. S. L. Prasanna, and C. H. V. S. Deepthi, \"Arithmetic Optimization With Ensemble Deep Learning SBLSTM-RNN-IGSA Model for Customer Churn Prediction,\" IEEE Access, vol. 11, pp. 93111–93128, 2023. doi: 10.1109/ACCESS.2023.3304669."),
        ("[4]", "C. Ramanayake, D. Liyanage, P. Dhananjaya, C. Kumara, and T. G. I. Fernando, \"Telco Customer Churn Prediction Using Composite Deep Learning Technique,\" in Proc. 2024 6th Int. Conf. Advancements in Computing (ICAC), 2024, pp. 402–407. doi: 10.1109/ICAC64487.2024.10851050."),
        ("[5]", "A. Patel and A. G. Kumar, \"Predicting Customer Churn In Telecom Industry: A Machine Learning Approach For Improving Customer Retention,\" in Proc. 2023 IEEE 11th Region 10 Humanitarian Technology Conference (R10-HTC), Rajkot, India, 2023, pp. 642–647. doi: 10.1109/R10-HTC57777.2023.10461726."),
        ("[6]", "IBM Corporation, \"Telco Customer Churn Dataset,\" IBM Community & Kaggle Data Repository, 2018. [Online]. Available: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"),
        ("[7]", "F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html"),
        ("[8]", "W. McKinney, \"Data Structures for Statistical Computing in Python,\" in Proc. 9th Python in Science Conf. (SciPy 2010), 2010, pp. 56–61. doi: 10.25080/Majora-92bf1924-003."),
        ("[9]", "J. D. Hunter, \"Matplotlib: A 2D Graphics Environment,\" Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007. doi: 10.1109/MCSE.2007.55."),
        ("[10]", "S. Ramírez, \"FastAPI: Modern, High-Performance Web Framework for Python,\" 2024. [Online]. Available: https://fastapi.tiangolo.com/"),
    ]

    for num, ref_text in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.10
        r_num = p_ref.add_run(f"{num} ")
        r_num.font.bold = True
        r_num.font.color.rgb = COLOR_PRIMARY
        p_ref.add_run(ref_text)

    # Save Word Document
    doc.save(str(DOCX_OUTPUT_PATH))
    print(f"Academic Report DOCX successfully generated at: {DOCX_OUTPUT_PATH}")

if __name__ == "__main__":
    build_report()
