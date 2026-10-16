# 📡 StayWise — Customer Churn Prediction ML Lab

**An End-to-End Educational Machine Learning Laboratory Project for Academic Practical Exams and Viva Voce Defense.**

---

## 1. Project Overview

StayWise predicts customer churn using the **IBM Telco Customer Churn** dataset (7,043 subscribers, 20 predictive features, 26.54% churn rate).

The project implements a complete, leak-free machine learning workflow comparing **four foundational algorithms**:
1. **Logistic Regression** (Linear baseline with probability calibration and odds interpretability)
2. **K-Nearest Neighbors (KNN)** (Instance-based lazy learner highlighting distance metrics and feature scaling)
3. **Random Forest** (Ensemble tree-based method highlighting bagging, non-linearity, and feature importance)
4. **Support Vector Machine (SVM)** (Maximum-margin linear classifier highlighting hyperplane optimization and support vectors)

---

## 2. End-to-End Architecture

```
Raw IBM Telco Dataset (7,043 rows, 21 columns)
  ├── 1. Data Cleaning: Fix 11 whitespace strings in TotalCharges -> Impute Median
  ├── 2. Stratified Train/Test Split (80% Train: 5,634 | 20% Test: 1,409)
  ├── 3. Scikit-Learn ColumnTransformer (Zero Data Leakage)
  │     ├── Numerical (tenure, MonthlyCharges, TotalCharges) -> Median Imputer + StandardScaler
  │     └── Categorical (16 features) -> Frequent Imputer + OneHotEncoder(ignore_unknown)
  ├── 4. 5-Fold Stratified Cross-Validation on Training Set (Hyperparameter Tuning)
  │     ├── Logistic Regression (Tuned C in [0.1, 1.0, 10.0])
  │     ├── K-Nearest Neighbors (Tuned odd k in [5, 9, 15, 21])
  │     ├── Random Forest (Tuned n_estimators in [50, 100], max_depth in [5, 10])
  │     └── Support Vector Machine (Tuned C in [0.1, 1.0, 10.0], linear kernel)
  ├── 5. Benchmark Evaluation on Untouched Holdout Test Set (Accuracy, Precision, Recall, F1, ROC-AUC)
  ├── 6. Pipeline Serialization with Joblib (`models/best_model.joblib`)
  └── 7. Web Application Serving via Streamlit (`app.py`)
```

---

## 3. Empirical Results (Holdout Test Set: 1,409 rows)

All four models were evaluated on the **exact same untouched test set** (`X_test`, `y_test`, stratified 20% split) to guarantee an authentic and fair benchmark:

| Algorithm | Hyperparameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Selected) | $C=10.0$, $\text{max\_iter}=1000$ | **80.55%** | 65.72% | **55.88%** | **0.6040** | **0.8411** |
| **Random Forest** | $n=100$, $\text{max\_depth}=5$ | 79.35% | **67.51%** | 42.78% | 0.5237 | 0.8397 |
| **Support Vector Machine (SVM)** | $\text{kernel}=\text{linear}$, $C=0.1$ | 78.85% | 61.88% | 52.94% | 0.5706 | 0.8266 |
| **K-Nearest Neighbors (KNN)** | $k=21$, Euclidean distance | 78.00% | 58.94% | 56.42% | 0.5765 | 0.8272 |

### Model Selection Rationale
In customer churn intelligence, **Recall is prioritized over raw Accuracy** because failing to identify a customer who churns (False Negative) costs the company recurring revenue and customer lifetime value. A False Positive (sending an unnecessary retention offer) has minimal cost.

**Logistic Regression** was selected as the final deployment model because:
1. It achieves the **highest Recall (55.88%)** among all high-precision models.
2. It achieves the **highest F1-score (0.6040)** and **highest ROC-AUC (0.8411)**.
3. It provides well-calibrated posterior probabilities and is computationally lightweight.

---

## 4. Support Vector Machine (SVM) — Viva Voce Guide

### Q1: What is Support Vector Machine (SVM)?
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression. In binary classification, it seeks an optimal decision boundary called a **hyperplane** that separates two classes with the **maximum possible margin** (distance between the boundary and the closest data points).

### Q2: How does SVM predict customer churn?
1. Each customer is represented as a point in high-dimensional feature space $\mathbf{x} = [x_1, x_2, \dots, x_p]$ (after scaling numerical values and one-hot encoding categories).
2. The linear SVM evaluates the linear decision function:
   $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
3. If $f(\mathbf{x}) \ge 0$, the customer lies on the positive side of the boundary $\rightarrow$ predicted as **Churn (1)**.
4. If $f(\mathbf{x}) < 0$, the customer lies on the negative side $\rightarrow$ predicted as **Stay / No Churn (0)**.
5. With `probability=True`, Platt scaling fits a logistic sigmoid over the decision values to output a calibrated probability between 0.0 and 1.0.

### Q3: What do the Hyperplane, Margin, and Support Vectors mean?
- **Hyperplane:** The linear decision surface $\mathbf{w}^T \mathbf{x} + b = 0$ dividing churners from non-churners. In a 2D plane it is a line; in 3D space it is a flat plane; in higher dimensions it is an $(n-1)$-dimensional affine hyperplane.
- **Margin:** The geometric distance between the hyperplane and the nearest training observations from either class. The margin width is given by $\frac{2}{\|\mathbf{w}\|}$. SVM maximizes this margin to reduce generalization error.
- **Support Vectors:** The specific data points located right on the margin boundaries ($\mathbf{w}^T \mathbf{x} + b = \pm 1$) or violating them. If any training points *other* than the support vectors are shifted or removed, the hyperplane position remains completely unchanged.

### Q4: How does SVM differ from Logistic Regression, KNN, and Random Forest?
| Dimension | Logistic Regression | K-Nearest Neighbors (KNN) | Random Forest | Support Vector Machine (SVM) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Parametric linear probabilistic model | Non-parametric lazy instance learner | Non-parametric ensemble of decision trees | Parametric/non-parametric maximum-margin classifier |
| **Loss Function** | Log-Loss (cross-entropy over all data points) | No loss function (stores training data) | Impurity reduction (Gini/Entropy) | Hinge Loss ($\max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$) |
| **Focus of Optimization** | Fits entire distribution probabilities | Looks at local neighborhood of $k$ points | Averages diverse tree splits | Focuses only on critical boundary points (support vectors) |
| **Sensitivity to Scaling** | Sensitive to regularization scale | **Extremely sensitive** (distance-based) | Invariant (monotonic split thresholds) | **Extremely sensitive** (margin distance) |
| **Prediction Speed** | Fast ($O(p)$) | Slow ($O(N \cdot p)$ search) | Moderate ($O(T \cdot \text{depth})$) | Fast ($O(p)$ for linear) |

### Q5: Why did SVM perform moderately (78.85% Accuracy, 0.8266 ROC-AUC) on this dataset?
1. **Class Overlap & Noise:** Real-world customer churn data has significant overlap—customers with very similar contracts and tenure might churn or stay due to unobserved factors. Hinge loss heavily penalizes points violating the margin, making linear SVM slightly more rigid than Logistic Regression's smooth log-odds curve.
2. **Class Imbalance:** The Telco dataset is imbalanced (73.5% non-churn vs. 26.5% churn). Standard SVM margin optimization can be slightly pulled toward the majority class unless cost-sensitive class weights are tuned.
3. **Linearity vs Non-linearity:** The linear kernel restricts the boundary to a flat plane. While it avoids overfitting, it cannot capture complex non-linear combinations like Random Forest without a non-linear kernel (e.g. RBF), which would add substantial training time.

---

## 5. Setup and Execution Instructions

### 1. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Data Preprocessing Verification
```powershell
.\.venv\Scripts\python.exe src/preprocess.py
```

### 3. Train All Models (LR, KNN, RF, SVM)
```powershell
.\.venv\Scripts\python.exe src/train.py
```

### 4. Run Comprehensive Model Evaluation & Plots
```powershell
.\.venv\Scripts\python.exe src/evaluate.py
```

### 5. Run Automated Test Suite (Pytest)
```powershell
.\.venv\Scripts\pytest.exe -v tests/
```

### 6. Launch Interactive Streamlit Web Application
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 6. Top Viva Voce Questions & Answers

1. **Why is data leakage dangerous and how did you prevent it?**  
   *Answer:* Data leakage occurs when test set knowledge influences training (e.g., fitting the scaler or imputer on the full dataset before splitting). We prevented it by performing a stratified 80/20 train/test split first, and only fitting `ColumnTransformer` on `X_train`.

2. **Why is feature scaling mandatory for KNN and SVM?**  
   *Answer:* Both KNN (Euclidean distance) and SVM (maximum geometric margin) calculate spatial distances between points. Without `StandardScaler`, high-magnitude features like `TotalCharges` ($0 - 8,684) dwarf features like `SeniorCitizen` ($0 - 1$), causing distances to be dominated by a single variable.

3. **What is the difference between $C$ in Logistic Regression and $C$ in SVM?**  
   *Answer:* In Scikit-Learn, both represent inverse regularization strength. In SVM, $C$ balances margin width against margin violations (slack variables $\xi_i$). A smaller $C$ ($0.1$) permits more violations for a wider margin (more regularization), while a large $C$ ($10.0$) enforces a narrower margin with stricter classification penalties.

4. **Why do we serialize a Scikit-Learn `Pipeline` rather than just the classifier?**  
   *Answer:* Raw classifiers accept only numerical arrays of matching dimension. The `Pipeline` bundles preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) with the classifier, enabling raw dictionaries or DataFrames from the Streamlit UI to be passed directly without manual re-transformation.
