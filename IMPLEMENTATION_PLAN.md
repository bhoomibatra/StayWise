# StayWise — Implementation Plan & Phase-by-Phase Roadmap
**Project:** StayWise — Customer Churn Prediction ML Lab Project  
**Target Level:** Undergraduate ML Lab / College Practical Exam & Viva  
**Current Status:** Planning Complete (Awaiting Phase 1 Approval)

---

## 1. Project Overview & Philosophy

StayWise is an educational, end-to-end Machine Learning laboratory project designed to predict telecom customer churn using the standard **IBM Telco Customer Churn** dataset.

The core goal of StayWise is to demonstrate the fundamental Machine Learning workflow in an authentic, transparent, and beginner-friendly manner:
1. **Problem Formulation:** Binary classification (Will the customer churn? Yes / No).
2. **Exploration & Preprocessing:** Data hygiene, handling hidden missing values, categorical encoding, feature scaling, and preventing data leakage.
3. **Model Training & Comparison:** Training and tuning three foundational algorithms:
   - **Logistic Regression** (Linear baseline with probability calibration and odds interpretability)
   - **K-Nearest Neighbors (KNN)** (Instance-based non-parametric classifier highlighting distance metrics and feature scaling)
   - **Random Forest** (Ensemble tree-based method highlighting bagging, non-linearity, and feature importance)
4. **Rigorous Evaluation:** Comparing models fairly on a shared holdout test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Operationalization:** Exporting a unified preprocessing + model pipeline using `joblib` and serving it via a clean **Streamlit** web interface for interactive single-customer predictions and lab demonstrations.

---

## 2. Minimalist Architecture & Folder Structure

To keep the codebase readable, maintainable, and easy to explain in a viva examination, the project avoids over-engineering, microservices, databases, or redundant abstractions.

```
StayWise/
├── data/
│   └── telco_churn.csv              # IBM Telco Customer Churn raw CSV
├── models/
│   ├── best_model.joblib            # Exported Scikit-Learn Pipeline (fitted preprocessor + classifier)
│   ├── metrics.json                 # Consolidated comparison metrics & best model metadata
│   └── evaluation_plots.png         # Saved Confusion Matrices & ROC Curves
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset acquisition, loading, cleaning (TotalCharges fix)
│   ├── preprocess.py                # ColumnTransformer & Train/Test splitting (zero leakage)
│   ├── train.py                     # Training logic & CV tuning for LR, KNN, and Random Forest
│   └── evaluate.py                  # Evaluation metrics calculation, confusion matrices, ROC curves
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py           # Pytest unit tests for preprocessing integrity
│   └── test_model.py                # Pytest unit tests for training, predictions, and pipeline reload
├── app.py                           # Single-file Streamlit web application
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Project documentation, results summary, and viva guide
└── IMPLEMENTATION_PLAN.md           # This master execution document
```

---

## 3. Technology Stack

- **Language:** Python 3.10+
- **Data Manipulation & Math:** `pandas`, `numpy`
- **Machine Learning Library:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn` (for static/eval plots)
- **Model Serialization:** `joblib`
- **Interactive UI:** `streamlit`
- **Automated Testing:** `pytest`

*Note on simplicity:* No FastAPI, React, Docker, cloud services, or SQL databases are used. All logic is pure Python with Scikit-learn pipelines.

---

## 4. Phase Execution Protocol

Execution will strictly proceed **phase by phase**:
1. You (the user) will instruct the agent: *"Execute Phase X"*.
2. The agent will verify all prerequisites for Phase X.
3. The agent will implement only the files and tasks associated with Phase X.
4. The agent will run the designated verification commands.
5. If any errors occur, the agent will fix them immediately.
6. The agent will update the completion checklist in this document.
7. The agent will explain the key ML concepts and decisions, then pause and wait for your approval before starting the next phase.

---

## Phase 1 — Environment Setup and Dataset Exploration

### 1. Objective
Establish a clean virtual environment, install pinned dependencies, download the raw IBM Telco Customer Churn dataset into `data/telco_churn.csv`, and create a data exploration script to analyze dataset shape, column types, target class distribution (`Churn`), and detect anomalies (such as whitespace strings in `TotalCharges`).

### 2. Prerequisites
- Python 3.10+ installed on the host machine.
- Terminal / PowerShell access.

### 3. Tasks
1. Create `requirements.txt` containing only the necessary packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`, and `pytest`.
2. Create and activate a local virtual environment (`.venv`).
3. Install dependencies from `requirements.txt`.
4. Create the `data/` directory.
5. Create `src/data_loader.py` with functions to:
   - Download the raw Telco Churn CSV from the standard public raw URL (`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`) or load directly if already placed in `data/telco_churn.csv`.
   - Inspect summary statistics: row count (7,043), column count (21), data types, missing values, and churn rate (~26.5% churn).
6. Create an exploratory verification command to print dataset summary and identify the `TotalCharges` space issue.

### 4. Files
- `requirements.txt` (Create)
- `src/__init__.py` (Create)
- `src/data_loader.py` (Create)
- `data/telco_churn.csv` (Acquired)

### 5. Implementation Guidance
- The raw Telco dataset from IBM contains 7,043 rows and 21 columns.
- The `TotalCharges` column has object dtype because 11 rows with `tenure == 0` have whitespace spaces (`" "`) instead of numerical values.
- In `data_loader.py`, write `load_raw_data(data_path)` which loads the CSV, converts `TotalCharges` spaces to `NaN`, and logs the count of nulls.
- Offline Fallback: If downloading over network fails in a college lab environment, place `telco_churn.csv` directly into `data/`.
- Keep the script clean: do not write complex classes; use simple, readable Python functions with docstrings.

### 6. Commands
```bash
# 1. Create and activate virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run data loader exploration
python -c "from src.data_loader import load_raw_data, explore_data; df = load_raw_data(); explore_data(df)"
```

### 7. Verification
- Virtual environment is active and all libraries import without error.
- `data/telco_churn.csv` exists and has exactly 7,043 rows and 21 columns.
- Exploration output clearly shows:
  - Total records: 7,043
  - Churn distribution: No (~73.5%), Yes (~26.5%)
  - Detection of 11 whitespace values in `TotalCharges`

### 8. Expected Outcome
A working Python environment and a downloaded dataset with documented exploration findings, confirming the class imbalance and column data types.

### 9. Common Mistakes & Solutions
- **Mistake:** Forgetting that `TotalCharges` has string spaces `" "` instead of standard `NaN`, which causes later mathematical operations or modeling to crash with `ValueError: could not convert string to float: ' '`.
  - *Fix:* Use `pd.to_numeric(df['TotalCharges'], errors='coerce')`.
- **Mistake:** Using global Python rather than virtual environment.
  - *Fix:* Verify `( .venv )` prefix in terminal or check `sys.executable`.

### 10. Completion Checklist
- [x] `requirements.txt` created with pinned versions
- [x] Virtual environment `.venv` created and dependencies installed
- [x] `data/telco_churn.csv` downloaded and verified
- [x] `src/data_loader.py` implemented and verified
- [x] Target column `Churn` class balance documented (~26.5% positive)

### 11. Learning Objectives (Viva Preparation)
- **What is class imbalance?** In Telco Churn, ~73.5% of customers stay and ~26.5% leave. A naive model that always predicts "No" achieves 73.5% accuracy, making accuracy alone a misleading metric.
- **Why is `customerID` not a feature?** It is a unique identifier with high cardinality that conveys zero generalizable predictive pattern. Leaving it in causes overfitting.

---

## Phase 2 — Data Preprocessing

### 1. Objective
Build a robust, leak-free Scikit-Learn preprocessing pipeline (`ColumnTransformer`) that handles missing values, encodes categorical variables, standardizes numerical features, drops `customerID`, and splits data into stratified train/test sets.

### 2. Prerequisites
- Phase 1 completed (`data/telco_churn.csv` loaded and explored).

### 3. Tasks
1. Create `src/preprocess.py`.
2. Define feature lists:
   - Target: `Churn` (mapped to binary: `No` -> 0, `Yes` -> 1).
   - Ignored column: `customerID`.
   - Numerical columns: `tenure`, `MonthlyCharges`, `TotalCharges`.
   - Categorical columns (binary & multi-class nominal): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
3. Construct a Scikit-Learn `ColumnTransformer`:
   - Numerical Pipeline: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - Categorical Pipeline: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
4. Implement `prepare_data(data_path, test_size=0.20, random_state=42)` that returns `X_train, X_test, y_train, y_test, preprocessor`.
5. Ensure train/test split uses `stratify=y` so class proportions are strictly preserved in both sets.

### 4. Files
- `src/preprocess.py` (Create)

### 5. Implementation Guidance
- **Data Leakage Rule:** Never fit the imputer or scaler on the full dataset before splitting. Preprocessors must be fitted *only* on the training split, or bundled into a Scikit-learn `Pipeline`.
- Using `handle_unknown='ignore'` in `OneHotEncoder` prevents crashes when novel categories appear during inference in Streamlit.
- The 11 rows where `TotalCharges` is missing (due to `tenure == 0`) will be imputed cleanly using median training values.

### 6. Commands
```bash
python -c "from src.preprocess import prepare_data; X_train, X_test, y_train, y_test, preprocessor = prepare_data(); print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}'); print(f'Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}')"
```

### 7. Verification
- `X_train` contains exactly 80% (5,634 rows) and `X_test` contains 20% (1,409 rows).
- `customerID` is excluded from feature matrices.
- `y_train` and `y_test` churn rates both match ~0.2654 (verified stratification).
- Preprocessor can transform `X_train` without any NaN outputs.

### 8. Expected Outcome
A clean, reusable preprocessing module providing `X_train`, `X_test`, `y_train`, `y_test`, and a pre-configured Scikit-Learn `ColumnTransformer`.

### 9. Common Mistakes & Solutions
- **Mistake:** Calling `fit_transform` on the entire dataset and then running `train_test_split`. This leaks information (mean/std of test data) into the model.
  - *Fix:* Always perform `train_test_split` first, then fit on `X_train` only, or use Scikit-Learn `Pipeline`.
- **Mistake:** Non-stratified split resulting in uneven churn distribution in test set.
  - *Fix:* Explicitly set `stratify=y` in `train_test_split`.

### 10. Completion Checklist
- [x] `src/preprocess.py` implemented
- [x] `TotalCharges` correctly parsed to numeric with imputation
- [x] `customerID` dropped from feature set
- [x] `ColumnTransformer` assembled with `StandardScaler` and `OneHotEncoder`
- [x] Stratified 80/20 split verified without data leakage

### 11. Learning Objectives (Viva Preparation)
- **What is Data Leakage?** When information from outside the training dataset (such as the mean of the test set) is used to create or tune the model. It causes overly optimistic validation scores that fail in production.
- **Why scale features?** Algorithms calculating geometric distances (KNN) or gradient-based loss functions with regularization (Logistic Regression) are sensitive to feature scales. If `TotalCharges` ranges from 0 to 8,000 and `tenure` from 0 to 72, the unscaled distance will be dominated entirely by `TotalCharges`.

---

## Phase 3 — Logistic Regression

### 1. Objective
Implement, train, tune, and evaluate **Logistic Regression** as the foundational linear classification baseline using cross-validation on the training set, and evaluate predictions and probability estimates on the test set.

### 2. Prerequisites
- Phase 2 completed (`src/preprocess.py` available and verified).

### 3. Tasks
1. Create `src/train.py`.
2. Construct a Scikit-Learn `Pipeline` pairing the `ColumnTransformer` preprocessor with `LogisticRegression(max_iter=1000, random_state=42)`.
3. Perform 5-fold Stratified Cross-Validation on the training set to evaluate baseline ROC-AUC and F1-score.
4. Experiment with the regularization parameter `C` using a compact grid: `C in [0.1, 1.0, 10.0]` with `GridSearchCV(scoring='roc_auc', cv=5)`.
5. Fit the best Logistic Regression pipeline on the full `X_train`, `y_train`.
6. Predict classes (`predict`) and probabilities (`predict_proba`) on `X_test`.
7. Output baseline evaluation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 4. Files
- `src/train.py` (Create)

### 5. Implementation Guidance
- Logistic Regression calculates the log-odds: $\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$.
- The sigmoid function maps real numbers to probabilities in range $[0, 1]$.
- Default decision threshold is 0.5.
- Set `class_weight=None` or compare with `class_weight='balanced'` to see the effect on Recall.

### 6. Commands
```bash
python -c "from src.train import train_logistic_regression; lr_pipeline, metrics = train_logistic_regression(); print('Logistic Regression Metrics:', metrics)"
```

### 7. Verification
- Script executes without convergence warnings (`max_iter=1000` ensures convergence).
- Cross-validation identifies optimal `C`.
- Test set metrics are computed and printed:
  - Accuracy: ~80%
  - ROC-AUC: ~0.84
- Predictions and predicted probabilities are valid arrays between 0.0 and 1.0.

### 8. Expected Outcome
A trained Scikit-learn Logistic Regression pipeline with documented 5-fold CV score and holdout test set performance.

### 9. Common Mistakes & Solutions
- **Mistake:** Convergence warning `ConvergenceWarning: lbfgs failed to converge`.
  - *Fix:* Increase `max_iter` from default 100 to 1000 and ensure numerical features are scaled.
- **Mistake:** Evaluating `GridSearchCV` on the test set.
  - *Fix:* Always supply `X_train`, `y_train` to `GridSearchCV`. Never feed `X_test` into hyperparameter search.

### 10. Completion Checklist
- [x] Logistic Regression pipeline integrated with preprocessor
- [x] 5-fold Stratified CV performed on training set
- [x] Regularization hyperparameter `C` tuned
- [x] Test set predictions and probabilities generated
- [x] Test metrics logged (Accuracy, Precision, Recall, F1, ROC-AUC)

### 11. Learning Objectives (Viva Preparation)
- **How does Logistic Regression make a prediction?** It fits a linear combination of input features, passes the result through the logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to yield a probability $P(Y=1|X)$, and classifies as 1 if $P \ge 0.5$.
- **What is the effect of the regularization parameter `C`?** `C` is the inverse of regularization strength ($C = 1/\lambda$). Smaller values specify stronger regularization, penalizing large weights and preventing overfitting.

---

## Phase 4 — K-Nearest Neighbors (KNN)

### 1. Objective
Implement, train, tune, and evaluate **K-Nearest Neighbors (KNN)**, exploring the effect of neighbor count $k$ via cross-validation, and demonstrating the critical importance of feature scaling and distance metrics.

### 2. Prerequisites
- Phase 2 and Phase 3 completed.

### 3. Tasks
1. Extend `src/train.py` with `train_knn()`.
2. Construct a pipeline bundling `preprocessor` (which includes `StandardScaler`) with `KNeighborsClassifier()`.
3. Perform a grid search over odd values $k \in [5, 9, 15, 21]$ using 5-fold Stratified Cross-Validation on training data (4 candidates $\times$ 5 folds = 20 fast fits).
4. Fit the best KNN model on the full training set.
5. Generate test set predictions and class probabilities on `X_test`.
6. Evaluate and record Accuracy, Precision, Recall, F1, and ROC-AUC.
7. Include a simple helper function `compare_knn_scaling()` demonstrating that omitting feature scaling reduces ROC-AUC/Recall because large numeric values dominate distance calculation.

### 4. Files
- `src/train.py` (Modify to add KNN functions)

### 5. Implementation Guidance
- KNN is an *instance-based, lazy learner*: it stores training instances and predicts based on majority vote among the $k$ nearest neighbors in Euclidean feature space.
- Odd values of $k$ ($5, 9, 15, 21$) avoid tie votes in binary classification.
- Small $k$ (e.g. $k=3$) creates a complex decision boundary sensitive to noise; larger $k$ ($k=15-21$) creates a smoother, more robust boundary.

### 6. Commands
```bash
python -c "from src.train import train_knn; knn_pipeline, metrics = train_knn(); print('KNN Best Metrics:', metrics)"
```

### 7. Verification
- 5-fold CV successfully tests all specified $k$ values and selects the best candidate (typically $k \approx 15-25$).
- Pipeline evaluates on `X_test` without error.
- Verified that KNN runs on scaled features and achieves test ROC-AUC ~0.80+.

### 8. Expected Outcome
A tuned Scikit-learn KNN pipeline with cross-validated $k$ selection, test set evaluation metrics, and a practical demonstration of why feature scaling is mandatory for distance-based models.

### 9. Common Mistakes & Solutions
- **Mistake:** Even $k$ values in binary classification causing tie-breaks.
  - *Fix:* Use odd values for $k$ (`[5, 9, 15, 21]`).
- **Mistake:** Attempting to run KNN on unscaled one-hot encoded and numerical data together without `StandardScaler`.
  - *Fix:* Ensure the pipeline includes `StandardScaler()` for continuous features and proper one-hot encoding for categorical features.

### 10. Completion Checklist
- [x] KNN pipeline built with preprocessing
- [x] Grid search over odd $k$ values using 5-fold CV
- [x] Optimal $k$ selected based on validation performance
- [x] Model evaluated on `X_test` with metrics recorded
- [x] Impact of feature scaling on distance documented for viva

### 11. Learning Objectives (Viva Preparation)
- **Why is KNN called a "lazy learner"?** Unlike Logistic Regression, KNN has no explicit training phase where model parameters or weights are learned; it stores the training data and performs computations at query time.
- **Why is feature scaling essential for KNN?** KNN uses distance metrics (e.g. Euclidean distance $\sqrt{\sum (x_i - y_i)^2}$). If one feature has scale $0-8000$ and another $0-1$, the larger feature dominates distance calculations, rendering all other features irrelevant.

---

## Phase 5 — Random Forest

### 1. Objective
Implement, train, tune, and evaluate a **Random Forest Classifier**, exploring ensemble learning principles (bagging and feature randomness), tuning key hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`), and extracting feature importance.

### 2. Prerequisites
- Phases 2, 3, and 4 completed.

### 3. Tasks
1. Extend `src/train.py` with `train_random_forest()`.
2. Construct a pipeline bundling `preprocessor` with `RandomForestClassifier(random_state=42)`.
3. Perform 5-fold Stratified Cross-Validation tuning over a concise, beginner-friendly grid (4 total combinations $\times$ 5 folds = 20 fits, executes in seconds):
   - `n_estimators`: `[50, 100]` (number of decision trees in the forest)
   - `max_depth`: `[5, 10]` (tree depth limit to prevent overfitting)
4. Fit the best Random Forest pipeline on `X_train`, `y_train`.
5. Predict test labels and class probabilities on `X_test`.
6. Extract and rank the top 10 most influential features using `pipeline.named_steps['preprocessor'].get_feature_names_out()` and `classifier.feature_importances_`.
7. Compute and record test metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.

### 4. Files
- `src/train.py` (Modify to add Random Forest functions)

### 5. Implementation Guidance
- Random Forest is an ensemble of decision trees trained via **bootstrap aggregating (bagging)** with random feature subsets selected at each node split.
- Setting `max_depth` (e.g., 5 vs 10) directly shows how depth constraints prevent tree overfitting.
- Keeping the grid to 4 combinations ensures fast execution (< 15 seconds on any laptop) while fully demonstrating hyperparameter search.

### 6. Commands
```bash
python -c "from src.train import train_random_forest; rf_pipeline, metrics, top_features = train_random_forest(); print('Random Forest Metrics:', metrics); print('Top 5 Features:', top_features[:5])"
```

### 7. Verification
- Grid search completes in reasonable time (< 1 minute).
- Model achieves test ROC-AUC ~0.84+ and competitive accuracy (~79-81%).
- Feature importances are properly matched with one-hot encoded feature column names and logged. Top features should intuitively include `Contract`, `tenure`, `TotalCharges`, and `TechSupport`.

### 8. Expected Outcome
A tuned Random Forest model with CV evaluation, holdout test performance metrics, and a ranked list of top feature importances.

### 9. Common Mistakes & Solutions
- **Mistake:** Overfitting by allowing unbounded tree depth (`max_depth=None`, `min_samples_split=2`) resulting in 100% train accuracy but poor generalization on test data.
  - *Fix:* Constrain `max_depth` and `min_samples_split` during cross-validation.
- **Mistake:** Mismatching feature importance indices with raw column names after One-Hot Encoding.
  - *Fix:* Use `preprocessor.get_feature_names_out()` to correctly label encoded features.

### 10. Completion Checklist
- [x] Random Forest pipeline built with preprocessing
- [x] 5-fold CV grid search over `n_estimators` and `max_depth`
- [x] Best model selected and evaluated on holdout test set
- [x] Top 10 feature importances extracted with correct feature names
- [x] Training vs. test performance inspected for overfitting

### 11. Learning Objectives (Viva Preparation)
- **What is the difference between Bagging and Boosting?** Bagging (Bootstrap Aggregation, used in Random Forest) trains multiple trees in parallel on random bootstrap samples of data and averages their predictions to reduce variance. Boosting trains trees sequentially, each focusing on correcting errors made by previous trees to reduce bias.
- **Why does Random Forest reduce overfitting compared to a single Decision Tree?** A single decision tree has high variance and tends to overfit noisy data. By combining predictions from an ensemble of diverse trees trained on different data subsets and feature splits, the variance is reduced without increasing bias.

---

## Phase 6 — Model Comparison

### 1. Objective
Systematically compare all three algorithms on the exact same holdout test set (`X_test`, `y_test`) across standard evaluation metrics, generate comparative visualizations (Confusion Matrices and ROC Curves), and select the recommended deployment model based on empirical evidence.

### 2. Prerequisites
- Phases 2 through 5 completed (LR, KNN, and Random Forest trained and evaluated).

### 3. Tasks
1. Create `src/evaluate.py`.
2. Implement functions to:
   - Compute a comprehensive metrics table comparing all 3 models: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Generate and save side-by-side Confusion Matrices.
   - Plot overlaid ROC Curves with corresponding AUC scores on a single plot.
3. Save comparative plots and metrics to `models/metrics.json` and `models/evaluation_plots.png`.
4. Provide a written analysis explaining:
   - Which model is superior overall.
   - The trade-off between Precision and Recall in customer churn (e.g., False Negatives: failing to identify an at-risk customer who leaves vs. False Positives: offering unneeded discounts to a loyal customer).
   - Selection of the primary model to deploy in the Streamlit application.

### 4. Files
- `src/evaluate.py` (Create)
- `models/metrics.json` (Artifact created)
- `models/evaluation_plots.png` (Artifact created)

### 5. Implementation Guidance
- Keep evaluation strictly on `X_test` (1,409 rows) for all three models to ensure fair comparison.
- Use `sklearn.metrics` for `classification_report`, `confusion_matrix`, `roc_curve`, and `roc_auc_score`.
- Use `matplotlib` to render clean, readable charts that can be saved directly to disk and rendered in the Streamlit UI.

### 6. Commands
```bash
python -c "from src.evaluate import compare_all_models; summary_df = compare_all_models(); print(summary_df)"
```

### 7. Verification
- `models/metrics.json` contains valid metric entries for Logistic Regression, KNN, and Random Forest.
- `models/evaluation_plots.png` is generated, showing 3 clear confusion matrices and 3 overlaid ROC curves.
- Evaluation output matches expected patterns:
  - Logistic Regression & Random Forest achieve highest ROC-AUC (~0.84).
  - KNN performs slightly lower (~0.80 - 0.81).
  - Clear rationale is documented for model selection.

### 8. Expected Outcome
A unified metrics comparison table, saved evaluation charts, and a clear, evidence-based selection of the best-performing model.

### 9. Common Mistakes & Solutions
- **Mistake:** Comparing models evaluated on different splits or cross-validation folds.
  - *Fix:* Evaluate all final fitted pipelines on the identical holdout `X_test`, `y_test`.
- **Mistake:** Choosing a model based solely on raw Accuracy in an imbalanced dataset.
  - *Fix:* Emphasize F1-score and ROC-AUC, explaining why Recall is critical in churn prediction.

### 10. Completion Checklist
- [ ] `src/evaluate.py` implemented
- [ ] Metrics table compiled across all three models
- [ ] Confusion matrices plotted and saved
- [ ] ROC curves plotted with AUC scores
- [ ] Trade-off between Precision and Recall documented for churn context
- [ ] Best model selected based on empirical metrics

### 11. Learning Objectives (Viva Preparation)
- **Why is ROC-AUC better than Accuracy for imbalanced data?** Accuracy rewards predicting the majority class regardless of ability to identify minority cases. ROC-AUC evaluates the model's ability to rank positive instances higher than negative instances across all possible decision thresholds, making it invariant to class distribution.
- **In customer churn, is Precision or Recall more important?** Usually **Recall** is prioritized because the business cost of a False Negative (losing a customer worth thousands in lifetime value) is typically much higher than the cost of a False Positive (spending a small retention incentive or email on a customer who wasn't going to leave anyway).

---

## Phase 7 — Saving and Reloading the Model

### 1. Objective
Serialize the entire Scikit-Learn pipeline (preprocessing + selected model) into a single artifact using `joblib`, and build verification checks ensuring the saved artifact can be reloaded in a fresh session and produce identical predictions on raw customer inputs without manual transformation.

### 2. Prerequisites
- Phase 6 completed (best model identified).

### 3. Tasks
1. Extend `src/train.py` with `save_model(pipeline, filepath='models/best_model.joblib')` and `load_model(filepath='models/best_model.joblib')`.
2. Serialize the winning pipeline (including the fitted `ColumnTransformer` and classifier).
3. Update `models/metrics.json` to record the selected `best_model` name, test metrics, and timestamp alongside comparison metrics.
4. Implement a reload verification script:
   - Load `models/best_model.joblib` into memory.
   - Pass an un-preprocessed single raw customer dictionary/DataFrame.
   - Verify that `.predict()` and `.predict_proba()` return expected predictions and probabilities.
   - Assert that reloaded model predictions match original in-memory pipeline outputs to 6 decimal places.

### 4. Files
- `src/train.py` (Modify to add serialization functions)
- `models/best_model.joblib` (Artifact created)
- `models/metrics.json` (Updated with best model info)

### 5. Implementation Guidance
- Saving the **entire pipeline** rather than just the classifier ensures that the application doesn't need to manually replicate scaling or one-hot encoding on new user input.
- Raw input goes in $\rightarrow$ Pipeline handles imputing, scaling, and one-hot encoding $\rightarrow$ Output is churn prediction.
- Keep the serialization clean: standard `joblib.dump()` and `joblib.load()`.

### 6. Commands
```bash
python -c "from src.train import save_best_model, test_reloaded_model; save_best_model(); test_reloaded_model()"
```

### 7. Verification
- `models/best_model.joblib` file exists and is non-empty (~100 KB - 5 MB depending on model).
- `models/model_info.json` contains model name, hyperparameter settings, and test metrics.
- Verification script passes: Reloaded model takes raw DataFrame input and predicts without raising `KeyError` or dimension mismatch errors.

### 8. Expected Outcome
A self-contained `.joblib` file that bundles preprocessing and inference, verified to produce reproducible predictions from raw feature dictionaries.

### 9. Common Mistakes & Solutions
- **Mistake:** Saving only the classifier without the fitted `ColumnTransformer`. The web app then fails because it receives raw strings and unscaled numbers that the classifier cannot ingest.
  - *Fix:* Always pass the full `Pipeline([('preprocessor', preprocessor), ('classifier', model)])` into `joblib.dump()`.
- **Mistake:** Python/Scikit-learn version mismatch when reloading.
  - *Fix:* Document exact library versions in `requirements.txt`.

### 10. Completion Checklist
- [ ] Serialization functions implemented using `joblib`
- [ ] Full pipeline (preprocessor + model) saved to `models/best_model.joblib`
- [ ] Metadata file `models/metrics.json` updated with best model info
- [ ] Verification script asserts identity between in-memory and reloaded predictions
- [ ] Raw input prediction verified on sample customer record

### 11. Learning Objectives (Viva Preparation)
- **Why do we serialize Scikit-Learn `Pipeline` rather than just the classifier?** A classifier only accepts numerical arrays in the exact dimension and scale it was trained on. A pipeline bundles the imputation, one-hot encoding, and scaling transformers together with the model, guaranteeing that raw user inputs undergo the exact same mathematical transformations.
- **What is `joblib` and why is it preferred over `pickle` for ML models?** `joblib` is optimized for serializing Python objects containing large NumPy arrays and Scikit-Learn models with higher efficiency and lower memory overhead than standard `pickle`.

---

## Phase 8 — Streamlit Interface

### 1. Objective
Develop an intuitive, lightweight, single-file **Streamlit** web application (`app.py`) providing:
1. An overview of the StayWise ML Lab project and dataset.
2. An interactive Model Comparison tab displaying metrics and visual evaluation charts.
3. An interactive Single-Customer Churn Prediction form with real-time risk gauge and probability readout.
4. A simple feature importance explanation showing what factors influenced the prediction.

### 2. Prerequisites
- Phase 6 and Phase 7 completed (`models/best_model.joblib` and `models/metrics.json` available).

### 3. Tasks
1. Create `app.py`.
2. Build four clean navigation sections/tabs:
   - **Tab 1: Project Overview & EDA:** Problem statement, dataset summary, class distribution, and project workflow.
   - **Tab 2: Model Comparison:** Interactive comparison table displaying Accuracy, Precision, Recall, F1, ROC-AUC across Logistic Regression, KNN, and Random Forest, plus embedded ROC and Confusion Matrix plots.
   - **Tab 3: Churn Predictor:** Input controls for customer features (Tenure slider, Contract dropdown, MonthlyCharges slider, InternetService selector, etc.), a "Predict Churn" button, and a visual display showing Churn Probability, Risk Category (Low, Medium, High), and final decision.
   - **Tab 4: Feature Insights & Viva Guide:** Top features driving churn (e.g. Month-to-month contracts, high monthly charges, short tenure) with plain-English interpretations.
3. Implement robust error handling (e.g., informative alert if model artifact has not been trained yet).
4. Verify the web app runs smoothly on local host (`http://localhost:8501`).

### 4. Files
- `app.py` (Create)

### 5. Implementation Guidance
- Keep `app.py` self-contained and clean.
- Use `st.cache_resource` to load `models/best_model.joblib` once to prevent re-reading the disk on every user interaction.
- Create default customer profiles (e.g. "Typical Loyal Customer" vs. "High Risk Churner") as quick presets so viva examiners can test scenarios in one click.

### 6. Commands
```bash
streamlit run app.py
```

### 7. Verification
- App starts locally on port 8501 without warnings or crashes.
- Model comparison tab clearly shows performance numbers and charts.
- Submitting the customer form instantly computes churn probability.
- Selecting different customer attributes (e.g., changing Contract from "Month-to-month" to "Two year") changes predicted churn probability logically.

### 8. Expected Outcome
A fully operational, responsive local web application ready for live viva presentation and customer churn scenario testing.

### 9. Common Mistakes & Solutions
- **Mistake:** Attempting to reload the model from disk on every widget change, resulting in UI lag.
  - *Fix:* Decorate model loading function with `@st.cache_resource`.
- **Mistake:** Mismatch between Streamlit input dictionary keys and raw CSV column names (e.g. `monthly_charges` vs `MonthlyCharges`).
  - *Fix:* Ensure input dictionary keys exactly match the training dataset column names.

### 10. Completion Checklist
- [ ] `app.py` implemented with clean tabs and layout
- [ ] Model loading cached with `@st.cache_resource`
- [ ] Model comparison table and evaluation figures rendered
- [ ] Interactive form accepts all customer parameters
- [ ] Output displays churn probability, binary prediction, and risk level
- [ ] Preset demo customer profiles added for easy viva presentation

### 11. Learning Objectives (Viva Preparation)
- **How is the prediction probability converted into an action in industry?** A probability score (e.g. 78% churn probability) allows businesses to rank customers by risk and allocate intervention budgets (e.g. calling top 5% at-risk customers with customized contract discounts).
- **What is the purpose of `@st.cache_resource`?** Streamlit reruns the entire Python script from top to bottom every time a user changes a widget. `@st.cache_resource` caches the loaded model in memory so it is only loaded once, ensuring high responsiveness.

---

## Phase 9 — Testing and Final Verification

### 1. Objective
Establish a suite of automated unit tests using `pytest` to verify the complete machine learning lifecycle: data loading, preprocessing pipelines, model training convergence, prediction shape/bounds, and model reloading integrity.

### 2. Prerequisites
- Phases 1 through 8 completed.

### 3. Tasks
1. Create `tests/__init__.py`.
2. Create `tests/test_preprocess.py`:
   - Test that raw data loading handles missing values and parses `TotalCharges` correctly.
   - Test that `customerID` is absent from processed features.
   - Test that `train_test_split` stratifies target classes properly.
   - Test that `ColumnTransformer` transforms input data into numeric arrays with zero NaN values.
3. Create `tests/test_model.py`:
   - Test that all three models (Logistic Regression, KNN, Random Forest) train and output probabilities in $[0.0, 1.0]$.
   - Test that `models/best_model.joblib` loads properly and outputs a 1D array of binary predictions.
   - Test edge-case inputs (e.g., maximum tenure, zero monthly charges) to ensure no crashes.
4. Run the full test suite with `pytest -v` and ensure 100% pass rate.

### 4. Files
- `tests/__init__.py` (Create)
- `tests/test_preprocess.py` (Create)
- `tests/test_model.py` (Create)

### 5. Implementation Guidance
- Keep tests lightweight and fast by running model tests with small iterations or subsets if needed, while verifying full pipeline integrity.
- Assertions should check concrete properties: `assert 0.0 <= proba <= 1.0`, `assert set(y_pred).issubset({0, 1})`.

### 6. Commands
```bash
pytest -v tests/
```

### 7. Verification
- All test functions execute and pass with green output.
- No warnings or uncaught exceptions.
- End-to-end integration test verifies that `app.py` dependencies and serialized models are fully functional.

### 8. Expected Outcome
A robust, automated test suite giving complete confidence that the project is free of bugs, syntax errors, or data leakage issues before final presentation.

### 9. Common Mistakes & Solutions
- **Mistake:** Tests failing because `data/telco_churn.csv` or `models/best_model.joblib` paths are hardcoded relative to a different working directory.
  - *Fix:* Use `pathlib.Path` or `os.path.join(os.path.dirname(__file__), ...)` to resolve paths relative to repository root.
- **Mistake:** Hardcoding exact floating-point metrics in assertions that might fluctuate slightly across OS/BLAS implementations.
  - *Fix:* Assert ranges (e.g., `assert roc_auc > 0.80`) rather than exact equality.

### 10. Completion Checklist
- [ ] `tests/test_preprocess.py` created and passing
- [ ] `tests/test_model.py` created and passing
- [ ] Edge cases (tenure=0, large total charges) tested
- [ ] `pytest -v` runs and reports 100% tests passing

### 11. Learning Objectives (Viva Preparation)
- **Why write automated tests for ML projects?** ML code fails silently: a bug in preprocessing (e.g., improper scaling or column misalignment) won't throw an error, but will silently destroy model accuracy. Unit testing catches data shape, value range, and leakage regressions before deployment.

---

## Phase 10 — Documentation and Viva Preparation

### 1. Objective
Complete comprehensive, polished documentation in `README.md`, including project abstract, architecture diagram, instructions to reproduce, actual benchmark results, limitations, and a dedicated **ML Lab Viva Voce Question Bank** with clear, textbook-accurate answers.

### 2. Prerequisites
- All preceding phases (1 through 9) completed, evaluated, and verified.

### 3. Tasks
1. Create/Update `README.md` with:
   - **Project Abstract:** Clear summary of StayWise, problem statement, and business relevance of customer churn.
   - **Architecture & Workflow:** ASCII diagram illustrating the pipeline from raw data to Streamlit UI.
   - **Setup & Execution Guide:** Step-by-step instructions to create environment, run preprocessing, train models, run tests, and launch Streamlit.
   - **Empirical Results Table:** Exact Accuracy, Precision, Recall, F1, and ROC-AUC measured across Logistic Regression, KNN, and Random Forest.
   - **Algorithm Explanations:** Intuitive, mathematical, and practical summaries of each algorithm.
   - **Project Limitations & Future Scope:** Discussion of class imbalance strategies (SMOTE, class weights), survival analysis, and customer lifetime value (CLV).
   - **Comprehensive Viva Voce Q&A Section:** Top 15 likely questions an examiner will ask during lab defense with clear, confident answers.
2. Review all files in the repository to ensure cleanliness, consistent formatting, and absence of temporary artifacts.

### 4. Files
- `README.md` (Create / Update)

### 5. Implementation Guidance
- Present actual numbers in the README table derived from the Phase 6 evaluation, not placeholder or estimated numbers.
- Write the Viva Q&A section with concise, conceptually sound explanations that an examiner wants to hear.

### 6. Commands
```bash
# Verify test suite and clean run
pytest -v tests/
streamlit run app.py
```

### 7. Verification
- `README.md` is complete, readable, and beautifully formatted in GitHub Markdown.
- Every command listed in `README.md` runs without error on a fresh setup.
- Viva preparation section covers all 3 algorithms, metrics, preprocessing choices, and operational aspects.

### 8. Expected Outcome
A complete, academic-grade project package ready for submission and presentation in the college ML lab exam.

### 9. Common Mistakes & Solutions
- **Mistake:** Leaving vague placeholders like "TBD" or "Results will be updated" in documentation.
  - *Fix:* Ensure actual numbers from Phase 6 are transcribed into the README.
- **Mistake:** Inability to explain why a particular model was chosen during the viva.
  - *Fix:* Memorize the trade-offs documented in the Viva Q&A section (e.g. why ROC-AUC / Recall was prioritized over simple accuracy).

### 10. Completion Checklist
- [ ] `README.md` created with project abstract and setup instructions
- [ ] Real empirical results table documented
- [ ] Algorithm theoretical and practical summaries included
- [ ] Project limitations and future scope documented
- [ ] Top 15 Viva Q&A written and studied
- [ ] Entire repository clean, tested, and ready for viva

### 11. Learning Objectives (Viva Preparation)
- **Complete confidence in defending StayWise:**
  1. Explaining the end-to-end data pipeline from raw CSV to deployed web app.
  2. Justifying the choice of the three algorithms and explaining their theoretical mechanics.
  3. Explaining data leakage and how Scikit-Learn `Pipeline` prevents it.
  4. Defending metric selection: why Recall and ROC-AUC trump Accuracy for churn prediction.
  5. Explaining feature importance and actionable business insights.

---

## 5. Summary Matrix of Implementation Phases

| Phase | Title | Key Output | Verification Command |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Environment Setup & Exploration | `requirements.txt`, `src/data_loader.py`, `data/telco_churn.csv` | `python -c "from src.data_loader import explore_data..."` |
| **Phase 2** | Data Preprocessing | `src/preprocess.py` (ColumnTransformer, zero leakage split) | `python -c "from src.preprocess import prepare_data..."` |
| **Phase 3** | Logistic Regression | `src/train.py` (5-fold CV tuning of `C`, test metrics) | `python -c "from src.train import train_logistic_regression..."` |
| **Phase 4** | K-Nearest Neighbors | `src/train.py` (Tuning odd $k$, scaling impact analysis) | `python -c "from src.train import train_knn..."` |
| **Phase 5** | Random Forest | `src/train.py` (Bagging tuning, feature importances) | `python -c "from src.train import train_random_forest..."` |
| **Phase 6** | Model Comparison | `src/evaluate.py`, `models/model_comparison.json`, plots | `python -c "from src.evaluate import compare_all_models..."` |
| **Phase 7** | Saving & Reloading Model | `models/best_model.joblib`, pipeline verification | `python -c "from src.train import test_reloaded_model..."` |
| **Phase 8** | Streamlit Interface | `app.py` (Interactive prediction, comparison, insights) | `streamlit run app.py` |
| **Phase 9** | Testing & Final Verification | `tests/test_preprocess.py`, `tests/test_model.py` | `pytest -v tests/` |
| **Phase 10** | Documentation & Viva Prep | `README.md` with complete Viva Q&A & project defense | Final walkthrough & viva review |

---

## 6. Execution Rules for Pair Programming

When you are ready to begin, simply tell me:  
👉 **"Start Phase 1"** (or specify any phase).

I will follow these rules:
1. Check that prerequisites for that phase are met.
2. Implement only the tasks and files for that phase.
3. Run the exact verification command and fix any issues.
4. Update the completion checklist in this plan.
5. Explain what was built, explain the key viva concepts, and pause to wait for your prompt to start the next phase.
