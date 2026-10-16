# StayWise — Viva Voce Preparation Guide

**Project:** StayWise — Customer Churn Prediction and Retention Intelligence  
**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features)  
**Algorithms:** Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Machine  
**Best Model:** Logistic Regression (Accuracy: 80.55%, Recall: 55.88%, ROC-AUC: 0.8411)

---

## Section A — Project Introduction and Motivation

---

**Q1. In one or two sentences, what is StayWise?**

StayWise is a machine learning project that predicts whether a telecom customer is likely to cancel their subscription (churn). It compares four classification algorithms — Logistic Regression, KNN, Random Forest, and SVM — and serves predictions through an interactive Streamlit web application.

---

**Q2. Why is predicting customer churn important?**

Acquiring a new customer costs five to seven times more than retaining an existing one. If a company can identify at-risk customers before they cancel, it can intervene with targeted offers, contract discounts, or priority support. StayWise gives companies a data-driven list of customers who need retention effort, rather than waiting for cancellations to happen.

---

**Q3. What dataset does StayWise use and where does it come from?**

StayWise uses the **IBM Telco Customer Churn** dataset, publicly available on IBM's GitHub. It contains **7,043 rows and 21 columns**, representing telecommunications subscribers and whether they churned (`Yes`) or stayed (`No`). The dataset is stored locally as `data/telco_churn.csv`.

---

**Q4. What is the target variable and what does it mean?**

The target variable is `Churn`. It is a categorical column with values `Yes` (customer cancelled) and `No` (customer stayed). In the code, it is mapped to binary integers: `Yes → 1` and `No → 0`. The dataset is imbalanced: approximately **26.54% churners** and **73.46% non-churners**.

---

**Q5. Which features are used for prediction and which are removed?**

`customerID` is removed because it is a unique identifier — it has no relationship with churn and would cause the model to memorise individual records rather than learn generalizable patterns.

The remaining 19 columns are divided into:
- **3 numerical features:** `tenure`, `MonthlyCharges`, `TotalCharges`
- **16 categorical features:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## Section B — Data Cleaning and Preprocessing

---

**Q6. Was there any missing data in the dataset? How was it handled?**

Yes — **11 rows** have a whitespace string `" "` in the `TotalCharges` column instead of a number. These are rows where `tenure = 0` (brand new customers who had not yet been billed). The code detects these using `pd.to_numeric(errors='coerce')`, which converts invalid strings to `NaN`. The `NaN` values are then filled using the **median of the training set only** via `SimpleImputer(strategy='median')` inside the Scikit-learn `Pipeline`. This is the only missing data in the dataset.

---

**Q7. What is One-Hot Encoding and why is it used?**

Machine learning algorithms work with numbers, not text. One-hot encoding converts each categorical column into multiple binary (0/1) columns — one per category. For example, `Contract` with values `Month-to-month`, `One year`, and `Two year` becomes three columns: `Contract_Month-to-month`, `Contract_One year`, `Contract_Two year`. Each customer has exactly one `1` and the rest `0`. The project uses `OneHotEncoder(handle_unknown='ignore')` so that if an unseen category appears at prediction time (e.g., from the Streamlit form), it is silently ignored rather than crashing.

---

**Q8. What is Feature Scaling and why does StayWise apply it?**

Feature scaling standardizes numerical features so that they all have the same order of magnitude. Without scaling, `TotalCharges` (range: 0–8,685) would dominate `SeniorCitizen` (range: 0–1) in any distance or margin calculation. StayWise uses `StandardScaler`, which transforms each numerical feature to have **zero mean and unit variance** (z-score normalization). This is essential for KNN and SVM, but has no effect on Random Forest.

---

**Q9. What is the Train/Test split and how was it done?**

The dataset is split **80% training (5,634 rows) and 20% test (1,409 rows)** using `train_test_split(stratify=y, random_state=42)`. Stratification ensures both splits have the same proportion of churners (~26.54%), so neither split accidentally contains too many or too few churners. `random_state=42` makes the split reproducible.

---

**Q10. What is Data Leakage and how does StayWise prevent it?**

Data leakage happens when information from the test set is used — even indirectly — during training. A common mistake is fitting a `StandardScaler` on the entire dataset before splitting, which means the scaler knows the mean and variance of the test set. This makes the model appear to perform better than it truly does on unseen data.

StayWise prevents this by performing the train/test split **first**, then bundling the `ColumnTransformer` inside a Scikit-learn `Pipeline`. The pipeline is fitted only on `X_train`, so the scaler, imputer, and encoder never see any test set values.

---

## Section C — Model Training and Cross-Validation

---

**Q11. What is Cross-Validation and why is it used for hyperparameter tuning?**

Cross-validation evaluates a model on multiple non-overlapping subsets of the training data. StayWise uses **5-fold Stratified Cross-Validation**: the training set (5,634 rows) is divided into 5 equal folds. The model trains on 4 folds and validates on the remaining fold, repeating 5 times. The average validation ROC-AUC across all 5 folds is the score. This process is repeated for each hyperparameter combination in the grid, and the combination with the highest average CV score is selected. The **test set is never used during this process**.

---

**Q12. What is a hyperparameter? Give examples from StayWise.**

A hyperparameter is a setting chosen before training that controls the learning process, not learned from data.

Examples from StayWise:
- **Logistic Regression:** `C` (inverse regularization strength) — tested values: `[0.1, 1.0, 10.0]`, best: `C = 10.0`
- **KNN:** `k` (number of neighbours) — tested values: `[5, 9, 15, 21]`, best: `k = 21`
- **Random Forest:** `n_estimators` (number of trees) and `max_depth` (maximum tree depth) — best: `n_estimators=100`, `max_depth=5`
- **SVM:** `C` (penalty for margin violations) — tested values: `[0.1, 1.0, 10.0]`, best: `C = 0.1`

---

## Section D — Logistic Regression

---

**Q13. How does Logistic Regression work?**

Logistic Regression fits a linear combination of the input features and passes it through the **sigmoid function** to produce a probability between 0 and 1:

```
P(Churn = 1 | X) = 1 / (1 + e^(-(w₀ + w₁·x₁ + ... + wₙ·xₙ)))
```

If this probability is ≥ 0.5, the customer is predicted to churn. The model is trained by minimizing **log-loss** (cross-entropy) across all training examples. The `C` parameter controls the regularization strength: a smaller `C` penalizes large weights more strongly, reducing overfitting.

---

**Q14. Why was Logistic Regression chosen as the best model?**

Logistic Regression was selected because it achieved:
- The **highest Recall (55.88%)** — it catches the most churners
- The **highest F1-Score (0.6040)** — best balance of precision and recall
- The **highest ROC-AUC (0.8411)** — best ability to distinguish churners from non-churners at all thresholds

In customer churn, **missing a churner (False Negative) is far more costly** than incorrectly flagging a loyal customer (False Positive). Logistic Regression minimizes this cost best among the four models. It is also transparent, fast to train and predict, and produces well-calibrated probability scores.

---

## Section E — K-Nearest Neighbors

---

**Q15. How does KNN work?**

KNN is a "lazy learner" — it does not learn any parameters during training. Instead, it stores the entire training set. When a new customer needs a prediction, KNN finds the `k` most similar customers in the training set (using Euclidean distance) and predicts the majority class among those neighbours. For `k = 21`, if 13 of the 21 nearest customers churned, the new customer is predicted to churn.

---

**Q16. Why is Feature Scaling mandatory for KNN?**

KNN uses Euclidean distance:
```
distance = √((x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)²)
```

Without scaling, `TotalCharges` (which can differ by thousands between customers) completely dominates this calculation. A difference of 1,000 in `TotalCharges` contributes 1,000,000 to the squared distance, while a difference of 1 in `SeniorCitizen` contributes only 1. After `StandardScaler`, all features are on a comparable scale and contribute meaningfully to the distance.

---

**Q17. Why are odd values of k preferred in binary classification?**

If `k` is even, it is possible to have exactly equal votes (e.g., 10 neighbours say Churn, 10 say No Churn), which creates an unresolvable tie. Odd values of `k` (5, 9, 15, 21) guarantee a strict majority and avoid this ambiguity.

---

## Section F — Random Forest

---

**Q18. How does Random Forest work?**

Random Forest is an ensemble method. It trains many **decision trees** (100 in our best model) on different **bootstrap samples** of the training data (random samples with replacement). At each split node in a tree, only a random subset of features is considered. Each tree makes a prediction, and the final prediction is the **majority vote** across all trees. This process — combining many diverse, independently trained models — is called **bagging** (Bootstrap Aggregating).

---

**Q19. Why does Random Forest reduce overfitting compared to a single Decision Tree?**

A single decision tree tends to overfit: it can memorize the training data completely (achieving near-100% training accuracy) but performs poorly on unseen data. Random Forest reduces this by:
1. **Diversity through bootstrapping** — each tree sees a different random subset of training data, so the trees make different errors.
2. **Feature randomness** — each split considers only a random subset of features, preventing any single feature from dominating all trees.
3. **Averaging** — averaging the predictions from many imperfect trees cancels out individual errors, reducing variance without increasing bias.

---

**Q20. What is Feature Importance in Random Forest?**

Feature importance (Mean Decrease in Impurity / Gini Importance) measures how much each feature reduces uncertainty (impurity) across all split nodes in all trees. The top features in StayWise's Random Forest are:

1. `Contract_Month-to-month` (16.61%) — customers on monthly contracts churn most
2. `tenure` (14.19%) — newer customers churn more
3. `OnlineSecurity_No` (9.06%) — customers without online security are higher risk
4. `TotalCharges` (7.75%) — lower total charges correlates with shorter tenure and higher churn risk
5. `InternetService_Fiber optic` (7.42%) — fibre customers show higher churn rates

---

## Section G — Support Vector Machine (SVM)

---

**Q21. How does SVM work?**

SVM finds a **hyperplane** (a flat decision boundary) that separates churners from non-churners with the **maximum margin** — the largest possible buffer zone between the boundary and the nearest data points from either class. The boundary is defined mathematically as:

```
w·x + b = 0
```

If `w·x + b ≥ 0`, the customer is predicted as Churn (1). If it is < 0, the customer is predicted as No Churn (0). With `probability=True`, a Platt scaling sigmoid maps the decision score to a calibrated probability.

---

**Q22. What are a Hyperplane, Margin, and Support Vectors?**

- **Hyperplane:** The decision boundary surface `w·x + b = 0`. In 2D it is a line, in 3D it is a plane, in higher dimensions it is a hyperplane.
- **Margin:** The perpendicular distance between the hyperplane and the nearest data points from each class. The margin width is `2 / ||w||`. SVM maximises this margin to improve generalisation.
- **Support Vectors:** The specific training data points that lie exactly on the margin boundaries (`w·x + b = ±1`). Only these points influence the position and orientation of the hyperplane. Removing any other training point would not change the model at all.

---

**Q23. What does the regularization parameter C do in SVM?**

`C` controls the trade-off between a wide margin and low training error:
- **Small C (e.g., 0.1):** The model tolerates more margin violations (misclassified training points) in exchange for a wider margin. This is stronger regularization and reduces overfitting.
- **Large C (e.g., 10.0):** The model penalizes misclassifications heavily, resulting in a narrower margin and tighter fit to the training data. This can overfit.

The best value found by 5-fold CV in StayWise was **C = 0.1**, meaning the model benefits from a wider, more regularized margin on this dataset.

---

**Q24. Why did SVM achieve a lower ROC-AUC (0.8266) than Logistic Regression (0.8411)?**

Three reasons:
1. **Class imbalance:** The dataset is ~73.5% non-churners. Standard SVM margin optimization can be pulled toward the majority class. Logistic Regression handles this more gracefully with its probability calibration.
2. **Linear boundary rigidity:** The linear SVM draws a single flat hyperplane. Churn patterns likely have some non-linear structure that a linear boundary cannot capture without a kernel transformation.
3. **Data overlap:** Churners and non-churners in this dataset share similar feature profiles (similar monthly charges, similar tenure), making a perfectly separating margin impossible. Logistic Regression's smooth probabilistic model handles this overlap better than SVM's hard boundary.

---

## Section H — Model Evaluation Metrics

---

**Q25. Why is Accuracy a misleading metric for this dataset?**

Because the dataset is imbalanced — 73.46% of customers do NOT churn. A naive classifier that always predicts "No Churn" would achieve **73.46% accuracy** without learning anything. It would catch **zero churners** (Recall = 0%). Accuracy rewards predicting the majority class regardless of the model's ability to identify the minority class (churners). This is why StayWise prioritises Recall, F1-Score, and ROC-AUC.

---

**Q26. What is Precision and Recall and why is there a trade-off?**

- **Precision = TP / (TP + FP):** Of all customers the model predicted to churn, what fraction actually did? High precision means fewer false alarms.
- **Recall = TP / (TP + FN):** Of all customers who actually churned, what fraction did the model catch? High recall means fewer missed churners.

There is a trade-off: if the model becomes more aggressive (predicts "Churn" more often), it catches more real churners (higher Recall) but also flags more loyal customers incorrectly (lower Precision). In churn prediction, Recall is prioritised because the cost of missing a churner (losing a customer permanently) is much higher than the cost of a false alarm (sending one unnecessary retention email).

---

**Q27. What is ROC-AUC?**

The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (Recall) on the y-axis against the False Positive Rate on the x-axis at every possible decision threshold. The AUC (Area Under the Curve) summarizes the curve as a single number:
- **AUC = 1.0:** Perfect classifier — ranks all churners above all non-churners
- **AUC = 0.5:** Random classifier — no discriminating ability
- **StayWise's best (Logistic Regression): AUC = 0.8411** — the model has strong ranking ability

ROC-AUC is threshold-independent, meaning it evaluates the model's fundamental ranking quality regardless of where you draw the line between "churn" and "no churn".

---

**Q28. What is a Confusion Matrix?**

A confusion matrix shows all four possible prediction outcomes for a binary classifier:

```
                   Predicted: No Churn   Predicted: Churn
Actual: No Churn   True Negative (TN)    False Positive (FP)
Actual: Churn      False Negative (FN)   True Positive (TP)
```

For StayWise's best model (Logistic Regression) on 1,409 test rows:
- **TN = 918:** Correctly identified as loyal → no action needed
- **FP = 117:** Incorrectly flagged as churner → unnecessary retention cost (low)
- **FN = 164:** Missed churner → customer leaves without intervention (high cost)
- **TP = 210:** Correctly identified as churner → retention team can intervene

---

## Section I — Pipelines, Serialization, and the Application

---

**Q29. What is a Scikit-learn Pipeline and why is it important?**

A Scikit-learn `Pipeline` chains multiple processing steps into a single object. In StayWise, each model pipeline is:
```
Pipeline([
    ('preprocessor', ColumnTransformer),   # impute + scale + encode
    ('classifier', LogisticRegression)     # make predictions
])
```

This is important for two reasons:
1. **No data leakage:** The `fit()` call fits both the preprocessor and the model on training data only. The `transform()` and `predict()` calls apply the same fitted transformers to new data.
2. **Deployment simplicity:** The Streamlit app can pass a raw, un-processed customer DataFrame directly to `pipeline.predict()` without any manual feature engineering — the pipeline handles everything internally.

---

**Q30. What is Joblib and why is it used instead of pickle?**

`joblib` is a Python library optimised for serializing (saving) large NumPy arrays and Scikit-learn models. It is preferred over the standard `pickle` library because it is more memory-efficient when saving models that contain large arrays (such as the one-hot encoded feature weights in Logistic Regression or the 100 decision trees in Random Forest). The entire fitted pipeline — preprocessor AND classifier — is saved to `models/best_model.joblib` so that prediction on new data requires no retraining.

---

**Q31. How does the Streamlit application work?**

`app.py` is a single-file Streamlit application. When a user opens it in the browser:
1. The saved `models/best_model.joblib` is loaded once using `@st.cache_resource` (so it is not reloaded on every widget interaction).
2. In the **Customer Churn Predictor** tab, the user fills in a form with 19 customer attributes.
3. On clicking "Compute Churn Prediction", the values are assembled into a one-row pandas DataFrame and passed to `pipeline.predict()` and `pipeline.predict_proba()`.
4. The pipeline internally imputes missing values, standardizes numbers, one-hot encodes categories, and feeds the result into Logistic Regression.
5. The app displays the predicted class (Churn / No Churn), the churn probability (e.g., 67.2%), and the risk level (Low / Medium / High).

---

## Section J — Limitations and Context

---

**Q32. What are the main limitations of StayWise?**

1. **Class imbalance is not explicitly handled.** No SMOTE oversampling, no `class_weight='balanced'`, and no probability threshold tuning was applied. Recall could potentially be improved by lowering the prediction threshold from 0.5.
2. **Linear-only SVM.** Non-linear kernels (RBF) might capture more complex churn patterns but were not tested due to training time constraints.
3. **Static dataset.** The model is trained on a historical snapshot. Customer behaviour changes over time, so the model would need periodic retraining in a production setting.
4. **No feature selection.** All 19 features are used. Some (like `gender`) contribute minimal predictive signal and could be removed.

---

**Q33. If you could improve this project further, what would you add?**

1. **Threshold tuning:** Lower the classification threshold to 0.35–0.40 to increase Recall at the expense of some Precision, since catching more churners is more valuable.
2. **Class balancing:** Use `class_weight='balanced'` in Logistic Regression and SVM, or apply SMOTE oversampling before training.
3. **More algorithms:** Test XGBoost or LightGBM, which often outperform the above four algorithms on tabular data.
4. **SHAP explanations:** Use SHAP (SHapley Additive exPlanations) to show which specific features most influenced an individual customer's prediction — useful for the Streamlit app.
5. **Retraining pipeline:** Build an automated script that retrains models when new data is uploaded, updates `metrics.json`, and deploys the best pipeline automatically.

---

*StayWise — Viva Voce Preparation Guide | Department of Computer Science & Engineering*
