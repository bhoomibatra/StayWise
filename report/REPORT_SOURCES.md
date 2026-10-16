# StayWise — Academic Report Sources & Literature Verification

This document compiles the verified primary research literature, benchmark datasets, software frameworks, and experimental citations utilized in the **StayWise Customer Churn Prediction and Retention Intelligence** project report.

---

## 1. Verified Peer-Reviewed Literature (2021 – 2024)

### Reference [1]
- **Authors:** Praveen Lalwani, Manas Kumar Mishra, Jasroop Singh Chadhar, and Pradeep Sethi
- **Publication Year:** 2022 (Accepted 2021)
- **Paper Title:** *Customer churn prediction system: a machine learning approach*
- **Journal:** *Computing* (Springer), Volume 104, Issue 2, Pages 271–294
- **DOI:** [10.1007/s00607-021-00908-y](https://doi.org/10.1007/s00607-021-00908-y)
- **Dataset:** IBM Telco Customer Churn & Cell2Cell
- **Methodology & Algorithms:** Benchmarked Logistic Regression, Naive Bayes, Support Vector Machines, Decision Trees, Random Forest, and XGBoost; applied SMOTE for class imbalance and sequential feature selection.
- **Reported Performance:** Random Forest achieved 81.3% accuracy, 0.84 ROC-AUC, and 0.58 F1-score on the balanced Telco dataset.
- **Relevance & Research Gap:** Provided baseline evidence for tree-ensemble efficacy but relied on offline batch scoring without operational retention workflow integration.

### Reference [2]
- **Authors:** Shengnan Wu, Wei Chuen Yau, Thian Song Ong, and Siew Chin Chong
- **Publication Year:** 2021
- **Paper Title:** *Integrated Churn Prediction and Customer Segmentation Framework for Telco Business*
- **Journal:** *IEEE Access*, Volume 9, Pages 62118–62136
- **DOI:** [10.1109/ACCESS.2021.3073776](https://doi.org/10.1109/ACCESS.2021.3073776)
- **Dataset:** Public Telecom Benchmark Data
- **Methodology & Algorithms:** Integrated K-Means customer segmentation with supervised classifiers (Random Forest, Logistic Regression, XGBoost) to generate tier-specific retention actions.
- **Reported Performance:** Random Forest attained 82.4% accuracy and 0.85 ROC-AUC across heterogeneous subscriber clusters.
- **Relevance & Research Gap:** Demonstrated the necessity of connecting model risk outputs with customer segmentation, directly inspiring StayWise's High/Medium/Low operational risk tiering.

### Reference [3]
- **Authors:** Naveen Jajam, N. P. Challa, K. S. L. Prasanna, and C. H. V. S. Deepthi
- **Publication Year:** 2023
- **Paper Title:** *Arithmetic Optimization With Ensemble Deep Learning SBLSTM-RNN-IGSA Model for Customer Churn Prediction*
- **Journal:** *IEEE Access*, Volume 11, Pages 93111–93128
- **DOI:** [10.1109/ACCESS.2023.3304669](https://doi.org/10.1109/ACCESS.2023.3304669)
- **Dataset:** Kaggle / IBM Telco Customer Churn Dataset
- **Methodology & Algorithms:** Bidirectional LSTM with Recurrent Neural Networks optimized via Improved Gravitational Search and Arithmetic Optimization Algorithm.
- **Reported Performance:** Reported high classification accuracy (~84.2%) and ROC-AUC of 0.87.
- **Relevance & Research Gap:** Achieved high accuracy through complex deep architectures but lacked interpretability and required prohibitive computational training resources compared to transparent linear and ensemble classifiers.

### Reference [4]
- **Authors:** Chamod Ramanayake, Danushka Liyanage, Pramuditha Dhananjaya, Chaminda Kumara, and T. G. I. Fernando
- **Publication Year:** 2024
- **Paper Title:** *Telco Customer Churn Prediction Using Composite Deep Learning Technique*
- **Conference:** *2024 6th International Conference on Advancements in Computing (ICAC)*, Pages 402–407
- **DOI:** [10.1109/ICAC64487.2024.10851050](https://doi.org/10.1109/ICAC64487.2024.10851050)
- **Dataset:** IBM Telco Customer Churn
- **Methodology & Algorithms:** Composite hybrid deep learning integrating multi-layer dense networks with non-linear feature embeddings.
- **Reported Performance:** 80.8% accuracy and 0.835 ROC-AUC on holdout test partition.
- **Relevance & Research Gap:** Confirmed that deep composite models yield minimal marginal gains over well-regularized classical algorithms (such as Logistic Regression and Random Forest) on tabular customer demographic data.

### Reference [5]
- **Authors:** Abhikumar Patel and Amit G. Kumar
- **Publication Year:** 2023
- **Paper Title:** *Predicting Customer Churn In Telecom Industry: A Machine Learning Approach For Improving Customer Retention*
- **Conference:** *2023 IEEE 11th Region 10 Humanitarian Technology Conference (R10-HTC)*, Rajkot, India, Pages 642–647
- **DOI:** [10.1109/R10-HTC57777.2023.10461726](https://doi.org/10.1109/R10-HTC57777.2023.10461726)
- **Dataset:** IBM Telco Dataset (7,043 instances)
- **Methodology & Algorithms:** Evaluated Decision Trees, Random Forest, Logistic Regression, and Support Vector Machines; highlighted tenure and contract types as primary churn determinants.
- **Reported Performance:** Random Forest accuracy of 79.8% and Logistic Regression accuracy of 80.2%.
- **Relevance & Research Gap:** Validated the comparative efficacy of Logistic Regression and Random Forest on the identical dataset, but lacked an automated real-time intervention workflow.

---

## 2. Dataset & Foundational Tool Citations

### Reference [6]
- **Title:** *IBM Telco Customer Churn Dataset*
- **Publisher/Platform:** IBM Community & Kaggle Data Repository
- **Year:** 2018 (Standard Benchmark)
- **URL:** [https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Description:** 7,043 subscriber records, 21 attributes including demographics, subscribed telephone/internet services, billing details, and binary churn label.

### Reference [7]
- **Authors:** F. Pedregosa et al.
- **Publication Year:** 2011
- **Paper Title:** *Scikit-learn: Machine Learning in Python*
- **Journal:** *Journal of Machine Learning Research*, Volume 12, Pages 2825–2830
- **URL:** [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

### Reference [8]
- **Authors:** Wes McKinney
- **Publication Year:** 2010
- **Paper Title:** *Data Structures for Statistical Computing in Python*
- **Proceedings:** *9th Python in Science Conference (SciPy 2010)*, Pages 56–61
- **DOI:** [10.25080/Majora-92bf1924-003](https://doi.org/10.25080/Majora-92bf1924-003)

### Reference [9]
- **Authors:** J. D. Hunter
- **Publication Year:** 2007
- **Paper Title:** *Matplotlib: A 2D Graphics Environment*
- **Journal:** *Computing in Science & Engineering*, Volume 9, Issue 3, Pages 90–95
- **DOI:** [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

### Reference [10]
- **Title:** *FastAPI: Modern, High-Performance Web Framework for Python*
- **Author:** Sebastián Ramírez
- **Year:** 2024
- **Official Documentation:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

---

## 3. Systematic Literature Comparison Matrix

| No. | Reference & Year | Dataset Used | Primary Algorithms | Reported Key Metrics | Identified Research Gap & Relation to StayWise |
|:---:|:---|:---|:---|:---|:---|
| **[1]** | Lalwani et al. (2022) *Computing* | IBM Telco & Cell2Cell | LR, NB, SVM, DT, RF, XGBoost | RF Acc: 81.3%, AUC: 0.84, F1: 0.58 | Focused on static batch evaluation; no real-time scoring engine or CRM retention tasking. |
| **[2]** | Wu et al. (2021) *IEEE Access* | Public Telco Benchmark | K-Means + RF, LR, XGBoost | RF Acc: 82.4%, AUC: 0.85 | Examined clustering but lacked individual customer 360 drilldown and persistence. |
| **[3]** | Jajam et al. (2023) *IEEE Access* | Telco Churn (Kaggle) | BiLSTM-RNN + IGSA Optimization | Acc: 84.2%, AUC: 0.87 | High computational complexity; black-box deep model hinders explainable retention policies. |
| **[4]** | Ramanayake et al. (2024) *IEEE ICAC* | IBM Telco Churn | Composite Deep Neural Network | Acc: 80.8%, AUC: 0.835 | Minimal performance uplift over classical ML; does not justify operational deployment overhead. |
| **[5]** | Patel & Kumar (2023) *IEEE R10-HTC* | IBM Telco (7,043 rows) | LR, DT, RF, SVM | LR Acc: 80.2%, RF Acc: 79.8% | Did not calibrate probability thresholds or address post-prediction customer intervention workflows. |
| **[6]** | **StayWise (This Work)** | **IBM Telco (7,043 rows)** | **LR (C=10.0), KNN, RF, SVM** | **LR Test AUC: 0.8411, Recall: 55.88%, F1: 0.6040, Acc: 80.55%** | **Fills all identified gaps: leak-free Pipeline, customer 360 cockpit, SQLite task tracking, React SPA.** |

---

## 4. Verification & Integrity Checklist
- [x] All 5 research papers verified through IEEE Xplore and Springer indexed sources.
- [x] DOIs confirmed to resolve directly to publisher landing pages.
- [x] Dataset attributes and row counts (7,043 records) cross-verified against `data/telco_churn.csv`.
- [x] All experimental model metrics in StayWise extracted directly from held-out test splits (`models/metrics.json` and `models/evaluation_details.json`).
