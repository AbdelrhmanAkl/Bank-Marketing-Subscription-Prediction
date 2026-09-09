# Bank Marketing Subscription Prediction

An end-to-end Machine Learning project for predicting whether a bank customer is likely to subscribe to a term deposit.

The project focuses on imbalanced classification, model comparison, hyperparameter tuning, decision-threshold optimization, SHAP explainability, and prediction error analysis.

---

## Project Overview

Bank marketing campaigns typically contact a large number of customers while only a small percentage subscribe to the offered term deposit.

The objective of this project is to build a classification system that identifies customers with a higher likelihood of subscription, helping marketing teams prioritize customers and improve campaign efficiency.

Because the dataset is highly imbalanced, the project focuses on **F1-score, Precision, Recall, and ROC-AUC** rather than accuracy alone.

---

## Dataset

The project uses the Bank Marketing dataset containing:

- 45,211 customers
- 16 input features
- 1 binary target variable
- Numerical and categorical features

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| No Subscription | 39,922 | 88.3% |
| Subscription | 5,289 | 11.7% |

This imbalance makes a naive accuracy-based approach misleading.

---

## Machine Learning Pipeline

The project follows a complete ML workflow:

1. Exploratory Data Analysis
2. Data quality analysis
3. Stratified train/validation/test split
4. Numerical feature preprocessing
5. Categorical feature preprocessing
6. Baseline modeling
7. Model comparison
8. Hyperparameter tuning
9. Cross-validation
10. Decision-threshold optimization
11. Final model selection
12. SHAP explainability
13. Error analysis
14. Model persistence
15. Customer-level prediction

---

## Preprocessing

### Numerical Features

- Median imputation
- StandardScaler

### Categorical Features

- Most-frequent imputation
- One-hot encoding
- Unknown category handling

The preprocessing pipeline and classifier are combined into a single Scikit-learn Pipeline.

After preprocessing, the feature matrix contains **50 features**.

---

## Models Evaluated

The following models were compared:

- Logistic Regression
- Class-Weighted Logistic Regression
- Random Forest
- Tuned Random Forest
- HistGradientBoosting
- XGBoost

Stratified 5-Fold Cross-Validation was used where applicable.

---

## Model Comparison

| Model | Validation F1 | ROC-AUC | CV Mean F1 |
|---|---:|---:|---:|
| **Tuned Random Forest** | **0.4918** | 0.7965 | **0.4693** |
| HistGradientBoosting | 0.3367 | 0.8054 | 0.3325 |
| XGBoost | 0.3327 | **0.8077** | 0.3317 |
| Random Forest | 0.3233 | 0.7966 | 0.3000 |
| Logistic Regression | 0.2843 | 0.7699 | — |

The **Tuned Random Forest** was selected as the final model because it achieved the strongest F1-score and cross-validation F1 performance.

---

## Hyperparameter Tuning

RandomizedSearchCV with Stratified 5-Fold Cross-Validation was used to optimize the Random Forest.

### Best Parameters

```text
class_weight      = balanced_subsample
max_depth         = None
max_features      = 0.7
min_samples_leaf  = 7
min_samples_split = 10
n_estimators      = 527

Best Cross-Validation F1:

0.4693

### Tuned Random Forest Performance

Accuracy : 0.8769
Precision: 0.4759
Recall   : 0.5088
F1 Score : 0.4918
ROC-AUC  : 0.7965

---

## Decision Threshold Optimization

The default classification threshold of `0.50` was optimized using the validation set to improve the balance between Precision and Recall.

The selected threshold for the final Random Forest model was:

0.44

### Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 86.52% |
| Precision | 44.09% |
| Recall | 56.42% |
| F1-score | **49.50%** |
| ROC-AUC | 79.65% |

Threshold optimization improved the model's ability to identify customers who are more likely to subscribe.

---

## Final Model

The final model is a **Tuned Random Forest** with a decision threshold of `0.44`.

```text
Model     : Tuned Random Forest
Threshold : 0.44