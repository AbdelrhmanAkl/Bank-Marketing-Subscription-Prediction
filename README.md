# 🏦 Bank Marketing Subscription Prediction

An end-to-end **Machine Learning classification project** that predicts whether a bank customer is likely to subscribe to a term deposit.

The project focuses on **imbalanced classification**, model benchmarking, hyperparameter optimization, decision-threshold tuning, explainability, and customer-level prediction.

---

## 🚀 Live Demo

### 🌐 Try the Interactive Application

👉 **[Launch the Streamlit App](https://bank-marketing-subscription-prediction-yu3uarppazgwrybswiuxkq.streamlit.app/)**

The deployed application allows users to enter customer information and receive a real-time prediction of whether the customer is likely to subscribe to a term deposit.

**Deployment Stack:**

* Streamlit Community Cloud
* Hugging Face Hub
* Scikit-learn
* Python

---

## 🤗 Trained Model

The trained **Tuned Random Forest** model is hosted separately on Hugging Face and downloaded dynamically by the Streamlit application.

👉 **[View the Model on Hugging Face](https://huggingface.co/AbdelrahmanAkl/bank-marketing-subscription-model)**

**Model:** Tuned Random Forest
**Decision Threshold:** `0.44`
**Model Format:** `.joblib`

Hosting the model separately keeps the GitHub repository lightweight while allowing the deployed application to load the trained model at runtime.

---

## 📌 Project Overview

Bank marketing campaigns typically contact a large number of customers, while only a relatively small percentage eventually subscribe to the offered term deposit.

The objective of this project is to build a classification system that identifies customers with a higher probability of subscription.

This can help marketing teams:

* Prioritize high-potential customers
* Improve campaign efficiency
* Reduce unnecessary customer contacts
* Better balance marketing effort and conversion potential

Because the target variable is highly imbalanced, **accuracy alone is not sufficient** for evaluating model performance.

Therefore, the project emphasizes:

* **F1-score**
* **Precision**
* **Recall**
* **ROC-AUC**

---

## 📊 Dataset

The project uses the **Bank Marketing dataset**, containing:

* **45,211 customers**
* **16 input features**
* **1 binary target variable**
* Numerical and categorical features

### Target Distribution

| Class           |  Count | Percentage |
| --------------- | -----: | ---------: |
| No Subscription | 39,922 |      88.3% |
| Subscription    |  5,289 |      11.7% |

The strong class imbalance makes a naive accuracy-based approach potentially misleading.

---

## 🔄 Machine Learning Pipeline

The project follows a complete end-to-end ML workflow:

1. Exploratory Data Analysis
2. Data quality analysis
3. Stratified train/validation/test split
4. Numerical feature preprocessing
5. Categorical feature preprocessing
6. Baseline modeling
7. Model benchmarking
8. Hyperparameter tuning
9. Stratified cross-validation
10. Decision-threshold optimization
11. Final model selection
12. SHAP explainability
13. Error analysis
14. Model persistence
15. Customer-level prediction
16. Streamlit deployment

---

## ⚙️ Preprocessing

### Numerical Features

* Median imputation
* StandardScaler

### Categorical Features

* Most-frequent imputation
* One-hot encoding
* Unknown-category handling

The preprocessing steps and classifier are combined into a single **Scikit-learn Pipeline**, ensuring that the same transformations are consistently applied during training and inference.

After preprocessing, the feature matrix contains **50 features**.

---

## 🤖 Models Evaluated

Several classification algorithms were evaluated:

* Logistic Regression
* Class-Weighted Logistic Regression
* Random Forest
* Tuned Random Forest
* HistGradientBoosting
* XGBoost

Stratified 5-Fold Cross-Validation was used where applicable to obtain a more robust estimate of model performance.

---

## 📈 Model Comparison

| Model                   | Validation F1 |    ROC-AUC | CV Mean F1 |
| ----------------------- | ------------: | ---------: | ---------: |
| **Tuned Random Forest** |    **0.4918** |     0.7965 | **0.4693** |
| HistGradientBoosting    |        0.3367 |     0.8054 |     0.3325 |
| XGBoost                 |        0.3327 | **0.8077** |     0.3317 |
| Random Forest           |        0.3233 |     0.7966 |     0.3000 |
| Logistic Regression     |        0.2843 |     0.7699 |          — |

### 🏆 Model Selection

The **Tuned Random Forest** was selected as the final model because it achieved the strongest **F1-score** and **cross-validation F1 performance** among the evaluated models.

Although XGBoost achieved the highest ROC-AUC, the Tuned Random Forest provided a substantially stronger F1-score, making it more suitable for the project's primary objective of balancing Precision and Recall for the minority class.

---

## 🎯 Hyperparameter Tuning

`RandomizedSearchCV` with **Stratified 5-Fold Cross-Validation** was used to optimize the Random Forest.

### Best Parameters

```text
class_weight       = balanced_subsample
max_depth          = None
max_features       = 0.7
min_samples_leaf   = 7
min_samples_split  = 10
n_estimators       = 527
```

### Best Cross-Validation F1

```text
0.4693
```

### Tuned Random Forest Performance

| Metric       |      Score |
| ------------ | ---------: |
| Accuracy     |     87.69% |
| Precision    |     47.59% |
| Recall       |     50.88% |
| **F1 Score** | **49.18%** |
| ROC-AUC      |     79.65% |

---

## 🎚️ Decision Threshold Optimization

The default classification threshold of `0.50` was optimized using the validation set to improve the balance between Precision and Recall.

The selected threshold for the final Random Forest model was:

```text
0.44
```

### Validation Performance

| Metric       |      Score |
| ------------ | ---------: |
| Accuracy     |     86.52% |
| Precision    |     44.09% |
| Recall       |     56.42% |
| **F1-score** | **49.50%** |
| ROC-AUC      |     79.65% |

Lowering the decision threshold from `0.50` to `0.44` increased the model's sensitivity to potential subscribers and improved the balance between Precision and Recall.

---

## 🧠 Final Model

The final production model is a **Tuned Random Forest** combined with the complete preprocessing pipeline.

```text
Model              : Tuned Random Forest
Decision Threshold : 0.44
Features           : 50
Model Format       : Joblib
```

The trained model is persisted and hosted on Hugging Face for use by the deployed Streamlit application.

---

## 🔍 Explainability

**SHAP (SHapley Additive exPlanations)** was used to analyze model behavior and understand which features contribute most strongly to predictions.

This provides greater interpretability and helps identify the factors influencing customer subscription predictions.

---

## 🧪 Error Analysis

The project also includes prediction error analysis to investigate cases where the model:

* Predicts subscription incorrectly
* Misses potential subscribers
* Produces uncertain probability estimates

This analysis helps identify areas where model performance can potentially be improved.

---

## 🌐 Deployment Architecture

The deployed application follows this architecture:

```text
                    ┌─────────────────────┐
                    │      User Input      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Web App   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hugging Face Hub    │
                    │   Trained Model     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Scikit-learn        │
                    │ Preprocessing + RF  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Probability +       │
                    │ Threshold 0.44      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Subscription        │
                    │ Prediction          │
                    └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Category          | Technologies        |
| ----------------- | ------------------- |
| Programming       | Python              |
| Data Processing   | Pandas, NumPy       |
| Machine Learning  | Scikit-learn        |
| Gradient Boosting | XGBoost, LightGBM   |
| Explainability    | SHAP                |
| Visualization     | Matplotlib, Seaborn |
| Model Persistence | Joblib              |
| Web Application   | Streamlit           |
| Model Hosting     | Hugging Face Hub    |
| Version Control   | Git & GitHub        |

---

## 📂 Project Structure

```text
Bank-Marketing-Subscription-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── bank-full.csv
│
├── models/
│   └── bank_marketing_threshold.joblib
│
└── notebooks/
    └── Bank_Marketing_Subscription_Prediction.ipynb
```

> The trained 110 MB model is hosted externally on Hugging Face rather than stored directly in the GitHub repository.

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrhmanAkl/Bank-Marketing-Subscription-Prediction.git
cd Bank-Marketing-Subscription-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will automatically download the trained model from Hugging Face when required.

---

## 📚 Project Goals

This project demonstrates practical experience with:

* End-to-end Machine Learning workflows
* Imbalanced binary classification
* Feature preprocessing
* Ensemble learning
* Hyperparameter optimization
* Cross-validation
* Decision-threshold optimization
* Model explainability
* Error analysis
* Model persistence
* ML deployment
* External model hosting
* Interactive prediction applications

---

## 👨‍💻 Author

**Abdelrahman Ahmed Akl**

AI / Machine Learning Engineer | NLP | LLMs | AI Agents

---

## 🔗 Project Links

* 🚀 **Live Demo:** https://bank-marketing-subscription-prediction-yu3uarppazgwrybswiuxkq.streamlit.app/
* 💻 **GitHub Repository:** https://github.com/AbdelrhmanAkl/Bank-Marketing-Subscription-Prediction
* 🤗 **Hugging Face Model:** https://huggingface.co/AbdelrahmanAkl/bank-marketing-subscription-model
