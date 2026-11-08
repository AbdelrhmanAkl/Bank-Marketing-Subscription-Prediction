# Bank Marketing Subscription Prediction

A machine learning system that estimates how likely a bank customer is to subscribe to a term deposit, so marketing teams can focus calls on the customers most worth contacting.

The project covers the full workflow: exploratory analysis, imbalanced-classification modeling, hyperparameter tuning, decision-threshold optimization, explainability, error analysis, and deployment as an interactive web application.

**Live application:** https://bank-marketing-subscription-prediction-yu3uarppazgwrybswiuxkq.streamlit.app/
**Trained model:** https://huggingface.co/AbdelrahmanAkl/bank-marketing-subscription-model

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Application](#application)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Explainability and Error Analysis](#explainability-and-error-analysis)
7. [Architecture](#architecture)
8. [Project Structure](#project-structure)
9. [Getting Started](#getting-started)
10. [Configuration](#configuration)
11. [Limitations](#limitations)
12. [Technology Stack](#technology-stack)
13. [Author](#author)

---

## Business Problem

Term deposit campaigns contact a large number of customers, and only a small share of them subscribe. Contacting everyone wastes agent time and annoys customers who will never convert.

This project ranks customers by their probability of subscribing. The marketing team can then:

- Prioritize high-potential customers
- Reduce unnecessary contacts
- Balance outreach cost against expected conversions
- Adjust campaign aggressiveness by changing a single decision threshold

Because only about 12% of customers subscribe, accuracy is a misleading metric. A model that always predicts "no" would be 88% accurate and useless. Evaluation therefore centers on F1-score, precision, recall, and ROC-AUC.

---

## Application

The Streamlit application provides three views.

**Single customer.** Enter a customer profile and receive the subscription probability, a lead tier (Hot lead, Qualified lead, Nurture, Low priority), and a recommended next step. Example profiles can be loaded to demonstrate the tool quickly.

**Batch scoring.** Upload a CSV file of customers. The application scores every row, ranks customers from most to least likely to subscribe, summarizes the results, and offers the ranked list as a CSV download. A template file with the expected columns can be downloaded from the page.

**About the model.** A short explanation of how the model works and its held-out test performance.

The decision threshold can be changed from the sidebar. A lower threshold reaches more potential subscribers at the cost of more wasted contacts; a higher threshold concentrates effort on the most certain leads.

Branding (product name, tagline, support email) is configured by constants at the top of `app.py`.

---

## Dataset

The project uses the Bank Marketing dataset (`bank-full.csv`).

| Property | Value |
| --- | --- |
| Customers | 45,211 |
| Input features | 16 |
| Target | Binary: subscribed to a term deposit or not |
| Feature types | Numerical and categorical |

### Target distribution

| Class | Count | Share |
| --- | ---: | ---: |
| No subscription | 39,922 | 88.3% |
| Subscription | 5,289 | 11.7% |

### Input features

| Group | Features |
| --- | --- |
| Personal | age, job, marital, education |
| Financial | default, balance, housing, loan |
| Last contact | contact, day, month, duration |
| Campaign history | campaign, pdays, previous, poutcome |

---

## Methodology

### Workflow

1. Exploratory data analysis
2. Data quality analysis
3. Stratified train, validation, and test split
4. Preprocessing of numerical and categorical features
5. Baseline modeling
6. Model benchmarking
7. Hyperparameter tuning
8. Stratified cross-validation
9. Decision-threshold optimization
10. Final model selection
11. Explainability with SHAP
12. Error analysis
13. Model persistence
14. Deployment

### Preprocessing

| Feature type | Steps |
| --- | --- |
| Numerical | Median imputation, standard scaling |
| Categorical | Most-frequent imputation, one-hot encoding with unknown-category handling |

Preprocessing and the classifier are combined in a single scikit-learn `Pipeline`, so the same transformations are applied during training and inference. After encoding, the feature matrix has 50 columns.

### Models evaluated

- Logistic Regression
- Class-weighted Logistic Regression
- Random Forest
- Tuned Random Forest
- HistGradientBoosting
- XGBoost

Stratified 5-fold cross-validation was used where applicable.

### Hyperparameter tuning

`RandomizedSearchCV` with stratified 5-fold cross-validation was used to tune the Random Forest.

| Parameter | Value |
| --- | --- |
| class_weight | balanced_subsample |
| max_depth | None |
| max_features | 0.7 |
| min_samples_leaf | 7 |
| min_samples_split | 10 |
| n_estimators | 527 |

Best cross-validation F1: 0.4693

---

## Results

### Model comparison (validation set)

| Model | F1 | ROC-AUC | CV mean F1 |
| --- | ---: | ---: | ---: |
| **Tuned Random Forest** | **0.4918** | 0.7965 | **0.4693** |
| HistGradientBoosting | 0.3367 | 0.8054 | 0.3325 |
| XGBoost | 0.3327 | **0.8077** | 0.3317 |
| Random Forest | 0.3233 | 0.7966 | 0.3000 |
| Logistic Regression | 0.2843 | 0.7699 | not computed |

The Tuned Random Forest was selected because it has the strongest F1-score and cross-validated F1. XGBoost reaches a slightly higher ROC-AUC, but its F1 is far lower, and the project objective is to balance precision and recall on the minority class.

### Decision threshold

The default threshold of 0.50 was replaced by a value tuned on the validation set.

| Setting | Threshold |
| --- | ---: |
| Default | 0.50 |
| Selected | **0.44** |

Validation performance at the selected threshold:

| Metric | Score |
| --- | ---: |
| Accuracy | 86.52% |
| Precision | 44.09% |
| Recall | 56.42% |
| F1-score | 49.50% |
| ROC-AUC | 79.65% |

### Final model on the held-out test set

| Metric | Score |
| --- | ---: |
| F1-score | 47.86% |
| ROC-AUC | 78.32% |

The small drop from validation to test performance is expected and indicates the model generalizes without large overfitting.

---

## Explainability and Error Analysis

SHAP (SHapley Additive exPlanations) was used to identify which features drive predictions and in which direction. This makes the model's behavior easier to audit and to explain to business stakeholders.

Error analysis examined three groups of cases: customers incorrectly predicted to subscribe, potential subscribers the model missed, and predictions with uncertain probabilities. The findings guide where future feature engineering and data collection are most likely to help.

---

## Architecture

```text
User input (form or CSV upload)
        |
        v
Streamlit application
        |
        v
Hugging Face Hub  --->  trained model (.joblib), cached after first load
        |
        v
scikit-learn Pipeline (preprocessing + Random Forest)
        |
        v
Subscription probability  --->  compared with the decision threshold
        |
        v
Decision, lead tier, and recommended next step
```

The trained model is about 110 MB. It is hosted on Hugging Face instead of GitHub, which keeps the repository small, and the application downloads it once at startup.

---

## Project Structure

```text
Bank-Marketing-Subscription-Prediction/
|-- app.py
|-- README.md
|-- requirements.txt
|-- .streamlit/
|   `-- config.toml
|-- data/
|   `-- bank-full.csv
|-- models/
|   `-- bank_marketing_threshold.joblib
`-- notebooks/
    `-- Bank_Marketing_Subscription_Prediction.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Internet access on first run, to download the model

### Installation

```bash
git clone https://github.com/AbdelrhmanAkl/Bank-Marketing-Subscription-Prediction.git
cd Bank-Marketing-Subscription-Prediction

python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS and Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run the application:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### Batch file format

The uploaded CSV must contain these 16 columns, with the same category values used in training (for example, `job` values such as `admin.` or `blue-collar`, and lowercase three-letter month names):

```text
age, job, marital, education, default, balance, housing, loan,
contact, day, month, duration, campaign, pdays, previous, poutcome
```

A ready-made template is available inside the application.

---

## Configuration

| Setting | Location | Purpose |
| --- | --- | --- |
| `BRAND_NAME`, `BRAND_TAGLINE`, `SUPPORT_EMAIL` | top of `app.py` | White-label the interface |
| `MODEL_REPO`, `MODEL_FILE` | top of `app.py` | Point to a different hosted model |
| `DEFAULT_THRESHOLD` | top of `app.py` | Initial decision threshold |
| Theme colors | `.streamlit/config.toml` | Streamlit accent and background colors |

---

## Limitations

- **Call duration is only known after the call.** The `duration` feature strongly predicts the outcome but is not available before a customer is contacted. The model is therefore best used to rank customers who have already had contact, or as a basis for retraining a pre-call variant that excludes this feature.
- **Moderate precision.** At the selected threshold, roughly half of the customers flagged as likely subscribers will not subscribe. The model supports prioritization and should not be the only basis for a decision about a customer.
- **Single-market data.** The dataset comes from one bank's historical campaigns. Performance on a different market, period, or product should be validated before relying on it.
- **Static model.** Customer behavior drifts over time. The model should be monitored and retrained periodically on recent campaign data.

---

## Technology Stack

| Category | Tools |
| --- | --- |
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| Model hosting | Hugging Face Hub |
| Web application | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version control | Git and GitHub |

---

## Author

**Abdelrahman Ahmed Akl**
AI and Machine Learning Engineer, focused on NLP, LLMs, and AI agents.

- GitHub: https://github.com/AbdelrhmanAkl/Bank-Marketing-Subscription-Prediction
- Hugging Face: https://huggingface.co/AbdelrahmanAkl/bank-marketing-subscription-model
- Live application: https://bank-marketing-subscription-prediction-yu3uarppazgwrybswiuxkq.streamlit.app/
