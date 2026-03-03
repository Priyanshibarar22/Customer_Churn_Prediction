#  Customer Churn Prediction — Telecom Industry



##  Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [EDA Findings](#-exploratory-data-analysis--key-findings)
- [Preprocessing](#-preprocessing-steps)
- [Model Training](#-model-training--hyperparameter-tuning)
- [Results](#-results)
- [Business Insights](#-business-insights--recommendations)
- [Future Implementation](#-future-scope)


---

##  Overview

Customer churn — when a customer stops using a company's service — is one of the most costly challenges in the telecom industry.

> **Acquiring a new customer costs 5–7× more than retaining an existing one.**

This project builds a complete, production-grade **machine learning pipeline** to predict whether a telecom customer will churn. By identifying at-risk customers before they leave, telecom companies can deploy targeted, cost-effective retention strategies that protect revenue and improve customer lifetime value.

---

##  Problem Statement

> Given a telecom customer's demographic profile, account information, and service usage data — **predict whether the customer will churn (Yes / No).**

This is a **supervised binary classification** problem with significant **class imbalance** (~73% No Churn, ~27% Churn).

---

##  Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Total Records** | 7,043 customers |
| **Total Features** | 21 columns |
| **Target Variable** | `Churn` — Yes / No |
| **Class Distribution** | ~73.5% No Churn · ~26.5% Churn |
| **Missing Values** | 11 hidden whitespace entries in `TotalCharges` |

###  Feature Dictionary

| # | Feature | Type | Description |
|---|---|---|---|
| 0 | customerID | Object | Unique ID — dropped (no predictive value) |
| 1 | gender | Object | Male / Female |
| 2 | SeniorCitizen | Int64 | 1 = Senior, 0 = Not senior |
| 3 | Partner | Object | Has a partner — Yes/No |
| 4 | Dependents | Object | Has dependents — Yes/No |
| 5 | tenure | Int64 | Months with the company (0–72) |
| 6 | PhoneService | Object | Has phone service — Yes/No |
| 7 | MultipleLines | Object | Has multiple lines |
| 8 | InternetService | Object | DSL / Fiber optic / No |
| 9 | OnlineSecurity | Object | Has online security add-on |
| 10 | OnlineBackup | Object | Has online backup add-on |
| 11 | DeviceProtection | Object | Has device protection |
| 12 | TechSupport | Object | Has tech support |
| 13 | StreamingTV | Object | Streams TV |
| 14 | StreamingMovies | Object | Streams movies |
| 15 | Contract | Object | Month-to-month / One year / Two year |
| 16 | PaperlessBilling | Object | Uses paperless billing — Yes/No |
| 17 | PaymentMethod | Object | Electronic check / Mailed check / Bank transfer / Credit card |
| 18 | MonthlyCharges | Float64 | Monthly bill amount ($18.25–$118.75) |
| 19 | TotalCharges | Object→Float | Total charges (required type conversion) |
| 20 | **Churn** | **Object** | **Target: Yes / No** |

---

##  Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Environment** | Google Colab |
| **Data Manipulation** | Pandas 2.0, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Preprocessing** | Scikit-learn (StandardScaler, train_test_split) |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Modeling** | Scikit-learn (Logistic Regression, Random Forest) |
| **Gradient Boosting** | XGBoost |
| **Hyperparameter Tuning** | GridSearchCV (5-fold CV) |
| **Evaluation** | ROC-AUC, Confusion Matrix, Classification Report |



---

##  ML Pipeline

```
┌──────────────────────────────────────────┐
│            RAW DATA                      │
│    Telco_Customer_Churn.csv              │
│    7,043 rows × 21 columns               │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│     EXPLORATORY DATA ANALYSIS (EDA)      │
│  Churn dist · Tenure · MonthlyCharges    │
│  Contract type · Correlation Heatmap     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          DATA PREPROCESSING              │
│  Drop customerID · Fix TotalCharges      │
│  Train-Test Split (80/20)                │
│  One-Hot Encoding · StandardScaler       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│       CLASS IMBALANCE — SMOTE            │
│    73:27 ratio → 50:50 balanced          │
│    5,634 → 8,276 training samples        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│   MODEL TRAINING + GridSearchCV (cv=5)   │
│  Logistic Regression · Random Forest     │
│  XGBoost Classifier                      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│           MODEL EVALUATION               │
│  ROC Curve · Confusion Matrix            │
│  Classification Report · Feature Imp.   │
└──────────────────────────────────────────┘
```

---

##  Exploratory Data Analysis — Key Findings

###  Churn Distribution
- **73.5% No Churn** vs **26.5% Churn** — significant class imbalance
- Addressed using **SMOTE** on training data only

###  Tenure vs Churn
- Customers with **tenure < 10 months** show the **highest churn rate**
- Churn drops sharply beyond 24 months
- **The first 12 months is the most critical retention window**

###  MonthlyCharges vs Churn
- Churners median: **~$79/month** vs Non-churners: **~$61/month**
- High-paying customers feel poor value — more likely to switch

###  Contract Type vs Churn

| Contract Type | Approx. Churn Rate |
|---|---|
| Month-to-month | ~43%  |
| One year | ~11%  |
| Two year | ~3%  |

###  Correlation Heatmap

| Feature Pair | Correlation |
|---|---|
| tenure ↔ TotalCharges | **0.83** Strong |
| MonthlyCharges ↔ TotalCharges | **0.65** Moderate |
| tenure ↔ MonthlyCharges | **0.25** Weak |

---

##  Preprocessing Steps

| Step | Method | Reason |
|---|---|---|
| Drop customerID | `df.drop("customerID")` | No predictive value |
| Fix TotalCharges | `pd.to_numeric(..., errors='coerce')` | Stored as object with whitespace |
| Fill NaN | `.fillna(0)` | 11 blank TotalCharges entries |
| Train-Test Split | 80/20, `random_state=42` | Reproducibility |
| One-Hot Encoding | `pd.get_dummies(drop_first=True)` | Encode 18 categorical columns |
| StandardScaler | `fit_transform` on train; `transform` on test | Normalize, prevent leakage |
| SMOTE | On training set only | Balance 73:27 → 50:50 |

---

##  Model Training & Hyperparameter Tuning

### GridSearchCV Setup
- Cross-validation: **5-fold**
- Scoring: **ROC-AUC**
- Training data: **SMOTE-balanced (8,276 samples)**

### Best Parameters Found

| Model | Best Parameters |
|---|---|
| Logistic Regression | `C=10, solver='liblinear'` |
| Random Forest | `max_depth=None, min_samples_split=2, n_estimators=200` |
| XGBoost | `learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8` |

---

##  Results

### ROC-AUC Scores

| Rank | Model | ROC-AUC |
|---|---|---|
| 1st | **Logistic Regression** | **0.8620** |
| 2nd | XGBoost Classifier | 0.8497 |
| 3rd | Random Forest | 0.8447 |

### Confusion Matrix — Logistic Regression

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** |  756 (TN) |  280 (FP) |
| **Actual: Churn** |  64 (FN) |  309 (TP) |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — No Churn | 0.92 | 0.73 | 0.81 | 1,036 |
| **1 — Churn** | **0.52** | **0.83** | **0.64** | **373** |
| **Accuracy** | | | **0.76** | **1,409** |
| Weighted Avg | 0.82 | 0.76 | 0.77 | 1,409 |

###  Top Features (by Logistic Regression Coefficients)

| Rank | Feature | Meaning |
|---|---|---|
| 1 | MonthlyCharges | Higher bills → higher churn risk |
| 2 | tenure | Newer customers → higher churn risk |
| 3 | InternetService_Fiber optic | Fiber users churn more |
| 4 | TotalCharges | Correlates with tenure |
| 5 | Contract_Two year | Strongly prevents churn |

---

##  Business Insights & Recommendations

| Finding | Action |
|---|---|
| New customers (< 12 months) churn most | Launch **onboarding loyalty programs** in Months 1–6 |
| High MonthlyCharges → higher churn | Offer **personalized discounts** to top-billing customers |
| Month-to-month → 43% churn | Incentivize upgrades to **annual contracts** |
| Fiber optic users churn more | Investigate **service quality issues** for this segment |
| Two-year contracts → ~3% churn | Promote **long-term plan sign-up bonuses** |

### Estimated Business Impact
> On 10,000 customers (2,700 at-risk with 27% churn rate):
> - Model catches ~**2,241 churners** (83% recall)
> - Retain 30% of those = **~672 customers saved**
> - At $65/month average = **~$43,680/month in protected revenue**

---

##  Future Implementation

### 1.  Advanced Modeling
- **ANN (Artificial Neural Networks)** via TensorFlow/Keras
- **LightGBM** and **CatBoost** for gradient boosting comparison
- **Stacking/Voting Ensemble** combining all three models

### 2.  Feature Engineering
- Interaction features: `tenure × MonthlyCharges`, `charges_per_month_ratio`
- Behavioral data: call logs, data usage, complaint history
- **PCA** for dimensionality reduction

### 3.  Imbalance Handling Improvements
- Compare with **ADASYN**, **Borderline-SMOTE**, cost-sensitive learning
- Tune `class_weight='balanced'` as an alternative to SMOTE

### 4.  Model Explainability (XAI)
- **SHAP values** for per-customer transparent predictions
- **LIME** for local interpretability
- **Streamlit / Gradio dashboard** for stakeholder demos

### 5.  Deployment
- **FastAPI / Flask REST API** for real-time predictions
- **Streamlit web app** with live churn score calculator
- **Docker** containerization + **AWS / GCP / Azure** cloud deployment
- **MLflow** for experiment tracking and model versioning

### 6.  Real-Time Pipeline
- Time-series features to capture behavioral drift
- **Apache Kafka + Spark Streaming** for real-time scoring
- Automated retraining on a rolling window to handle concept drift

### 7.  Business Integration
- **Customer Lifetime Value (CLV)** model to prioritize interventions
- CRM integration (**Salesforce / HubSpot**) for automated alerts
- **A/B testing framework** to measure retention campaign ROI
- **Churn Risk Scoring Dashboard** for customer success teams



---

##  References

- [Kaggle — Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)


---



