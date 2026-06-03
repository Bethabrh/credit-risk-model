# Credit Risk Modeling Project

HEAD
## Overview
This project focuses on building a credit risk modeling pipeline to predict the likelihood of loan default. It applies data preprocessing, exploratory data analysis (EDA), feature engineering, and proxy target creation using RFM clustering principles.

The goal is to support financial risk assessment aligned with Basel II principles.


## Project Structure

- data/ → Raw and processed datasets (not tracked in git)
- notebooks/ → Jupyter notebooks for analysis and modeling
- src/ → Python scripts for data processing and modeling
- models/ → Saved trained models
- reports/ → Visualizations and final reports
- README.md → Project documentation



## Key Components

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis
- Correlation study
- Risk pattern identification

### 3. Proxy Target Creation
- Used RFM-inspired clustering approach
- Defined risk segments for supervised learning

### 4. Modeling
- Logistic Regression baseline
- Tree-based models (Random Forest / XGBoost if used)


## Technologies Used
- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn



## Installation

```bash
pip install -r requirements.txt
```

## 1. Project Overview

This project focuses on building a credit risk analytics system using transaction data. The goal is to analyze customer behavior and prepare a dataset that can later be used to predict the likelihood of fraud or default risk.

The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, and pipeline preparation for machine learning modeling.



## 2. Business Understanding (Credit Risk & Basel II Context)

Credit risk modeling is a key component in financial institutions where the objective is to estimate the probability that a customer will default or behave fraudulently.

This project is aligned with Basel II regulatory principles, which emphasize:

- **Risk Quantification:** Banks must estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).
- **Capital Requirement:** Financial institutions must hold sufficient capital based on risk exposure.
- **Model Interpretability:** Models must be explainable to regulators and stakeholders.

### Proxy Target Consideration
In this dataset, `FraudResult` is used as a proxy target for credit risk behavior. This introduces risk because:
- Fraud ≠ default (not a perfect label)
- It may bias model learning
- Business interpretation must be handled carefully

### Trade-off: Interpretability vs Performance
- Simple models (Logistic Regression) → high interpretability, lower accuracy
- Complex models (Random Forest, XGBoost) → higher performance, lower explainability

For financial systems, interpretability is often prioritized due to regulatory requirements.



## 3. Dataset Description

The dataset contains transaction-level financial data including:

- Customer information (`CustomerId`)
- Transaction details (`Amount`, `Value`)
- Time features (`TransactionStartTime`)
- Category features (`ProductCategory`, `ChannelId`)
- Target variable (`FraudResult`)

Additional engineered features include:
- Recency
- Frequency
- Monetary value (RFM)
- Transaction time breakdown (hour, day, month, year)



## 4. Project Structure
credit-risk-model/
│
├── data/
│ └── raw/
│
├── notebooks/
│ ├── eda.ipynb
│ └── task3_testing.ipynb
│
├── src/
│ ├── data_processing.py
│ ├── feature_engineering.py
│ └── pipeline.py
│
├── requirements.txt
├── README.md
└── .gitignore




## 5. Installation & Setup

bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt
## 6. Workflow Summary
Step 1: Data Understanding
Loaded raw transaction dataset
Checked shape, missing values, and data types
Step 2: Exploratory Data Analysis (EDA)
Distribution analysis of Amount and Value
Category-level analysis (ProductCategory, ProviderId)
Correlation analysis of numerical features
Time-based transaction patterns
Outlier detection
Step 3: Feature Engineering
Customer-level aggregation (RFM features)
Transaction time decomposition
Merging engineered features with dataset
Step 4: Data Preprocessing Pipeline
Missing value imputation
Standard scaling for numerical features
One-hot encoding for categorical features
## 7. Key Insights
Transaction amounts are highly skewed with significant outliers.
A small number of providers dominate transaction activity.
Transactions show strong time-based patterns (hourly behavior).
Product categories are unevenly distributed.
Amount and Value features are highly correlated, indicating redundancy.
## 8. Future Improvements
Train baseline machine learning models (Logistic Regression, Random Forest)
Handle class imbalance (SMOTE or weighting)
Improve feature selection techniques
Add model explainability (SHAP values)
Deploy as an API for real-time scoring
<<<<<<< HEAD
a3722e (Task 3 completed - EDA, feature engineering, and improved project documentation)
=======



# Credit Risk Model API

## Overview
This project builds a machine learning model to predict credit risk and exposes it using FastAPI.


## How to run locally
bash
python -m src.train
uvicorn src.api.main:app --reload
>>>>>>> 3e66cfb (Final Task 6: FastAPI + Docker + CI/CD completed)
