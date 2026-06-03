# Credit Risk Modeling Project

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

bash
pip install -r requirements.txt