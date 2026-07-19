# Credit Risk Prediction

## Overview
Developed a machine learning model to predict whether a borrower is likely to experience a serious delinquency (default) within the next two years using financial and repayment history data.

## Problem Statement
Financial institutions need to identify high-risk borrowers before approving loans. The objective of this project is to classify borrowers into:

- **0** → Low Risk (Unlikely to default)
- **1** → High Risk (Likely to default)

## Dataset
- **Dataset:** Credit Risk Benchmark Dataset
- **Task:** Binary Classification
- **Target Variable:** `dlq_2yrs`

The dataset contains borrower information such as:
- Credit utilization
- Debt ratio
- Monthly income
- Payment history
- Open credit lines
- Real estate loans
- Number of dependents

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Missing Value Analysis
3. Outlier Treatment using 99th Percentile Capping
4. Correlation Analysis
5. Feature Selection using ANOVA F-test
6. Train-Test Split
7. Model Training
8. Model Evaluation

## Models Implemented

- Logistic Regression
- Logistic Regression (Reduced Features)
- Logistic Regression with PCA
- L1 Logistic Regression
- L2 Logistic Regression
- Elastic Net Logistic Regression
- Decision Tree
- Random Forest

## Best Model

**Random Forest**

Performance:
- **Accuracy:** 79%
- **ROC-AUC Score:** 0.86

The Random Forest model provided the best balance between classification performance and generalization, making it the final selected model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Repository Structure

```
├── Credit Risk Benchmark Dataset.csv
├── credit risk.ipynb
└── README.md
```

## Key Learnings

- Exploratory Data Analysis
- Data Cleaning
- Outlier Treatment
- Feature Selection
- Classification Algorithms
- Model Evaluation
- ROC-AUC Interpretation
- Credit Risk Modeling


- Handle class imbalance using SMOTE
- Experiment with XGBoost and LightGBM
- Deploy the model as a web application using Streamlit or Flask
