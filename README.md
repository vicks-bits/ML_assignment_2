a) Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related features.

This is a binary classification problem, where the target variable is:

<=50K

>50K

The project also includes building and deploying an interactive Streamlit web application for real-time model testing and evaluation.

b) Dataset Description

The dataset used is the Adult Income Dataset from the UCI Machine Learning Repository.

Dataset Characteristics:

Total Instances: 48,842

Total Features: 14

Target Variable: Income (>50K or <=50K)

Problem Type: Binary Classification

Feature Examples:

Age

Workclass

Education

Marital Status

Occupation

Relationship

Race

Sex

Capital Gain

Capital Loss

Hours per Week

Native Country

The dataset was preprocessed by:

Handling missing values

Encoding categorical variables

Scaling numerical features

Splitting into train-test sets

The dataset was directly read from the UCI online repository (no local download), ensuring reproducibility.

c) Models Used

    The following six classification models were implemented on the same dataset:
    
    Logistic Regression
    
    Decision Tree Classifier
    
    K-Nearest Neighbor (KNN)
    
    Naive Bayes (Gaussian)
    
    Random Forest (Ensemble Model)
    
    XGBoost (Ensemble Model)

Model Comparison Table:

| ML Model Name            | Accuracy   | AUC        | Precision  | Recall     | F1 Score   | MCC        |
| ------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression      | 0.8231     | 0.8598     | 0.7445     | 0.4608     | 0.5692     | 0.4868     |
| Decision Tree            | 0.8087     | 0.7489     | 0.6218     | 0.6275     | 0.6246     | 0.4963     |
| KNN                      | 0.8258     | 0.8585     | 0.6768     | 0.5993     | 0.6357     | 0.5234     |
| Naive Bayes              | 0.7984     | 0.8595     | 0.7099     | 0.3471     | 0.4662     | 0.3946     |
| Random Forest (Ensemble) | 0.8530     | 0.9069     | 0.7419     | 0.6444     | 0.6898     | 0.5966     |
| XGBoost (Ensemble)       | **0.8696** | **0.9277** | **0.7718** | **0.6895** | **0.7283** | **0.6446** |

Observations on Model Performance:

| ML Model Name            | Observation about Model Performance                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performs well overall with strong AUC, but relatively low recall indicates difficulty capturing all high-income individuals.                              |
| Decision Tree            | Balanced precision and recall, but lower AUC suggests weaker probability calibration and possible overfitting.                                            |
| KNN                      | Good balance between precision and recall; benefits from feature scaling; moderate improvement over single-tree model.                                    |
| Naive Bayes              | High AUC but low recall and F1 score; independence assumption limits its effectiveness on correlated census features.                                     |
| Random Forest (Ensemble) | Significant performance improvement over individual models; strong AUC and MCC indicate robust generalization.                                            |
| XGBoost (Ensemble)       | Best performing model across all metrics; highest Accuracy, AUC, F1, and MCC; demonstrates superior handling of feature interactions and class imbalance. |

