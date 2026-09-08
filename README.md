# Diabetes Prediction using Support Vector Machine (SVM)

## Overview

This project implements a Support Vector Machine (SVM) based binary classification system for diabetes prediction using the Pima Indians Diabetes Dataset.

The project explores different SVM kernels, evaluates their performance, and analyzes classification thresholds.

## Tasks Completed

### Task I1 — SVM Basics
- Dataset loading
- Missing value handling
- Feature scaling using StandardScaler
- Train-test split (80/20)
- Linear SVM implementation
- Explanation of hyperplane, margin, support vectors, and feature scaling

### Task I2 — Kernelized SVM
Three SVM kernels were compared:
- Linear
- Polynomial
- RBF

### Kernel Performance

| Kernel | Accuracy |
|---|---:|
| Linear SVM | 70.13% |
| Polynomial SVM | 71.43% |
| RBF SVM | 74.03% |

The RBF kernel achieved the highest accuracy.

### Task I3 — Model Evaluation

Best Model: **RBF SVM**

- Accuracy: 74.03%
- Precision: 65.22%
- Recall: 55.56%
- F1-Score: 60.00%
- ROC-AUC: 79.63%

Evaluation includes:
- Confusion Matrix
- ROC Curve
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Task I4 — Decision Threshold Analysis

The SVM decision function is used to investigate how different classification thresholds affect precision and recall.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook

## Project Structure

```text
diabetes--app/
│
├── Task_I_SVM/
│   ├── Notebook/
│   │   └── Task_I1_SVM_Basics.ipynb
│   └── Dataset/
│       └── diabetes.csv
│
├── model/
│   └── svm_model.pkl
│
├── app.py
├── diabetes.csv
├── requirements.txt
├── train_model.py
└── README.md



