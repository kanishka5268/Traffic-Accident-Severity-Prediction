# Random Forest Classification with SMOTE and Hyperparameter Tuning

## Overview

This project implements a **Random Forest Classifier** for predicting the **Severity** class using a tabular dataset. The workflow addresses class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)** and improves model performance through **RandomizedSearchCV** for hyperparameter tuning.

---

## Features

- Data preprocessing
- Train-test split
- Class balancing using SMOTE
- Random Forest Classification
- Hyperparameter tuning with RandomizedSearchCV
- Model evaluation using:
  - Accuracy
  - Macro Recall
  - Classification Report
  - Confusion Matrix
- Feature Importance Analysis
- ROC Curve Visualization

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## Source

https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

```
HogayaFinally.csv
```

**Target Column**

```
Severity
```

All remaining columns are treated as input features.

---

## Workflow

1. Load the dataset.
2. Split features and target variable.
3. Perform train-test split.
4. Apply SMOTE to balance the training dataset.
5. Train a Random Forest classifier.
6. Tune hyperparameters using RandomizedSearchCV.
7. Evaluate the optimized model.
8. Visualize feature importance, confusion matrix, and ROC curves.

---

## Output

The notebook generates:

- Best Hyperparameters
- Cross-validation Recall
- Accuracy Score
- Macro Recall
- Classification Report
- Confusion Matrix
- Feature Importance Plot
- ROC Curves

---

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

---

## Usage

1. Clone this repository.

```bash
git clone https://github.com/kanishka5268/AI-ML.git
```

2. Open the notebook in Google Colab or Jupyter Notebook.

3. Place the dataset (`HogayaFinally.csv`) in the working directory.

4. Run all notebook cells sequentially.

---

## Future Improvements

- Compare multiple machine learning algorithms.
- Add SHAP explainability.
- Perform feature selection.
- Improve model performance using ensemble techniques.
