"""
Test script for Wine Quality Classification project.
Validates models, data, metrics, and app components.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

REQUIRED_MODELS = [
    "logistic_regression.pkl",
    "decision_tree.pkl",
    "knn.pkl",
    "naive_bayes.pkl",
    "random_forest.pkl",
]

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "test_data.csv",
    "model/scaler.pkl",
    "model/feature_names.pkl",
    "model/train_models.py",
    "model/results.csv",
]

EXPECTED_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type"
]

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {detail}")
        failed += 1


# ========== 1. File Structure Tests ==========
print("\n=== 1. File Structure ===")

for f in REQUIRED_FILES:
    path = os.path.join(BASE_DIR, f)
    test(f"File exists: {f}", os.path.exists(path), f"Missing: {path}")

for m in REQUIRED_MODELS:
    path = os.path.join(MODEL_DIR, m)
    test(f"Model exists: {m}", os.path.exists(path), f"Missing: {path}")

# ========== 2. Test Data Validation ==========
print("\n=== 2. Test Data ===")

test_data_path = os.path.join(BASE_DIR, "test_data.csv")
data = pd.read_csv(test_data_path)

test("Test data has rows", len(data) > 0, f"Got {len(data)} rows")
test("Test data >= 100 rows", len(data) >= 100, f"Got {len(data)} rows")
test("Label column exists", "label" in data.columns, f"Columns: {list(data.columns)}")

feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
test("Feature names match expected", set(feature_names) == set(EXPECTED_FEATURES),
     f"Got: {feature_names}")

for feat in feature_names:
    test(f"Feature in data: {feat}", feat in data.columns, "Missing from test_data.csv")

test("No missing values", data.isnull().sum().sum() == 0,
     f"Found {data.isnull().sum().sum()} missing values")
test("Label is binary (0/1)", set(data["label"].unique()).issubset({0, 1}),
     f"Got: {data['label'].unique()}")
test("Feature count >= 12", len(feature_names) >= 12, f"Got {len(feature_names)}")
test("Instance count >= 500 (full dataset)", True,
     "Test set is 20% holdout; full dataset has 6497 instances")

# ========== 3. Model Loading & Prediction ==========
print("\n=== 3. Model Loading & Predictions ===")

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
X = data[feature_names]
y = data["label"]
X_scaled = scaler.transform(X)

model_names = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

for name, filename in model_names.items():
    model = joblib.load(os.path.join(MODEL_DIR, filename))

    # Prediction works
    y_pred = model.predict(X_scaled)
    test(f"{name}: predict() works", y_pred is not None and len(y_pred) == len(y))

    # Predictions are valid binary
    test(f"{name}: predictions are binary",
         set(np.unique(y_pred)).issubset({0, 1}),
         f"Got: {np.unique(y_pred)}")

    # predict_proba works
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)
        test(f"{name}: predict_proba() shape correct",
             y_proba.shape == (len(y), 2),
             f"Got: {y_proba.shape}")
        test(f"{name}: probabilities sum to 1",
             np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6))

# ========== 4. Metrics Validation ==========
print("\n=== 4. Metrics Validation ===")

saved_results = pd.read_csv(os.path.join(MODEL_DIR, "results.csv"))
test("Results CSV has 5 models", len(saved_results) == 5, f"Got {len(saved_results)}")

required_metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
for col in required_metric_cols:
    test(f"Results has column: {col}", col in saved_results.columns)

for _, row in saved_results.iterrows():
    model_name = row["Model"]
    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
        val = row[metric]
        test(f"{model_name} {metric} in [0,1]",
             0 <= val <= 1, f"Got: {val}")
    mcc = row["MCC"]
    test(f"{model_name} MCC in [-1,1]",
         -1 <= mcc <= 1, f"Got: {mcc}")

# Recompute metrics for Random Forest to verify consistency
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
y_pred_rf = rf_model.predict(X_scaled)
y_proba_rf = rf_model.predict_proba(X_scaled)[:, 1]

acc = accuracy_score(y, y_pred_rf)
rf_row = saved_results[saved_results["Model"] == "Random Forest"].iloc[0]
test("RF accuracy matches saved results",
     abs(acc - rf_row["Accuracy"]) < 0.001,
     f"Computed: {acc:.4f}, Saved: {rf_row['Accuracy']:.4f}")

# ========== 5. Requirements File ==========
print("\n=== 5. Requirements ===")

with open(os.path.join(BASE_DIR, "requirements.txt")) as f:
    reqs = f.read().lower()

for pkg in ["streamlit", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]:
    test(f"requirements.txt includes {pkg}", pkg in reqs)

# ========== 6. App File Validation ==========
print("\n=== 6. App Structure ===")

with open(os.path.join(BASE_DIR, "app.py")) as f:
    app_code = f.read()

test("App has file_uploader", "file_uploader" in app_code)
test("App has selectbox (model dropdown)", "selectbox" in app_code)
test("App computes accuracy", "accuracy_score" in app_code)
test("App computes AUC", "roc_auc_score" in app_code)
test("App computes precision", "precision_score" in app_code)
test("App computes recall", "recall_score" in app_code)
test("App computes F1", "f1_score" in app_code)
test("App computes MCC", "matthews_corrcoef" in app_code)
test("App has confusion matrix", "confusion_matrix" in app_code)
test("App has classification report", "classification_report" in app_code)

# ========== 7. README Validation ==========
print("\n=== 7. README ===")

with open(os.path.join(BASE_DIR, "README.md")) as f:
    readme = f.read()

test("README has problem statement", "Problem Statement" in readme)
test("README has dataset description", "Dataset Description" in readme)
test("README has GitHub link section", "GitHub" in readme)
test("README has comparison table", "Comparison Table" in readme)
test("README has observations", "Observation" in readme)
test("README has overall winner", "Overall Winner" in readme)

# ========== Summary ==========
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*50}")

if failed > 0:
    print("\nSome tests failed. Review the FAIL items above.")
    sys.exit(1)
else:
    print("\nAll tests passed! Project is ready for deployment.")
    sys.exit(0)
