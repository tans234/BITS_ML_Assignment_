"""
Train all 5 classification models on the Wine Quality dataset.
Saves trained models as .pkl files and generates test_data.csv.
"""

import os
import ssl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix SSL for macOS Python
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import joblib

# --- Load and prepare dataset ---
def load_wine_data():
    """Load wine quality dataset from UCI repository."""
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red = pd.read_csv(red_url, sep=";")
    white = pd.read_csv(white_url, sep=";")

    red["wine_type"] = 0  # red
    white["wine_type"] = 1  # white

    df = pd.concat([red, white], ignore_index=True)

    # Binary classification: quality >= 7 is "good" (1), else "not good" (0)
    df["label"] = (df["quality"] >= 7).astype(int)
    df.drop("quality", axis=1, inplace=True)

    return df


def train_and_evaluate():
    print("Loading dataset...")
    df = load_wine_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df.to_csv(os.path.join(base_dir, "test_data.csv"), index=False)
    print(f"Test data saved: {len(test_df)} rows\n")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    model_dir = os.path.join(base_dir, "model")
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    # Save feature names and column order
    joblib.dump(list(X.columns), os.path.join(model_dir, "feature_names.pkl"))

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(model_dir, f"{safe_name}.pkl"))

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Model": name, "Accuracy": acc, "AUC": auc,
            "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
        })

        print(f"  Accuracy={acc:.4f}  AUC={auc:.4f}  Precision={prec:.4f}  "
              f"Recall={rec:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    # Print comparison table
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison Table ===")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(os.path.join(model_dir, "results.csv"), index=False)
    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    train_and_evaluate()
