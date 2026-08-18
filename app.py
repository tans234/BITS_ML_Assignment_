"""
Streamlit app for Wine Quality Classification.
Demonstrates 5 ML models with evaluation metrics and confusion matrices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_models():
    """Load all trained models, scaler, and feature names."""
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return models, scaler, feature_names


def compute_metrics(y_true, y_pred, y_proba):
    """Compute all 6 evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Good", "Good"],
                yticklabels=["Not Good", "Good"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    return fig


# --- Main App ---
st.title("Wine Quality Classification")
st.markdown(
    "Compare 5 ML classification models on the Wine Quality dataset. "
    "Upload test data (CSV) or use the default test set."
)

# Load models
try:
    models, scaler, feature_names = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please run `python model/train_models.py` first.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Settings")

# CSV Upload
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(data)} rows)")
else:
    default_path = os.path.join(BASE_DIR, "test_data.csv")
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.sidebar.info(f"Using default test data ({len(data)} rows)")
    else:
        st.error("No test data found. Upload a CSV or run train_models.py first.")
        st.stop()

# Validate data has required columns
if "label" not in data.columns:
    st.error("CSV must contain a 'label' column (target variable).")
    st.stop()

missing_features = [f for f in feature_names if f not in data.columns]
if missing_features:
    st.error(f"CSV is missing features: {missing_features}")
    st.stop()

X_data = data[feature_names]
y_data = data["label"]
X_scaled = scaler.transform(X_data)

# Model selection
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.write(f"Samples: {len(data)}")
st.sidebar.write(f"Features: {len(feature_names)}")
st.sidebar.write(f"Class 0 (Not Good): {(y_data == 0).sum()}")
st.sidebar.write(f"Class 1 (Good): {(y_data == 1).sum()}")

# --- All Models Comparison ---
st.header("All Models Comparison")

all_results = []
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
    metrics = compute_metrics(y_data, y_pred, y_proba)
    metrics["Model"] = name
    all_results.append(metrics)

results_df = pd.DataFrame(all_results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]

# Format numeric columns
styled_df = results_df.style.format({
    "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
    "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
}).highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#90EE90")

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# Best model
best_idx = results_df["F1"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
st.success(f"Best model by F1 Score: **{best_model_name}** ({results_df.loc[best_idx, 'F1']:.4f})")

# --- Selected Model Details ---
st.header(f"Selected Model: {selected_model}")

model = models[selected_model]
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
metrics = compute_metrics(y_data, y_pred, y_proba)

# Metrics in columns
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("AUC", f"{metrics['AUC']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")
col4.metric("Recall", f"{metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{metrics['F1']:.4f}")
col6.metric("MCC", f"{metrics['MCC']:.4f}")

# Confusion Matrix and Classification Report side by side
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(y_data, y_pred, selected_model)
    st.pyplot(fig)
    plt.close(fig)

with col_right:
    st.subheader("Classification Report")
    report = classification_report(y_data, y_pred, target_names=["Not Good", "Good"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# --- Dataset Preview ---
with st.expander("Preview Test Data"):
    st.dataframe(data.head(20), use_container_width=True)
