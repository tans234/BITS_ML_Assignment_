# Wine Quality Classification - ML Assignment 2

## a. Problem Statement

Predict whether a wine is of **good quality** (quality score >= 7) or **not good** (quality score < 7) based on its physicochemical properties. This is a binary classification problem using 5 different ML models, with comprehensive evaluation metrics to compare their performance.

## b. Dataset Description

**Dataset**: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from UCI Machine Learning Repository

- **Source**: UCI ML Repository (P. Cortez et al., 2009)
- **Instances**: 6,497 (1,599 red + 4,898 white wines combined)
- **Features**: 12 (11 physicochemical + 1 wine type indicator)
- **Target**: Binary label — Good (quality >= 7) vs Not Good (quality < 7)
- **Class Distribution**: 5,220 Not Good (80.3%) vs 1,277 Good (19.7%) — imbalanced

| Feature | Description |
|---------|-------------|
| fixed acidity | Tartaric acid concentration (g/dm^3) |
| volatile acidity | Acetic acid concentration (g/dm^3) |
| citric acid | Citric acid concentration (g/dm^3) |
| residual sugar | Remaining sugar after fermentation (g/dm^3) |
| chlorides | Sodium chloride concentration (g/dm^3) |
| free sulfur dioxide | Free SO2 concentration (mg/dm^3) |
| total sulfur dioxide | Total SO2 concentration (mg/dm^3) |
| density | Density of wine (g/cm^3) |
| pH | pH value (0-14 scale) |
| sulphates | Potassium sulphate concentration (g/dm^3) |
| alcohol | Alcohol percentage (% vol) |
| wine_type | 0 = Red, 1 = White |

## c. GitHub Repository Link

[GitHub Repository](https://github.com/YOUR_USERNAME/ml-assignment-2)

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8538 | 0.7749 | 0.6250 | 0.6445 | 0.6346 | 0.5434 |
| KNN | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| Naive Bayes | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Random Forest (Ensemble) | 0.8923 | 0.9120 | 0.8333 | 0.5664 | 0.6744 | 0.6291 |

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves decent accuracy (82.2%) but struggles badly with recall (26.2%), meaning it misses most good wines. The linear decision boundary is too simplistic for the non-linear feature interactions in wine chemistry. High precision relative to recall indicates conservative predictions — it rarely predicts "good" unless very confident. |
| Decision Tree | Provides the most balanced precision-recall trade-off (62.5% / 64.5%) among all models. The F1 score (0.6346) reflects this balance. However, it is prone to overfitting on training data, and AUC (0.7749) is the lowest, indicating weaker probability calibration compared to ensemble methods. |
| KNN | Moderate performance across all metrics. Benefits from the scaled feature space but is sensitive to the class imbalance — the majority of neighbors tend to be "Not Good" wines due to the 80/20 split. AUC (0.8264) is reasonable, suggesting the distance-based ranking captures some quality signal. |
| Naive Bayes | Lowest accuracy (73.5%) due to the strong independence assumption being violated — wine features like density and alcohol are correlated. However, it achieves the highest recall (61.7%) among non-ensemble models, making it good at detecting good wines at the cost of many false positives (precision only 39%). |
| Random Forest (Ensemble) | Clear winner across almost all metrics. Highest accuracy (89.2%), AUC (91.2%), precision (83.3%), and MCC (0.6291). The ensemble of 100 trees captures complex feature interactions that single models miss. Recall (56.6%) could be improved with threshold tuning, but the overall balance is the best. |
| **Overall Winner** | **Random Forest** — Dominates in accuracy, AUC, precision, F1, and MCC. Its ensemble approach handles the class imbalance and non-linear feature relationships best. The high AUC (0.912) indicates excellent discriminative ability between good and not-good wines. |

## Project Structure

```
ml-assignment-2/
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── test_data.csv       # Test dataset (20% holdout)
├── test_app.py         # Test script
└── model/
    ├── train_models.py         # Model training script
    ├── scaler.pkl              # Fitted StandardScaler
    ├── feature_names.pkl       # Feature column names
    ├── results.csv             # Evaluation results
    ├── logistic_regression.pkl # Trained LR model
    ├── decision_tree.pkl       # Trained DT model
    ├── knn.pkl                 # Trained KNN model
    ├── naive_bayes.pkl         # Trained NB model
    └── random_forest.pkl       # Trained RF model
```

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates test_data.csv and .pkl files)
python model/train_models.py

# Run the Streamlit app
streamlit run app.py

# Run tests
python test_app.py
```

## Live App

[Streamlit App](https://YOUR_APP_URL.streamlit.app)
