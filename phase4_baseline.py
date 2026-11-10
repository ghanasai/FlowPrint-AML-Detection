
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# FLOWPRINT - PHASE 4
# Step 4.1: Logistic Regression Baseline
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("PHASE 4 - LOGISTIC REGRESSION BASELINE")
print("=" * 55)

train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")
test_df = pd.read_csv(data_dir / "test.csv")

feature_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

X_train = train_df[feature_columns]
y_train = train_df["isMoneyLaundering"]

X_val = val_df[feature_columns]
y_val = val_df["isMoneyLaundering"]

X_test = test_df[feature_columns]
y_test = test_df["isMoneyLaundering"]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# Scale features using training data only
# Handle class imbalance with balanced weights
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# Evaluate validation set first
val_prob = model.predict_proba(X_val)[:, 1]
val_pred = (val_prob >= 0.5).astype(int)

print("\nVALIDATION RESULTS")
print("Precision:", precision_score(y_val, val_pred, zero_division=0))
print("Recall:", recall_score(y_val, val_pred, zero_division=0))
print("F1:", f1_score(y_val, val_pred, zero_division=0))
print("PR-AUC (Average Precision):",
      average_precision_score(y_val, val_prob))
print("ROC-AUC:", roc_auc_score(y_val, val_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred))

print("\nStep 4.1 completed successfully!")



# ==========================================
# Step 4.2: Classification Threshold Tuning
# ==========================================

from sklearn.metrics import precision_recall_curve

print("\n" + "=" * 55)
print("STEP 4.2 - THRESHOLD OPTIMIZATION")
print("=" * 55)

# Compute precision and recall at different thresholds
precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

# Calculate F1 for each threshold
# Last precision/recall pair has no corresponding threshold
p = precision[:-1]
r = recall[:-1]

f1_scores = 2 * p * r / (p + r + 1e-12)

# Find threshold with highest validation F1
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nBest validation threshold:", best_threshold)
print("Best validation F1:", f1_scores[best_index])

# Predict using optimized threshold
optimized_pred = (
    val_prob >= best_threshold
).astype(int)

print("\nOPTIMIZED VALIDATION RESULTS")

print(
    "Precision:",
    precision_score(y_val, optimized_pred, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_val, optimized_pred, zero_division=0)
)

print(
    "F1:",
    f1_score(y_val, optimized_pred, zero_division=0)
)

print("\nOptimized Confusion Matrix:")
print(confusion_matrix(y_val, optimized_pred))

print(
    "\nTotal transactions flagged:",
    optimized_pred.sum()
)

print("\nStep 4.2 completed successfully!")
