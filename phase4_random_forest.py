
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)

# ==========================================
# FLOWPRINT - PHASE 4
# Step 4.3: Random Forest Baseline
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 4.3 - RANDOM FOREST BASELINE")
print("=" * 55)

# Load only the required columns
features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

columns = features + ["isMoneyLaundering"]

train_df = pd.read_csv(
    data_dir / "train.csv",
    usecols=columns
)

val_df = pd.read_csv(
    data_dir / "validation.csv",
    usecols=columns
)

X_train = train_df[features]
y_train = train_df["isMoneyLaundering"]

X_val = val_df[features]
y_val = val_df["isMoneyLaundering"]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# Random Forest handles nonlinear relationships
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Predict validation probabilities
val_prob = model.predict_proba(X_val)[:, 1]

print("\nRANKING METRICS")
print(
    "PR-AUC (Average Precision):",
    average_precision_score(y_val, val_prob)
)
print(
    "ROC-AUC:",
    roc_auc_score(y_val, val_prob)
)

# Optimize F1 threshold using validation only
precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

p = precision[:-1]
r = recall[:-1]

f1_scores = 2 * p * r / (p + r + 1e-12)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

val_pred = (val_prob >= best_threshold).astype(int)

print("\nOPTIMIZED VALIDATION RESULTS")
print("Best threshold:", best_threshold)
print(
    "Precision:",
    precision_score(y_val, val_pred, zero_division=0)
)
print(
    "Recall:",
    recall_score(y_val, val_pred, zero_division=0)
)
print(
    "F1:",
    f1_score(y_val, val_pred, zero_division=0)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred))

print("\nTotal transactions flagged:", val_pred.sum())

print("\nStep 4.3 completed successfully!")
