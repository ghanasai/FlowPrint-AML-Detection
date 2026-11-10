
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

train = pd.read_csv(DATA / "comparison_train.csv")
val = pd.read_csv(DATA / "comparison_validation.csv")
test = pd.read_csv(DATA / "comparison_test.csv")

# Confirm transaction-level separation
train_ids = set(train["original_row_id"])
val_ids = set(val["original_row_id"])
test_ids = set(test["original_row_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

X_train = train[FEATURES].fillna(0)
y_train = train["isMoneyLaundering"].astype(int)

X_val = val[FEATURES].fillna(0)
y_val = val["isMoneyLaundering"].astype(int)

X_test = test[FEATURES].fillna(0)
y_test = test["isMoneyLaundering"].astype(int)

print("\nTRAINING CLASS-WEIGHTED RANDOM FOREST")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start

print("Training time:", round(training_time, 2), "seconds")

# Choose threshold using validation data only
val_probs = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = (
    precision_recall_curve(y_val, val_probs)
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
)

best_idx = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_idx])

print("Optimized validation threshold:", best_threshold)
print("Validation F1:", float(f1_values[best_idx]))

# Evaluate on untouched test data
test_probs = model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

metrics = {
    "Model": "Class-weighted Random Forest",
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(
        y_test, test_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_pred, zero_division=0
    ),
    "PR-AUC": average_precision_score(
        y_test, test_probs
    ),
    "ROC-AUC": roc_auc_score(
        y_test, test_probs
    ),
    "TP": int(tp),
    "FP": int(fp),
    "FN": int(fn),
    "TN": int(tn),
    "Training time (s)": round(training_time, 2)
}

print("\nFINAL WEIGHTED MODEL RESULTS")
print("----------------------------")

for key, value in metrics.items():
    print(f"{key}: {value}")

pd.DataFrame([metrics]).to_csv(
    OUT / "weighted_rf_results.csv",
    index=False
)

joblib.dump(
    model,
    OUT / "weighted_random_forest.joblib"
)

print("\nExperiment completed.")
