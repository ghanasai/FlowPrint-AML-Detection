
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
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

ORIGINAL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

HISTORICAL_FEATURES = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

FEATURES = ORIGINAL_FEATURES + HISTORICAL_FEATURES

print("Loading comparison datasets...")

train = pd.read_csv(DATA / "comparison_train.csv")
val = pd.read_csv(DATA / "comparison_validation.csv")
test = pd.read_csv(DATA / "comparison_test.csv")

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL_FEATURES
)

assert history["original_row_id"].is_unique

def merge_history(frame, name):
    merged = frame.merge(
        history,
        on="original_row_id",
        how="left",
        validate="one_to_one"
    )

    assert len(merged) == len(frame)
    assert not merged[HISTORICAL_FEATURES].isna().any().any(), (
        f"Missing historical features in {name}"
    )

    return merged

train = merge_history(train, "train")
val = merge_history(val, "validation")
test = merge_history(test, "test")

# Verify split separation
train_ids = set(train["original_row_id"])
val_ids = set(val["original_row_id"])
test_ids = set(test["original_row_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

X_train = train[FEATURES]
y_train = train["isMoneyLaundering"].astype(int)

X_val = val[FEATURES]
y_val = val["isMoneyLaundering"].astype(int)

X_test = test[FEATURES]
y_test = test["isMoneyLaundering"].astype(int)

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Testing rows:", len(test))
print("Total features:", len(FEATURES))

print("\nTraining historical-feature Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=5,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start

print("Training time:", round(training_time, 2), "seconds")

print("\nOptimizing threshold using validation data...")

val_probs = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(precision[:-1] + recall[:-1], 1e-12)
)

best_idx = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_idx])

print("Best validation threshold:", best_threshold)
print("Validation F1:", float(f1_values[best_idx]))

print("\nEvaluating on untouched test data...")

test_probs = model.predict_proba(X_test)[:, 1]
predictions = (test_probs >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

metrics = {
    "Model": "Historical Feature Random Forest",
    "Features": len(FEATURES),
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, predictions, zero_division=0
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
    "Training_Time": round(training_time, 2)
}

print("\nFINAL HISTORICAL MODEL RESULTS")
print("--------------------------------")

for key, value in metrics.items():
    print(f"{key}: {value}")

pd.DataFrame([metrics]).to_csv(
    RESULTS / "historical_model_results.csv",
    index=False
)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    RESULTS / "historical_feature_importance.csv",
    index=False
)

print("\nTOP FEATURE IMPORTANCES")
print(importance.to_string(index=False))

joblib.dump(
    model,
    RESULTS / "historical_random_forest.joblib"
)

print("\nModel and results saved successfully.")
