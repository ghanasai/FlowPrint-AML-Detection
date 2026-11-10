
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

ORIGINAL = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month"
]

HISTORICAL = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

GRAPH = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

FEATURES = ORIGINAL + HISTORICAL + GRAPH

print("STEP 17: GRAPH-ENHANCED RANDOM FOREST")
print("=" * 55)

print("\nLoading original comparison datasets...")

train = pd.read_csv(DATA / "comparison_train.csv")
val = pd.read_csv(DATA / "comparison_validation.csv")
test = pd.read_csv(DATA / "comparison_test.csv")

print("Loading corrected historical features...")

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

print("Loading graph features...")

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

assert history["original_row_id"].is_unique
assert graph["original_row_id"].is_unique

def add_features(frame, name):
    result = frame.merge(
        history,
        on="original_row_id",
        how="left",
        validate="one_to_one"
    )

    result = result.merge(
        graph,
        on="original_row_id",
        how="left",
        validate="one_to_one"
    )

    assert len(result) == len(frame)
    assert not result[FEATURES].isna().any().any(), (
        f"Missing features in {name}"
    )

    return result

train = add_features(train, "train")
val = add_features(val, "validation")
test = add_features(test, "test")

# Confirm transaction-level split separation.
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

print("\nDATASET DETAILS")
print("Training:", len(train))
print("Validation:", len(val))
print("Testing:", len(test))
print("Total features:", len(FEATURES))
print("Training AML:", int(y_train.sum()))
print("Validation AML:", int(y_val.sum()))
print("Testing AML:", int(y_test.sum()))

print("\nTraining 18-feature Random Forest...")

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

print("\nOptimizing threshold on validation set...")

val_probs = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(precision[:-1] + recall[:-1], 1e-12)
)

best_idx = int(np.argmax(f1_values))
threshold = float(thresholds[best_idx])

print("Best validation threshold:", threshold)
print("Validation F1:", float(f1_values[best_idx]))

print("\nEvaluating on untouched test set...")

test_probs = model.predict_proba(X_test)[:, 1]
pred = (test_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, pred, labels=[0, 1]
).ravel()

metrics = {
    "Model": "Historical + Graph Random Forest",
    "Features": len(FEATURES),
    "Threshold": threshold,
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(
        y_test, pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, pred, zero_division=0
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

print("\nFINAL GRAPH MODEL RESULTS")
print("-" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

pd.DataFrame([metrics]).to_csv(
    RESULTS / "graph_model_results.csv",
    index=False
)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    RESULTS / "graph_feature_importance.csv",
    index=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

joblib.dump(
    model,
    RESULTS / "graph_random_forest_18_features.joblib"
)

print("\nGraph model experiment completed successfully.")
