
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
RESULTS.mkdir(parents=True, exist_ok=True)

ORIGINAL = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

GRAPH = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

FEATURES = ORIGINAL + GRAPH

print("STEP 18A: GRAPH FEATURE ABLATION")
print("=" * 55)

print("\nLoading comparison datasets...")

train = pd.read_csv(DATA / "comparison_train.csv")
val = pd.read_csv(DATA / "comparison_validation.csv")
test = pd.read_csv(DATA / "comparison_test.csv")

print("Loading graph features...")

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

assert graph["original_row_id"].is_unique

def merge_graph(frame, name):
    assert frame["original_row_id"].is_unique

    result = frame.merge(
        graph,
        on="original_row_id",
        how="left",
        validate="one_to_one"
    )

    assert len(result) == len(frame)
    assert not result[FEATURES].isna().any().any(), (
        f"Missing values in {name}"
    )

    return result

train = merge_graph(train, "training")
val = merge_graph(val, "validation")
test = merge_graph(test, "testing")

# Ensure the same transaction IDs are not
# present in multiple dataset splits.
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
print("Features:", len(FEATURES))
print("Training AML:", int(y_train.sum()))
print("Validation AML:", int(y_val.sum()))
print("Testing AML:", int(y_test.sum()))

print("\nTraining 12-feature Random Forest...")

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

print("\nSelecting threshold on validation data...")

val_probs = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

best_idx = int(np.argmax(f1_values))
threshold = float(thresholds[best_idx])

print("Best validation threshold:", threshold)
print("Validation F1:", float(f1_values[best_idx]))

print("\nEvaluating on test data...")

test_probs = model.predict_proba(X_test)[:, 1]
pred = (test_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, pred, labels=[0, 1]
).ravel()

metrics = {
    "Model": "Original + Graph RF",
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

print("\nGRAPH ABLATION RESULTS")
print("-" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

pd.DataFrame([metrics]).to_csv(
    RESULTS / "graph_ablation_results.csv",
    index=False
)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance", ascending=False
)

importance.to_csv(
    RESULTS / "graph_ablation_importance.csv",
    index=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

joblib.dump(
    model,
    RESULTS / "graph_ablation_12_features.joblib"
)

print("\nSTEP 18A COMPLETED SUCCESSFULLY")
