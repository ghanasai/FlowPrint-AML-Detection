
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 19A: TEMPORAL ROBUSTNESS EVALUATION")
print("=" * 60)

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

print("\nLoading test transactions...")

test = pd.read_csv(DATA / "comparison_test.csv")

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

# Timestamp is taken from the original
# chronological feature extraction.
timestamps = pd.read_csv(
    DATA / "historical_features.csv",
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

for name, frame in [
    ("test", test),
    ("history", history),
    ("graph", graph),
    ("timestamps", timestamps)
]:
    assert frame["original_row_id"].is_unique, (
        f"Duplicate IDs in {name}"
    )

test = test.merge(
    history, on="original_row_id",
    validate="one_to_one"
)

test = test.merge(
    graph, on="original_row_id",
    validate="one_to_one"
)


# Avoid duplicate timestamp columns during merging.
# Use the timestamp from historical_features.csv.

test = test.drop(
    columns=["timestamp", "timestamp_x", "timestamp_y"],
    errors="ignore"
)

test = test.merge(
    timestamps,
    on="original_row_id",
    how="left",
    validate="one_to_one"
)

assert "timestamp" in test.columns
assert test["timestamp"].notna().all(), (
    "Missing timestamps after merging"
)

print("Timestamp column verified successfully.")


test = test.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

assert len(test) == 163500
assert not test[
    ORIGINAL + HISTORICAL + GRAPH
].isna().any().any()

# Three chronological windows of equal
# transaction count.
test["period"] = np.repeat(
    [1, 2, 3],
    [54500, 54500, 54500]
)

print("\nTEMPORAL PERIODS")

for period, group in test.groupby("period"):
    print(
        f"Period {period}: "
        f"{group['timestamp'].min()} to "
        f"{group['timestamp'].max()} | "
        f"Transactions={len(group)} | "
        f"AML={int(group['isMoneyLaundering'].sum())}"
    )

# Use previously trained models and
# previously selected validation thresholds.
#
# Confirm model filenames before running.
# Do not optimize thresholds on the test set.

MODELS = {
    "Original + Graph": {
        "path": RESULTS / "graph_ablation_12_features.joblib",
        "features": ORIGINAL + GRAPH,
        "threshold": 0.10125796094810109
    },
    "Historical + Graph": {
        "path": RESULTS / "graph_random_forest_18_features.joblib",
        "features": ORIGINAL + HISTORICAL + GRAPH,
        "threshold": 0.23443083292747288
    }
}

rows = []

for model_name, config in MODELS.items():

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(config["path"])

    for period, group in test.groupby("period"):

        X = group[config["features"]]
        y = group["isMoneyLaundering"].astype(int)

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= config["threshold"]
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        result = {
            "Model": model_name,
            "Period": period,
            "Start": group["timestamp"].min(),
            "End": group["timestamp"].max(),
            "Transactions": len(group),
            "AML": int(y.sum()),
            "Precision": precision_score(
                y, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y, predictions, zero_division=0
            ),
            "F1": f1_score(
                y, predictions, zero_division=0
            ),
            "PR_AUC": (
                average_precision_score(y, probabilities)
                if y.sum() > 0 else np.nan
            ),
            "ROC_AUC": (
                roc_auc_score(y, probabilities)
                if y.nunique() == 2 else np.nan
            ),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        }

        rows.append(result)

        print(
            f"Period {period} | "
            f"F1={result['F1']:.4f} | "
            f"Recall={result['Recall']:.4f} | "
            f"PR-AUC={result['PR_AUC']:.4f} | "
            f"TP={tp} FP={fp} FN={fn}"
        )

results = pd.DataFrame(rows)

output = RESULTS / "temporal_robustness_results.csv"
results.to_csv(output, index=False)

print("\nFINAL TEMPORAL RESULTS")
print("-" * 60)

print(
    results[
        [
            "Model", "Period", "AML",
            "Precision", "Recall", "F1",
            "PR_AUC", "TP", "FP", "FN"
        ]
    ].to_string(index=False)
)

print("\nSaved:", output)
print("\nSTEP 19A COMPLETED")
