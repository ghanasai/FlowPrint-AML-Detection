
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 9.2 - FINAL TEST EVALUATION")
print("=" * 60)

transaction_features = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month"
]

history_features = [
    "sender_prev_out_count",
    "receiver_prev_in_count",
    "sender_prev_out_amount",
    "receiver_prev_in_amount"
]

temporal_features = [
    "sender_gap_minutes",
    "receiver_gap_minutes",
    "sender_has_history",
    "receiver_has_history"
]

all_features = (
    transaction_features
    + history_features
    + temporal_features
)

# Threshold selected during Phase 6 validation
FLOWPRINT_THRESHOLD = 0.45810797375502676

# Load chronological partitions
parts = []

for split in ["train", "validation", "test"]:
    part = pd.read_csv(
        DATA / f"{split}.csv",
        usecols=[
            "transaction_id", "timestamp",
            "nameOrig", "nameDest",
            "isMoneyLaundering",
            "laundering_typology"
        ] + transaction_features,
        parse_dates=["timestamp"]
    )
    part["split"] = split
    parts.append(part)

df = pd.concat(parts, ignore_index=True)

# Merge previously generated historical features
history = pd.read_csv(
    DATA / "historical_graph_features.csv"
)

df = df.merge(
    history,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

# Recreate corrected strictly historical temporal gaps
df = df.sort_values(
    ["timestamp", "transaction_id"]
).reset_index(drop=True)

for account_col, prefix in [
    ("nameOrig", "sender"),
    ("nameDest", "receiver")
]:
    unique_times = (
        df[[account_col, "timestamp"]]
        .drop_duplicates()
        .sort_values([account_col, "timestamp"])
        .copy()
    )

    unique_times["previous_time"] = (
        unique_times.groupby(account_col)["timestamp"]
        .shift(1)
    )

    df = df.merge(
        unique_times.rename(
            columns={
                "previous_time": f"{prefix}_previous_time"
            }
        ),
        on=[account_col, "timestamp"],
        how="left",
        validate="many_to_one",
        sort=False
    )

    gap = (
        df["timestamp"]
        - df[f"{prefix}_previous_time"]
    ).dt.total_seconds() / 60

    df[f"{prefix}_has_history"] = (
        gap.notna().astype(int)
    )

    df[f"{prefix}_gap_minutes"] = gap.fillna(-1)

    df.drop(
        columns=[f"{prefix}_previous_time"],
        inplace=True
    )

assert df["transaction_id"].is_unique
assert df[all_features].notna().all().all()

# Save the complete feature matrix for research experiments
feature_file = DATA / "flowprint_research_features.csv"

df.to_csv(feature_file, index=False)

print("Research feature matrix saved:", feature_file)

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

print("\nTraining samples:", len(train))
print("Test samples:", len(test))
print("Test AML transactions:", test["isMoneyLaundering"].sum())

experiments = {
    "Transaction RF (7)": transaction_features,
    "FlowPrint RF (15)": all_features
}

results = []

for name, features in experiments.items():
    print(f"\nTraining {name}...")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        train[features],
        train["isMoneyLaundering"]
    )

    scores = model.predict_proba(
        test[features]
    )[:, 1]

    # Same validation-selected threshold for FlowPrint.
    # Baseline uses a fixed 0.5 threshold.
    threshold = (
        FLOWPRINT_THRESHOLD
        if name == "FlowPrint RF (15)"
        else 0.5
    )

    predictions = (scores >= threshold).astype(int)

    y = test["isMoneyLaundering"]

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": name,
        "PR-AUC": average_precision_score(y, scores),
        "ROC-AUC": roc_auc_score(y, scores),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

    if name == "FlowPrint RF (15)":
        output = test[[
            "transaction_id",
            "timestamp",
            "nameOrig",
            "nameDest",
            "amount",
            "isMoneyLaundering",
            "laundering_typology"
        ]].copy()

        output["suspiciousness_score"] = scores
        output["predicted_suspicious"] = predictions

        output.to_csv(
            RESULTS / "final_test_predictions.csv",
            index=False
        )

        aml = output[
            output["isMoneyLaundering"] == 1
        ]

        typology = (
            aml.groupby("laundering_typology")
            .agg(
                total=("transaction_id", "count"),
                detected=("predicted_suspicious", "sum")
            )
            .reset_index()
        )

        typology["recall"] = (
            typology["detected"] / typology["total"]
        )

        print("\nTEST TYPOLOGY RECALL")
        print(typology.to_string(index=False))

        typology.to_csv(
            RESULTS / "final_test_typology.csv",
            index=False
        )

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL TEST COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS / "final_test_comparison.csv",
    index=False
)

print("\nFinal evaluation files saved.")
print("Step 9.2 completed successfully!")
