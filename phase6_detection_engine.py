
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 55)
print("FLOWPRINT - TEMPORAL DETECTION ENGINE")
print("=" * 55)

transaction_cols = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month"
]

history_cols = [
    "sender_prev_out_count",
    "receiver_prev_in_count",
    "sender_prev_out_amount",
    "receiver_prev_in_amount"
]

temporal_cols = [
    "sender_gap_minutes",
    "receiver_gap_minutes",
    "sender_has_history",
    "receiver_has_history"
]

features = transaction_cols + history_cols + temporal_cols

# 1. Load chronological data
parts = []
for split in ["train", "validation", "test"]:
    df = pd.read_csv(
        DATA / f"{split}.csv",
        usecols=[
            "transaction_id", "timestamp",
            "nameOrig", "nameDest",
            "isMoneyLaundering"
        ] + transaction_cols,
        parse_dates=["timestamp"]
    )
    df["split"] = split
    parts.append(df)

df = pd.concat(parts, ignore_index=True)

# 2. Merge existing historical graph features
history = pd.read_csv(
    DATA / "historical_graph_features.csv"
)

df = df.merge(
    history,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

# 3. Recalculate strictly historical time gaps
df = df.sort_values(
    ["timestamp", "transaction_id"]
).reset_index(drop=True)

# For each account and timestamp, obtain the latest
# strictly earlier timestamp, excluding simultaneous events.
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

    previous = unique_times.rename(
        columns={"previous_time": f"{prefix}_previous_time"}
    )

    df = df.merge(
        previous,
        on=[account_col, "timestamp"],
        how="left",
        validate="many_to_one",
        sort=False
    )

    gap = (
        df["timestamp"] - df[f"{prefix}_previous_time"]
    ).dt.total_seconds() / 60

    df[f"{prefix}_has_history"] = (
        gap.notna().astype(int)
    )

    df[f"{prefix}_gap_minutes"] = gap.fillna(-1)

    df.drop(
        columns=[f"{prefix}_previous_time"],
        inplace=True
    )

# 4. Check features
assert df["transaction_id"].is_unique
assert df[features].notna().all().all()

train = df[df["split"] == "train"]
val = df[df["split"] == "validation"]

print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Total features:", len(features))

# 5. Train model
print("\nTraining Random Forest...")

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

# 6. Validation predictions
prob = model.predict_proba(val[features])[:, 1]
y = val["isMoneyLaundering"]

pr_auc = average_precision_score(y, prob)
roc_auc = roc_auc_score(y, prob)

precision, recall, thresholds = (
    precision_recall_curve(y, prob)
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

print("\n" + "=" * 55)
print("TEMPORAL DETECTION RESULTS")
print("=" * 55)

print("PR-AUC:", round(pr_auc, 6))
print("ROC-AUC:", round(roc_auc, 6))
print("Best Validation F1:", round(f1[best_idx], 6))
print("Best Threshold:", best_threshold)

# 7. Save validation predictions
output = val[[
    "transaction_id",
    "timestamp",
    "nameOrig",
    "nameDest",
    "amount",
    "isMoneyLaundering"
]].copy()

output["suspiciousness_score"] = prob
output["predicted_suspicious"] = (
    prob >= best_threshold
).astype(int)

output.to_csv(
    RESULTS / "validation_suspicious_transactions.csv",
    index=False
)

joblib.dump(
    model,
    RESULTS / "flowprint_temporal_model.joblib"
)

print("\nFlagged transactions:",
      output["predicted_suspicious"].sum())

print("\nModel and validation predictions saved.")
print("Step 6.2 completed successfully!")
