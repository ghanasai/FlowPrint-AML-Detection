
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 9 - OPEN-WORLD TYPOLOGY EVALUATION")
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

features = (
    transaction_features
    + history_features
    + temporal_features
)

# Load data and historical features
history = pd.read_csv(
    DATA / "historical_graph_features.csv"
)

parts = []

for split in ["train", "validation"]:
    df = pd.read_csv(
        DATA / f"{split}.csv",
        usecols=[
            "transaction_id", "timestamp",
            "nameOrig", "nameDest",
            "isMoneyLaundering",
            "laundering_typology"
        ] + transaction_features,
        parse_dates=["timestamp"]
    )

    df["split"] = split
    parts.append(df)

df = pd.concat(parts, ignore_index=True)

df = df.merge(
    history,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

# Recreate the corrected temporal features
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

assert df[features].notna().all().all()

train = df[df["split"] == "train"]
val = df[df["split"] == "validation"]

typologies = [
    "integration",
    "layering",
    "structuring"
]

results = []

for held_out in typologies:

    print(f"\nTesting unseen typology: {held_out}")

    # Exclude held-out AML typology from training
    train_subset = train[
        ~(
            (train["isMoneyLaundering"] == 1)
            & (train["laundering_typology"] == held_out)
        )
    ]

    # Evaluate on normal transactions and unseen AML only
    val_subset = val[
        (val["isMoneyLaundering"] == 0)
        | (
            (val["isMoneyLaundering"] == 1)
            & (val["laundering_typology"] == held_out)
        )
    ]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        train_subset[features],
        train_subset["isMoneyLaundering"]
    )

    y_true = val_subset["isMoneyLaundering"]

    scores = model.predict_proba(
        val_subset[features]
    )[:, 1]

    pr_auc = average_precision_score(
        y_true, scores
    )

    roc_auc = roc_auc_score(
        y_true, scores
    )

    results.append({
        "Held-out Typology": held_out,
        "Unseen AML Samples": int(y_true.sum()),
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("OPEN-WORLD EVALUATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS / "open_world_results.csv",
    index=False
)

print("\nSaved open-world evaluation results.")
print("Phase 9.1 completed successfully!")
