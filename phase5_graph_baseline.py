
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

# ==========================================
# FLOWPRINT - PHASE 5
# Step 5.3: Graph-Aware Baseline
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 5.3 - GRAPH-AWARE BASELINE")
print("=" * 55)

transaction_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

graph_features = [
    "sender_prev_out_count",
    "receiver_prev_in_count",
    "sender_prev_out_amount",
    "receiver_prev_in_amount"
]

# Load historical features
history = pd.read_csv(
    data_dir / "historical_graph_features.csv"
)

assert history["transaction_id"].is_unique

# Load chronological partitions
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

# Join using stable transaction IDs
train_df = train_df.merge(
    history,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

val_df = val_df.merge(
    history,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

# Check for missing historical features
assert train_df[graph_features].notna().all().all()
assert val_df[graph_features].notna().all().all()

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))

experiments = {
    "Transaction Only": transaction_features,
    "Transaction + Graph": transaction_features + graph_features
}

results = []

for name, features in experiments.items():

    print(f"\nTraining: {name}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(
        train_df[features],
        train_df["isMoneyLaundering"]
    )

    probabilities = model.predict_proba(
        val_df[features]
    )[:, 1]

    labels = val_df["isMoneyLaundering"]

    pr_auc = average_precision_score(
        labels, probabilities
    )

    roc_auc = roc_auc_score(
        labels, probabilities
    )

    precision, recall, thresholds = (
        precision_recall_curve(labels, probabilities)
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    results.append({
        "Model": name,
        "Features": len(features),
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Best F1": best_f1,
        "Best Threshold": best_threshold
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("GRAPH-AWARE BASELINE RESULTS")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nStep 5.3 completed successfully!")
