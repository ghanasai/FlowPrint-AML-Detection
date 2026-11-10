
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 10.2 - CONTROLLED ABLATION STUDY")
print("=" * 60)

df = pd.read_csv(
    DATA / "flowprint_research_features.csv"
)

transaction = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month"
]

historical = [
    "sender_prev_out_count",
    "receiver_prev_in_count",
    "sender_prev_out_amount",
    "receiver_prev_in_amount"
]

temporal = [
    "sender_gap_minutes",
    "receiver_gap_minutes",
    "sender_has_history",
    "receiver_has_history"
]

experiments = {
    "Transaction Only": transaction,
    "Transaction + Historical": transaction + historical,
    "Transaction + Temporal": transaction + temporal,
    "Full FlowPrint": transaction + historical + temporal
}

train = df[df["split"] == "train"]
val = df[df["split"] == "validation"]

y_train = train["isMoneyLaundering"]
y_val = val["isMoneyLaundering"]

results = []

for name, features in experiments.items():

    print(f"\nTraining: {name}")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    model.fit(train[features], y_train)

    scores = model.predict_proba(
        val[features]
    )[:, 1]

    pr_auc = average_precision_score(
        y_val, scores
    )

    roc_auc = roc_auc_score(
        y_val, scores
    )

    precision, recall, thresholds = (
        precision_recall_curve(y_val, scores)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = np.argmax(f1)

    results.append({
        "Model": name,
        "Features": len(features),
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Best Validation F1": f1[best_idx],
        "Best Threshold": thresholds[best_idx]
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CONTROLLED ABLATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

results_df.to_csv(
    RESULTS / "controlled_ablation_results.csv",
    index=False
)

print("\nSaved: controlled_ablation_results.csv")
print("Phase 10.2 completed successfully!")
