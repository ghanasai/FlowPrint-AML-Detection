
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
# FLOWPRINT - PHASE 4
# Step 4.4: Feature Ablation
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 4.4 - FEATURE ABLATION")
print("=" * 55)

all_features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

train_df = pd.read_csv(
    data_dir / "train.csv",
    usecols=all_features + ["isMoneyLaundering"]
)

val_df = pd.read_csv(
    data_dir / "validation.csv",
    usecols=all_features + ["isMoneyLaundering"]
)

y_train = train_df["isMoneyLaundering"]
y_val = val_df["isMoneyLaundering"]

feature_sets = {
    "All Features": all_features,

    "Without Balances": [
        "amount",
        "hour",
        "day_of_week",
        "day_of_month",
        "month"
    ],

    "Amount Only": [
        "amount"
    ]
}

results = []

for experiment, features in feature_sets.items():

    print(f"\nTraining: {experiment}")

    X_train = train_df[features]
    X_val = val_df[features]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(
        y_val, probabilities
    )

    roc_auc = roc_auc_score(
        y_val, probabilities
    )

    precision, recall, thresholds = (
        precision_recall_curve(y_val, probabilities)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_f1 = np.max(f1)

    results.append({
        "Experiment": experiment,
        "Features": len(features),
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Best Validation F1": best_f1
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 55)
print("FEATURE ABLATION RESULTS")
print("=" * 55)

print(results_df.to_string(index=False))

print("\nStep 4.4 completed successfully!")
