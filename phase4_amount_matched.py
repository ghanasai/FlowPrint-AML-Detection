
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ==========================================
# FLOWPRINT - PHASE 4
# Step 4.6: Amount-Matched Evaluation
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 4.6 - AMOUNT-MATCHED EVALUATION")
print("=" * 55)

features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

columns = features + ["isMoneyLaundering"]

train_df = pd.read_csv(
    data_dir / "train.csv",
    usecols=columns
)

val_df = pd.read_csv(
    data_dir / "validation.csv",
    usecols=columns
)

# Train on the original training data
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining baseline...")
model.fit(
    train_df[features],
    train_df["isMoneyLaundering"]
)

# Separate validation classes
aml = val_df[
    val_df["isMoneyLaundering"] == 1
].copy()

normal = val_df[
    val_df["isMoneyLaundering"] == 0
].copy()

# Create logarithmic amount bins
# Works with positive transaction amounts
bin_edges = np.unique(
    np.concatenate([
        [0],
        np.logspace(0, 6, 25),
        [np.inf]
    ])
)

aml["amount_bin"] = pd.cut(
    aml["amount"], bins=bin_edges, include_lowest=True
)

normal["amount_bin"] = pd.cut(
    normal["amount"], bins=bin_edges, include_lowest=True
)

# Match normal transactions to AML transactions
# Within each amount bin, without replacement
matched_parts = []
rng = np.random.default_rng(42)

for amount_bin, aml_group in aml.groupby(
    "amount_bin", observed=True
):
    normal_group = normal[
        normal["amount_bin"] == amount_bin
    ]

    n = min(len(aml_group), len(normal_group))

    if n == 0:
        continue

    selected_aml = aml_group.sample(
        n=n, random_state=42
    )

    selected_normal_indices = rng.choice(
        normal_group.index.to_numpy(),
        size=n,
        replace=False
    )

    selected_normal = normal.loc[selected_normal_indices]

    matched_parts.extend([
        selected_aml,
        selected_normal
    ])

if not matched_parts:
    raise ValueError("No matching amount bins found.")

matched_df = pd.concat(
    matched_parts, ignore_index=True
)

print("\nMatched evaluation samples:", len(matched_df))
print("Matched AML:", matched_df["isMoneyLaundering"].sum())
print(
    "Matched normal:",
    (matched_df["isMoneyLaundering"] == 0).sum()
)

# Evaluate on matched validation data
probabilities = model.predict_proba(
    matched_df[features]
)[:, 1]

labels = matched_df["isMoneyLaundering"]

print("\nAMOUNT-MATCHED VALIDATION RESULTS")

print(
    "PR-AUC (Average Precision):",
    average_precision_score(labels, probabilities)
)

print(
    "ROC-AUC:",
    roc_auc_score(labels, probabilities)
)

print("\nStep 4.6 completed successfully!")
