
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 23B: AML TYPOLOGY ANALYSIS")
print("=" * 65)

ORIGINAL = [
    "amount",
    "oldbalanceOrg",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
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

print("\n1. Loading test transactions...")

test = pd.read_csv(
    DATA / "comparison_test.csv"
)

original = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=[
        "isMoneyLaundering",
        "laundering_typology"
    ]
)

original["original_row_id"] = np.arange(
    len(original)
)

# Use original row IDs rather than dataframe
# positions to preserve exact test alignment.
test = test.drop(
    columns=["laundering_typology"],
    errors="ignore"
)

test = test.merge(
    original[
        ["original_row_id", "laundering_typology"]
    ],
    on="original_row_id",
    how="left",
    validate="one_to_one"
)

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

assert test["original_row_id"].is_unique
assert history["original_row_id"].is_unique
assert graph["original_row_id"].is_unique

test = test.merge(
    history,
    on="original_row_id",
    how="left",
    validate="one_to_one"
)

test = test.merge(
    graph,
    on="original_row_id",
    how="left",
    validate="one_to_one"
)

assert len(test) == 163500
assert not test[FEATURES].isna().any().any()

y = test["isMoneyLaundering"].astype(int)

assert int(y.sum()) == 162

print("Test transactions:", len(test))
print("AML transactions:", int(y.sum()))

print("\n2. Loading 17-feature model...")

model = joblib.load(
    RESULTS / "rf_17_feature_model.joblib"
)

threshold = 0.23755805600307625

probabilities = model.predict_proba(
    test[FEATURES]
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

test["aml_probability"] = probabilities
test["predicted_aml"] = predictions

tn, fp, fn, tp = confusion_matrix(
    y, predictions, labels=[0, 1]
).ravel()

print("\nOVERALL MODEL PERFORMANCE")
print("Threshold:", threshold)
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

print("Precision:", precision_score(y, predictions))
print("Recall:", recall_score(y, predictions))
print("F1:", f1_score(y, predictions))
print("PR-AUC:", average_precision_score(
    y, probabilities
))

# Typology analysis is restricted to actual
# AML transactions. Do not assign typologies
# to normal transactions or false positives.
aml = test[y == 1].copy()

if aml["laundering_typology"].isna().any():
    raise ValueError(
        "Some AML transactions have missing typology labels."
    )

aml["laundering_typology"] = (
    aml["laundering_typology"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\n3. TYPOLOGY-WISE DETECTION")
print("-" * 65)

rows = []

for typology, group in aml.groupby(
    "laundering_typology"
):

    total = len(group)
    detected = int(group["predicted_aml"].sum())
    missed = total - detected
    recall = detected / total

    rows.append({
        "Typology": typology,
        "Total_AML": total,
        "Detected": detected,
        "Missed": missed,
        "Recall": recall,
        "Mean_AML_Score": group[
            "aml_probability"
        ].mean(),
        "Median_AML_Score": group[
            "aml_probability"
        ].median()
    })

results = pd.DataFrame(rows).sort_values(
    "Total_AML",
    ascending=False
)

print(results.to_string(index=False))

assert results["Total_AML"].sum() == 162
assert results["Detected"].sum() == tp
assert results["Missed"].sum() == fn

results.to_csv(
    RESULTS / "typology_detection_results.csv",
    index=False
)

print("\n4. SAVING TRANSACTION-LEVEL RESULTS")

aml[
    [
        "original_row_id",
        "laundering_typology",
        "aml_probability",
        "predicted_aml"
    ]
].to_csv(
    RESULTS / "typology_transaction_predictions.csv",
    index=False
)

print("\nSTEP 23B COMPLETED SUCCESSFULLY")
