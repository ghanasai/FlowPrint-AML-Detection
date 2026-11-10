
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, confusion_matrix,
    average_precision_score, roc_auc_score
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"

print("=" * 60)
print("PHASE 8 - FLOWPRINT EVALUATION")
print("=" * 60)

# Load validation predictions
pred = pd.read_csv(
    RESULTS / "validation_suspicious_transactions.csv"
)

# Load ground-truth typologies
truth = pd.read_csv(
    DATA / "validation.csv",
    usecols=[
        "transaction_id",
        "laundering_typology"
    ]
)

df = pred.merge(
    truth,
    on="transaction_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == len(pred)
assert df["transaction_id"].is_unique

y_true = df["isMoneyLaundering"]
y_pred = df["predicted_suspicious"]
scores = df["suspiciousness_score"]

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

print("\nTRANSACTION DETECTION")
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print("True negatives:", tn)

print("Precision:", precision_score(
    y_true, y_pred, zero_division=0
))
print("Recall:", recall_score(
    y_true, y_pred, zero_division=0
))
print("F1:", f1_score(
    y_true, y_pred, zero_division=0
))
print("PR-AUC:", average_precision_score(
    y_true, scores
))
print("ROC-AUC:", roc_auc_score(
    y_true, scores
))

# Typology-specific recall
print("\nTYPOLOGY-SPECIFIC DETECTION")

aml = df[df["isMoneyLaundering"] == 1].copy()

typology_results = (
    aml.groupby("laundering_typology", dropna=False)
    .agg(
        total=("transaction_id", "count"),
        detected=("predicted_suspicious", "sum")
    )
    .reset_index()
)

typology_results["recall"] = (
    typology_results["detected"]
    / typology_results["total"]
)

print(typology_results.to_string(index=False))

# Evaluate recovery of ground-truth AML links
true_pairs = set(zip(
    aml["nameOrig"], aml["nameDest"]
))

detected_aml = aml[
    aml["predicted_suspicious"] == 1
]

recovered_pairs = set(zip(
    detected_aml["nameOrig"],
    detected_aml["nameDest"]
))

pair_recall = (
    len(recovered_pairs) / len(true_pairs)
    if true_pairs else 0
)

print("\nNETWORK RECOVERY")
print("Ground-truth AML account pairs:", len(true_pairs))
print("Recovered AML account pairs:", len(recovered_pairs))
print("Ground-truth pair recall:", pair_recall)

# Candidate network statistics
summary = pd.read_csv(
    RESULTS / "candidate_network_summary.csv"
)

print("\nRECONSTRUCTED NETWORKS")
print("Candidate networks:", len(summary))
print("Largest network accounts:", summary["accounts"].max())
print("Largest network transactions:",
      summary["transactions"].max())

# Save typology evaluation
typology_results.to_csv(
    RESULTS / "typology_detection_results.csv",
    index=False
)

print("\nSaved typology detection results.")
print("Phase 8 completed successfully!")
